"""
Evaluate a checkpoint using the exact same path as training-time validation.

Unlike eval_checkpoint.py which manually builds net_g and runs inference,
this script uses ColorModel.feed_data + ColorModel.test + same InceptionV3
FID pipeline as nondist_validation, maximally reproducing training val behavior.

Usage:
  python scripts/eval_by_model.py \
    --opt experiments/ds-s-coco10k-final/train_ddcolor_dinov3_sdt.yml \
    --weight experiments/ds-s-coco10k-final/models/net_g_10000.pth

  # Override val dataset:
  python scripts/eval_by_model.py \
    --opt experiments/.../xxx.yml \
    --weight experiments/.../net_g_10000.pth \
    --dataroot_gt E:/work/Code/data/COCO/val2017 \
    --folder
"""

import argparse
import logging
import os
import os.path as osp
import sys
import tempfile

import numpy as np
import torch
import yaml

from tqdm import tqdm

_project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, _project_root)

from basicsr.data import build_dataloader, build_dataset
from basicsr.archs import build_network
from basicsr.models.color_model import ColorModel
from basicsr.utils import get_root_logger
from basicsr.utils.options import ordered_yaml, dict2str
from basicsr.utils.img_util import tensor_lab2rgb
from basicsr.metrics.custom_fid import (
    INCEPTION_V3_FID,
    get_activations,
    calculate_activation_statistics,
    calculate_frechet_distance,
)
from basicsr.metrics import calculate_metric
from basicsr.metrics.colorfulness import calculate_cf
from basicsr.utils import tensor2img


def evaluate(opt_path, weight_path, dataroot_gt=None, meta_info=None, gt_size=None,
             use_folder=False, dataset_name=None):
    # ---- load yaml ----
    loader, _ = ordered_yaml()
    with open(opt_path, 'r') as f:
        opt = yaml.load(f, Loader=loader)

    # ---- override val dataset ----
    val_opt = opt['datasets']['val']

    if use_folder and dataroot_gt:
        image_files = sorted(os.listdir(dataroot_gt))
        tmp_meta = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        for fname in image_files:
            tmp_meta.write(osp.join(dataroot_gt, fname) + '\n')
        tmp_meta.close()
        meta_info = tmp_meta.name

    if dataroot_gt:
        val_opt['dataroot_gt'] = dataroot_gt
    if meta_info:
        val_opt['meta_info_file'] = meta_info
    if gt_size:
        val_opt['gt_size'] = gt_size
    if dataset_name:
        val_opt['name'] = dataset_name

    logger = get_root_logger(log_level=logging.INFO)
    logger.info(dict2str(opt))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- build ColorModel (same as training) ----
    # Add missing keys that options.py normally injects
    opt['is_train'] = True
    opt['dist'] = False
    opt['rank'] = 0
    opt['world_size'] = 1
    model = ColorModel(opt)

    # Load checkpoint weights into net_g (same as training val when no EMA)
    ckpt = torch.load(weight_path, map_location=device)
    if 'params_ema' in ckpt and hasattr(model, 'net_g_ema'):
        logger.info('Loading EMA weights')
        model.net_g_ema.load_state_dict(ckpt['params_ema'], strict=True)
    elif 'params' in ckpt:
        logger.info('Loading params into net_g')
        model.net_g.load_state_dict(ckpt['params'], strict=True)
    else:
        model.net_g.load_state_dict(ckpt, strict=True)
    model.net_g.eval()

    # ---- build val dataloader ----
    val_opt['phase'] = 'val'
    val_dataset = build_dataset(val_opt)
    val_loader = build_dataloader(
        val_dataset, val_opt, num_gpu=1, dist=False, sampler=None, seed=0)
    logger.info(f'Validation images: {len(val_dataset)}')

    # ---- prepare metrics (same as training) ----
    metrics = opt['val'].get('metrics', {})
    metric_results = {m: 0 for m in metrics if m != 'fid'}
    delta_cf_sum = 0.0
    compute_fid = 'fid' in metrics

    if compute_fid:
        incep_path = osp.join(_project_root, 'pretrain', 'inception_v3_google-1a9a5a14.pth')
        incep_sd = torch.load(incep_path, map_location='cpu')
        block_idx = INCEPTION_V3_FID.BLOCK_INDEX_BY_DIM[2048]
        inception = INCEPTION_V3_FID(incep_sd, [block_idx]).to(device)
        inception.eval()
        fake_acts_list, real_acts_list = [], []
        real_mu, real_sigma = None, None

    # ---- loop (same flow as nondist_validation) ----
    pbar = tqdm(total=len(val_loader), unit='img')

    for val_data in val_loader:
        # Same as feed_data (with is_train check fixing color_enhance bug)
        model.feed_data(val_data)
        # Same as test(): use net_g_ema
        model.test()

        visuals = model.get_current_visuals()
        sr_img = tensor2img([visuals['result']])
        gt_img = tensor2img([visuals['gt']])

        # FID (same as training)
        if compute_fid:
            pred_t = visuals['result'].to(device)
            gt_t = visuals['gt'].to(device)
            fake_act = get_activations(pred_t, inception, 1)
            fake_acts_list.append(fake_act)
            if real_mu is None:
                real_act = get_activations(gt_t, inception, 1)
                real_acts_list.append(real_act)

        # Other metrics (same as training)
        metric_data = {'img': sr_img, 'img2': gt_img}
        for name in metric_results:
            metric_results[name] += calculate_metric(metric_data, metrics[name])

        # ΔCF
        cf_fake = calculate_cf(sr_img)
        cf_real = calculate_cf(gt_img)
        delta_cf_sum += abs(cf_fake - cf_real)

        pbar.update(1)
    pbar.close()

    # ---- finalize (same as training) ----
    for name in metric_results:
        metric_results[name] /= len(val_loader)
    metric_results['∆CF'] = delta_cf_sum / len(val_loader)

    if compute_fid:
        if real_mu is None:
            real_acts_all = np.concatenate(real_acts_list, 0)
            real_mu, real_sigma = calculate_activation_statistics(real_acts_all)
        fake_acts_all = np.concatenate(fake_acts_list, 0)
        fake_mu, fake_sigma = calculate_activation_statistics(fake_acts_all)
        metric_results['fid'] = calculate_frechet_distance(
            real_mu, real_sigma, fake_mu, fake_sigma)

    # ---- report ----
    print('\n' + '=' * 55)
    print(f'Checkpoint: {weight_path}')
    print(f'Val dataset: {val_opt["name"]} ({len(val_dataset)} images)')
    print(f'Model weights: {"EMA" if "params_ema" in ckpt else "params"}')
    print('-' * 55)
    for metric, value in metric_results.items():
        print(f'  {metric.upper():>6s}: {value:.4f}')
    print('=' * 55)

    return metric_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate checkpoint using training val path')
    parser.add_argument('--opt', type=str, required=True,
                        help='Path to training YAML config')
    parser.add_argument('--weight', type=str, required=True,
                        help='Path to .pth checkpoint')
    parser.add_argument('--dataroot_gt', type=str, default=None,
                        help='Override val dataset image directory')
    parser.add_argument('--meta_info', type=str, default=None,
                        help='Override val dataset meta_info_file')
    parser.add_argument('--gt_size', type=int, default=None,
                        help='Override val dataset gt_size')
    parser.add_argument('--name', type=str, default=None,
                        help='Override val dataset name')
    parser.add_argument('--folder', action='store_true',
                        help='Treat --dataroot_gt as a folder (auto-generate meta_info)')

    args = parser.parse_args()
    evaluate(args.opt, args.weight,
             dataroot_gt=args.dataroot_gt,
             meta_info=args.meta_info,
             gt_size=args.gt_size,
             use_folder=args.folder,
             dataset_name=args.name)
