"""
Standalone evaluation of a checkpoint on any validation set.

Usage:
  # Use the dataset specified in the YAML config:
  python scripts/eval_checkpoint.py \
    --opt experiments/ds-l_coco10k/train_ddcolor_dinov3_sdt_testlinux.yml \
    --weight experiments/ds-l_coco10k/models/net_g_10000.pth

  # Override dataset with CLI args (requires --dataroot_gt AND --meta_info):
  python scripts/eval_checkpoint.py \
    --opt experiments/ds-l_coco10k/train_ddcolor_dinov3_sdt_testlinux.yml \
    --weight experiments/ds-l_coco10k/models/net_g_10000.pth \
    --dataroot_gt E:/work/Code/data/COCO/val2017 \
    --meta_info data_list/coco_val_200.txt \
    --gt_size 256

  # Override with a folder (auto-generates meta_info from image files):
  python scripts/eval_checkpoint.py \
    --opt experiments/ds-l_coco10k/train_ddcolor_dinov3_sdt_testlinux.yml \
    --weight experiments/ds-l_coco10k/models/net_g_10000.pth \
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

from basicsr.data import build_dataloader, build_dataset
from basicsr.archs import build_network

from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.options import ordered_yaml, dict2str


def evaluate(opt_path, weight_path, dataroot_gt=None, meta_info=None, gt_size=None,
             use_folder=False, dataset_name=None):
    # ---- load yaml ----
    loader, _ = ordered_yaml()
    with open(opt_path, 'r') as f:
        opt = yaml.load(f, Loader=loader)

    # ---- override val dataset ----
    val_opt = opt['datasets']['val']

    if use_folder and dataroot_gt:
        # Generate temporary meta_info file from folder contents
        image_files = sorted(os.listdir(dataroot_gt))
        tmp_meta = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        for fname in image_files:
            tmp_meta.write(osp.join(dataroot_gt, fname) + '\n')
        tmp_meta.close()
        meta_info = tmp_meta.name
        print(f'Auto-generated meta_info ({len(image_files)} images): {meta_info}')

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

    val_opt['phase'] = 'val'  # required by build_dataloader
    val_dataset = build_dataset(val_opt)
    val_loader = build_dataloader(
        val_dataset, val_opt, num_gpu=1, dist=False, sampler=None, seed=0)
    logger.info(f'Validation images: {len(val_dataset)}')

    # ---- build model & load checkpoint ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net_g = build_network(opt['network_g'])
    net_g = net_g.to(device)

    ckpt = torch.load(weight_path, map_location=device)
    # handle both raw state_dict and wrapped checkpoints
    if 'params' in ckpt:
        net_g.load_state_dict(ckpt['params'], strict=True)
    elif 'params_ema' in ckpt:
        logger.info('Using EMA weights')
        net_g.load_state_dict(ckpt['params_ema'], strict=True)
    else:
        net_g.load_state_dict(ckpt, strict=True)

    net_g.eval()
    logger.info(f'Loaded weights from: {weight_path}')

    # ---- prepare FID ----
    metrics = opt['val'].get('metrics', {})
    compute_fid = 'fid' in metrics

    if compute_fid:
        from basicsr.metrics.custom_fid import (
            INCEPTION_V3_FID, get_activations,
            calculate_activation_statistics, calculate_frechet_distance
        )
        incep_path = 'pretrain/inception_v3_google-1a9a5a14.pth'
        incep_sd = torch.load(incep_path, map_location='cpu')
        block_idx = INCEPTION_V3_FID.BLOCK_INDEX_BY_DIM[2048]
        inception = INCEPTION_V3_FID(incep_sd, [block_idx]).to(device)
        inception.eval()
        fake_acts_list, real_acts_list = [], []
        real_mu, real_sigma = None, None

    # ---- prepare CF / PSNR ----
    from basicsr.metrics import calculate_metric
    from basicsr.metrics.colorfulness import calculate_cf
    metric_results = {m: 0 for m in metrics if m != 'fid'}
    delta_cf_sum = 0.0

    # ---- loop ----
    pbar = tqdm(total=len(val_loader), unit='img')
    from basicsr.utils.img_util import tensor_lab2rgb

    for val_data in val_loader:
        lq = val_data['lq'].to(device)

        # Reconstruct lq_rgb: L channel + zero ab → Lab → RGB
        lq_rgb = tensor_lab2rgb(torch.cat([lq, torch.zeros_like(lq), torch.zeros_like(lq)], dim=1))

        with torch.no_grad():
            output_ab = net_g(lq_rgb)
            output_lab = torch.cat([lq, output_ab], dim=1)

        output_rgb = tensor_lab2rgb(output_lab)

        # ground truth: Lab → RGB
        gt_rgb = tensor_lab2rgb(torch.cat([lq, val_data['gt'].to(device)], dim=1))

        # to numpy for metrics
        sr_img = tensor2img([output_rgb])
        gt_img = tensor2img([gt_rgb])

        # FID
        if compute_fid:
            pred_t = output_rgb.to(device)
            gt_t = gt_rgb.to(device)
            fake_act = get_activations(pred_t, inception, 1)
            fake_acts_list.append(fake_act)
            if real_mu is None:
                real_act = get_activations(gt_t, inception, 1)
                real_acts_list.append(real_act)

        # CF / PSNR
        metric_data = {'img': sr_img, 'img2': gt_img}
        for name in metric_results:
            metric_results[name] += calculate_metric(metric_data, metrics[name])

        # ΔCF
        cf_fake = calculate_cf(sr_img)
        cf_real = calculate_cf(gt_img)
        delta_cf_sum += abs(cf_fake - cf_real)

        pbar.update(1)
    pbar.close()

    # ---- finalize ----
    # average non-FID metrics
    for name in metric_results:
        metric_results[name] /= len(val_loader)

    # average ΔCF
    metric_results['ΔCF'] = delta_cf_sum / len(val_loader)

    # compute FID
    if compute_fid:
        if real_mu is None:
            real_acts_all = np.concatenate(real_acts_list, 0)
            real_mu, real_sigma = calculate_activation_statistics(real_acts_all)
        fake_acts_all = np.concatenate(fake_acts_list, 0)
        fake_mu, fake_sigma = calculate_activation_statistics(fake_acts_all)
        metric_results['fid'] = calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)

    # ---- report ----
    print('\n' + '=' * 55)
    print(f'Checkpoint: {weight_path}')
    print(f'Val dataset: {val_opt["name"]} ({len(val_dataset)} images)')
    print('-' * 55)
    for metric, value in metric_results.items():
        print(f'  {metric.upper():>6s}: {value:.4f}')
    print('=' * 55)

    return metric_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a DDColor checkpoint')
    parser.add_argument('--opt', type=str, required=True,
                        help='Path to training YAML config (used for model architecture + val config)')
    parser.add_argument('--weight', type=str, required=True,
                        help='Path to .pth checkpoint')
    parser.add_argument('--dataroot_gt', type=str, default=None,
                        help='Override val dataset image directory')
    parser.add_argument('--meta_info', type=str, default=None,
                        help='Override val dataset meta_info_file')
    parser.add_argument('--gt_size', type=int, default=None,
                        help='Override val dataset gt_size')
    parser.add_argument('--name', type=str, default=None,
                        help='Override val dataset name (for display)')
    parser.add_argument('--folder', action='store_true',
                        help='Treat --dataroot_gt as a folder of images (auto-generate meta_info)')

    args = parser.parse_args()
    evaluate(args.opt, args.weight,
             dataroot_gt=args.dataroot_gt,
             meta_info=args.meta_info,
             gt_size=args.gt_size,
             use_folder=args.folder,
             dataset_name=args.name)

