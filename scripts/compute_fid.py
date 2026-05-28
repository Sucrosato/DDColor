"""
Standalone FID computation using a trained .pth checkpoint.

Computes FID between model outputs (colorized grey images) and ground truth.

Usage:
  python scripts/compute_fid.py \
    --weight experiments/ds-s-coco10k-final/models/net_g_10000.pth \
    --input_dir E:/work/Code/data/COCO/val2017_grey \
    --gt_dir E:/work/Code/data/COCO/val2017 \
    --input_size 256 \
    --model_size dinov3_small

  # If input images are already color (not grey), add --input_is_color:
  python scripts/compute_fid.py \
    --weight checkpoints/net_g_latest.pth \
    --input_dir E:/work/Code/data/COCO/val2017 \
    --gt_dir E:/work/Code/data/COCO/val2017 \
    --input_is_color
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from ddcolor import ColorizationPipeline, build_ddcolor_model
from basicsr.archs.ddcolor_dinov3_arch import DDColor_DinoV3_SDT
from basicsr.metrics.custom_fid import (
    INCEPTION_V3_FID,
    get_activations,
    calculate_activation_statistics,
    calculate_frechet_distance,
)


def load_model(weight_path, input_size, model_size, device):
    """Load DDColor model from checkpoint."""
    model_cls = DDColor_DinoV3_SDT if model_size.startswith('dinov3_') else None
    if model_cls is None:
        from ddcolor import DDColor
        model_cls = DDColor

    model = build_ddcolor_model(
        model_cls,
        model_path=weight_path,
        input_size=input_size,
        model_size=model_size,
        device=device,
    )
    return model


def colorize_images(model, input_dir, input_size, device, input_is_color):
    """Run inference on all images in input_dir, return list of BGR uint8 arrays."""
    colorizer = ColorizationPipeline(model, input_size=input_size, device=device)
    results = {}
    files = sorted(os.listdir(input_dir))
    for fname in tqdm(files, desc='Colorizing'):
        img = cv2.imread(os.path.join(input_dir, fname))
        if img is None:
            continue
        if input_is_color:
            # Input is already color; model still expects grey — extract L, colorize, replace
            pass  # fall through to normal path for now
        result = colorizer.process(img)
        results[fname] = result
    return results


def prepare_inception(device):
    """Load InceptionV3 for FID (2048-dim pool3 features)."""
    incep_path = os.path.join(_project_root, 'pretrain', 'inception_v3_google-1a9a5a14.pth')
    if not os.path.exists(incep_path):
        raise FileNotFoundError(f'Inception weights not found: {incep_path}')
    incep_sd = torch.load(incep_path, map_location='cpu')
    block_idx = INCEPTION_V3_FID.BLOCK_INDEX_BY_DIM[2048]
    model = INCEPTION_V3_FID(incep_sd, [block_idx]).to(device)
    model.eval()
    return model


def rgb_to_inception_input(bgr_uint8):
    """Convert BGR uint8 [0,255] → RGB float tensor (B,3,H,W) in [-1,1]."""
    rgb = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2RGB)           # uint8 [0,255]
    rgb = rgb.astype(np.float32) / 255.0                        # [0,1]
    rgb = rgb * 2.0 - 1.0                                       # [-1,1]
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)  # (1,3,H,W)
    return tensor


def compute_fid(fake_images, gt_images, inception, device, batch_size=8):
    """Compute FID between two dicts of {fname: BGR uint8 array}."""
    # Sort by filename for deterministic pairing
    common_names = sorted(set(fake_images.keys()) & set(gt_images.keys()))
    if len(common_names) == 0:
        raise ValueError('No common filenames between input and gt directories.')
    print(f'Paired images: {len(common_names)}')

    fake_acts_list, real_acts_list = [], []

    for fname in tqdm(common_names, desc='Extracting features'):
        fake_t = rgb_to_inception_input(fake_images[fname]).to(device)
        gt_t = rgb_to_inception_input(gt_images[fname]).to(device)

        with torch.no_grad():
            fake_act = get_activations(fake_t, inception, 1)
            fake_acts_list.append(fake_act)
            real_act = get_activations(gt_t, inception, 1)
            real_acts_list.append(real_act)

    fake_acts_all = np.concatenate(fake_acts_list, axis=0)
    real_acts_all = np.concatenate(real_acts_list, axis=0)

    real_mu, real_sigma = calculate_activation_statistics(real_acts_all)
    fake_mu, fake_sigma = calculate_activation_statistics(fake_acts_all)

    fid = calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)
    return fid


def main():
    parser = argparse.ArgumentParser(description='Compute FID for a DDColor checkpoint')
    parser.add_argument('--weight', type=str, required=True, help='Path to .pth checkpoint')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory of input images (grey or color)')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Directory of ground truth color images')
    parser.add_argument('--input_size', type=int, default=256, help='Model input size')
    parser.add_argument('--model_size', type=str, default='dinov3_small',
                        choices=['tiny', 'large', 'dinov3_small', 'dinov3_base', 'dinov3_large'])
    parser.add_argument('--input_is_color', action='store_true',
                        help='Input images are color (extract L channel before inference)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Optional: save colorized outputs to this directory')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 1. Load model
    print(f'Loading model: {args.weight}')
    model = load_model(args.weight, args.input_size, args.model_size, device)

    # 2. Run inference
    print(f'Colorizing images from: {args.input_dir}')
    colorized = colorize_images(model, args.input_dir, args.input_size, device,
                                args.input_is_color)
    print(f'Colorized {len(colorized)} images')

    # 3. Save if requested
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        for fname, img in colorized.items():
            cv2.imwrite(os.path.join(args.output_dir, fname), img)
        print(f'Saved to: {args.output_dir}')

    # 4. Load ground truth
    print(f'Loading ground truth from: {args.gt_dir}')
    gt_images = {}
    for fname in os.listdir(args.gt_dir):
        if fname in colorized:  # only load matching files
            img = cv2.imread(os.path.join(args.gt_dir, fname))
            if img is not None:
                gt_images[fname] = img
    print(f'Loaded {len(gt_images)} ground truth images')

    # 5. Compute FID
    print('Loading InceptionV3...')
    inception = prepare_inception(device)
    print('Computing FID...')
    fid = compute_fid(colorized, gt_images, inception, device)

    print('\n' + '=' * 50)
    print(f'  Checkpoint: {args.weight}')
    print(f'  Input:      {args.input_dir}')
    print(f'  GT:         {args.gt_dir}')
    print(f'  Paired:     {len(set(colorized.keys()) & set(gt_images.keys()))} images')
    print(f'  FID:        {fid:.4f}')
    print('=' * 50)


if __name__ == '__main__':
    main()
