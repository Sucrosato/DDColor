"""Compute colorfulness scores for all images in a folder, save CSV and plot histogram.

Usage:
  python scripts/cf.py --input_dir E:/work/Code/data/imagenet-val5k --name imagenet_val5k
  python scripts/cf.py --input_dir E:/work/Code/data/ADE20Kval      --name ade20k_val
"""

import argparse
import os
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_cf(img):
    """Hasler & Suesstrunk colorfulness metric."""
    (B, G, R) = cv2.split(img.astype('float'))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    rbMean, rbStd = np.mean(rg), np.std(rg)
    ybMean, ybStd = np.mean(yb), np.std(yb)
    stdRoot = np.sqrt(rbStd ** 2 + ybStd ** 2)
    meanRoot = np.sqrt(rbMean ** 2 + ybMean ** 2)
    return stdRoot + 0.3 * meanRoot


def main():
    parser = argparse.ArgumentParser(description='Compute CF scores and plot distribution')
    parser.add_argument('--input_dir', type=str, required=True, help='Image folder')
    parser.add_argument('--name', type=str, required=True, help='Dataset name for output files')
    args = parser.parse_args()

    input_dir = os.path.normpath(args.input_dir)
    name = args.name

    # 1. Compute CF for all images
    files = sorted(os.listdir(input_dir))
    cf_data = []   # (filename, cf)
    cf_vals = []
    for fname in files:
        img = cv2.imread(os.path.join(input_dir, fname))
        if img is None:
            continue
        cf = calculate_cf(img)
        cf_data.append((fname, cf))
        cf_vals.append(cf)

    n = len(cf_vals)
    print(f'Images: {n}')
    print(f'Mean CF: {np.mean(cf_vals):.2f}')
    print(f'Std  CF: {np.std(cf_vals):.2f}')
    print(f'Min  CF: {np.min(cf_vals):.2f}')
    print(f'Max  CF: {np.max(cf_vals):.2f}')

    # 2. Save CSV
    csv_dir = os.path.join('results', 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f'{name}_cf.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'cf'])
        writer.writerows(cf_data)
    print(f'CSV saved: {csv_path}')

    # 3. Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(cf_vals, bins=80, edgecolor='white', alpha=0.8, color='steelblue')
    plt.axvline(np.mean(cf_vals), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(cf_vals):.2f}')
    plt.xlabel('Colorfulness (CF)', fontsize=13)
    plt.ylabel('Count', fontsize=13)
    plt.title(f'{name}  (n={n})', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plot_path = os.path.join(csv_dir, f'{name}_cf.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f'Plot saved: {plot_path}')


if __name__ == '__main__':
    main()
