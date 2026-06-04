"""Extract chrominance: set L to a fixed value, keep ab, convert back to RGB.

Usage:
  python scripts/chroma_only.py --input_dir E:/work/Code/data/some_folder
  python scripts/chroma_only.py --input_dir E:/work/Code/data/some_folder --luminance 60
  # Output: <input_dir>_chroma/
"""

import argparse
import os
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Remove luminance, keep chrominance only')
    parser.add_argument('--input_dir', type=str, required=True, help='Input folder')
    parser.add_argument('--luminance', type=float, default=70.0,
                        help='Fixed L value (0-100, default 70).')
    args = parser.parse_args()

    input_dir = os.path.normpath(args.input_dir)
    parent = os.path.dirname(input_dir)
    folder_name = os.path.basename(input_dir)
    output_dir = os.path.join(parent, f'{folder_name}_chroma')
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(os.listdir(input_dir))
    count = 0
    for fname in files:
        src_path = os.path.join(input_dir, fname)
        img = cv2.imread(src_path)
        if img is None:
            continue

        # BGR → LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
        # L 通道统一设为定值
        lab[:, :, 0] = args.luminance
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        # LAB → BGR
        result = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
        cv2.imwrite(os.path.join(output_dir, fname), result)
        count += 1

    print(f'Done: {count} images -> {output_dir}  (L={args.luminance})')


if __name__ == '__main__':
    main()
