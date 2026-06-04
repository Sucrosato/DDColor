"""Pair images from folders side by side for comparison.

Usage:
  # Two folders
  python scripts/compare_images.py --dir1 E:/work/Code/data/gen --dir2 E:/work/Code/data/gt
  # Output: results/<dir1>_compare/

  # Three folders
  python scripts/compare_images.py --dir1 E:/work/Code/data/gen --dir2 E:/work/Code/data/gt --dir3 E:/work/Code/data/grey
  # Output: results/<dir1>_compare/
  # Layout: dir3 | dir2 | dir1
"""

import argparse
import os
import cv2
import numpy as np

SEP_WIDTH = 4  # white separator width in pixels


def main():
    parser = argparse.ArgumentParser(description='Create side-by-side comparison images')
    parser.add_argument('--dir1', type=str, required=True, help='Folder 1 (rightmost)')
    parser.add_argument('--dir2', type=str, required=True, help='Folder 2 (middle)')
    parser.add_argument('--dir3', type=str, default=None, help='Folder 3 (leftmost, optional)')
    args = parser.parse_args()

    dir1 = os.path.normpath(args.dir1)
    dir2 = os.path.normpath(args.dir2)
    dirs = [dir2, dir1]  # default: dir2 left, dir1 right

    if args.dir3:
        dir3 = os.path.normpath(args.dir3)
        dirs.insert(0, dir3)

    output_dir = os.path.join('results', f'{os.path.basename(dir1)}_compare')
    os.makedirs(output_dir, exist_ok=True)

    files1 = sorted(os.listdir(dir1))
    found, missing = 0, {d: 0 for d in dirs}

    for fname in files1:
        path1 = os.path.join(dir1, fname)
        img1 = cv2.imread(path1)
        if img1 is None:
            continue

        # Load all source images
        parts = []
        all_ok = True
        for d in dirs:
            p = os.path.join(d, fname)
            img = cv2.imread(p)
            if img is None:
                missing[d] += 1
                all_ok = False
                break
            parts.append(img)

        if not all_ok:
            continue

        # Resize to same height
        h = max(p.shape[0] for p in parts)
        for i in range(len(parts)):
            if parts[i].shape[0] != h:
                w = int(parts[i].shape[1] * h / parts[i].shape[0])
                parts[i] = cv2.resize(parts[i], (w, h))

        # Concatenate with white separators
        sep = np.ones((h, SEP_WIDTH, 3), dtype=np.uint8) * 255
        combined = parts[0]
        for p in parts[1:]:
            combined = np.hstack([combined, sep, p])

        cv2.imwrite(os.path.join(output_dir, fname), combined)
        found += 1

    print(f'Done: {found} pairs -> {output_dir}')
    for d in dirs:
        if missing[d] > 0:
            print(f'  ({missing[d]} not found in {os.path.basename(d)})')


if __name__ == '__main__':
    main()
