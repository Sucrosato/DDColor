"""Resize all images in a folder to 299x299, output to <folder>_resized."""

import argparse
import os
import cv2


def main():
    parser = argparse.ArgumentParser(description='Resize all images in a folder to 299x299')
    parser.add_argument('--input_dir', type=str, required=True, help='Input folder')
    args = parser.parse_args()

    input_dir = os.path.normpath(args.input_dir)
    parent = os.path.dirname(input_dir)
    folder_name = os.path.basename(input_dir)
    output_dir = os.path.join(parent, f'{folder_name}_resized')
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(os.listdir(input_dir))
    count = 0
    for fname in files:
        src_path = os.path.join(input_dir, fname)
        img = cv2.imread(src_path)
        if img is None:
            continue
        resized = cv2.resize(img, (299, 299), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(output_dir, fname), resized)
        count += 1

    print(f'Done: {count} images -> {output_dir}')


if __name__ == '__main__':
    main()
