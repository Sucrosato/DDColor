"""Convert all images in a folder to grayscale via Lab L channel, output to <folder>_grey."""

import argparse
import os
import cv2


def main():
    parser = argparse.ArgumentParser(description='Convert images to grayscale via Lab L channel')
    parser.add_argument('--input_dir', type=str, required=True, help='Input folder')
    args = parser.parse_args()

    input_dir = os.path.normpath(args.input_dir)
    parent = os.path.dirname(input_dir)
    folder_name = os.path.basename(input_dir)
    output_dir = os.path.join(parent, f'{folder_name}_grey')
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(os.listdir(input_dir))
    count = 0
    for fname in files:
        src_path = os.path.join(input_dir, fname)
        img = cv2.imread(src_path)
        if img is None:
            continue
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        l_channel = lab[:, :, 0]
        cv2.imwrite(os.path.join(output_dir, fname), l_channel)
        count += 1

    print(f'Done: {count} images -> {output_dir}')


if __name__ == '__main__':
    main()
