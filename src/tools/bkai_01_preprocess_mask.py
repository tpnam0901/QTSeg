import os
import glob
import cv2
import numpy as np
import argparse
from tqdm.auto import tqdm


def read_mask(mask_path):
    image = cv2.imread(mask_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 100, 20])
    upper1 = np.array([10, 255, 255])
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160, 100, 20])
    upper2 = np.array([179, 255, 255])
    lower_mask = cv2.inRange(image, lower1, upper1)
    upper_mask = cv2.inRange(image, lower2, upper2)

    red_mask = lower_mask + upper_mask
    red_mask[red_mask != 0] = 2

    # boundary RED color range values; Hue (36 - 70)
    green_mask = cv2.inRange(image, (36, 25, 25), (70, 255, 255))
    green_mask[green_mask != 0] = 1

    full_mask = cv2.bitwise_or(red_mask, green_mask)
    full_mask = full_mask.astype(np.uint8)
    return full_mask


def main(args):
    target_paths = glob.glob(os.path.join(args.data_root, "train_gt/train_gt/*.jpeg"))
    out_binary = os.path.join(args.data_root, "train_gt/train_gt_binary")
    out_multi = os.path.join(args.data_root, "train_gt/train_gt_multi")
    os.makedirs(out_binary, exist_ok=True)
    os.makedirs(out_multi, exist_ok=True)
    for target_path in tqdm(target_paths):
        mask = read_mask(target_path)
        binary_mask = (mask != 0).astype(np.uint8)
        mask_name = os.path.basename(target_path)
        np.save(os.path.join(out_binary, mask_name.replace("jpeg", "npy")), binary_mask)
        np.save(os.path.join(out_multi, mask_name.replace("jpeg", "npy")), mask)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        help="Path to data root",
        default="../working/dataset/BKAI/",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = arg_parser()
    main(args)
