import os
import glob
import cv2
import numpy as np
import argparse
from tqdm.auto import tqdm


def main(args):
    sample_paths = glob.glob(os.path.join(args.data_root, "stage1_train/*"))
    out_dir = os.path.join(args.data_root, "preprocessed")
    img_out_dir = os.path.join(out_dir, "inputs")
    mask_out_dir = os.path.join(out_dir, "targets")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)
    img_out_dir = os.path.abspath(img_out_dir)
    mask_out_dir = os.path.abspath(mask_out_dir)

    for path in tqdm(sample_paths):
        image_path = glob.glob(os.path.join(path, "images/*"))[0]
        target_paths = glob.glob(os.path.join(path, "masks/*"))

        # Link image to out_dir
        os.symlink(
            os.path.abspath(image_path),
            os.path.join(img_out_dir, os.path.basename(image_path)),
        )

        # Merge masks
        mask = cv2.imread(target_paths[0], cv2.IMREAD_UNCHANGED)
        for target_path in target_paths[1:]:
            mask = np.where(
                cv2.imread(target_path, cv2.IMREAD_UNCHANGED) == 255, 255, mask
            )

        # Write mask to out_dir
        cv2.imwrite(os.path.join(mask_out_dir, os.path.basename(image_path)), mask)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        help="Path to data root",
        default="../working/dataset/DSB2018/",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = arg_parser()
    main(args)
