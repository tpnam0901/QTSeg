import os
import glob
import cv2
import numpy as np
import argparse
from tqdm.auto import tqdm


def main(args):

    target_paths = glob.glob(os.path.join(args.data_root, "*/*_mask_1.png"))
    special_case = os.path.join(args.data_root, "benign", "benign (195)_mask_2.png")

    for target_path in tqdm(target_paths):
        mask_1 = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        # Rename mask to mask_0
        if os.path.exists(target_path.replace("_mask_1", "_mask")):
            os.rename(
                target_path.replace("_mask_1", "_mask"),
                target_path.replace("_mask_1", "_mask_0"),
            )
        mask = cv2.imread(
            target_path.replace("_mask_1", "_mask_0"), cv2.IMREAD_GRAYSCALE
        )

        mask = np.where(mask_1 == 255, 255, mask)

        cv2.imwrite(target_path.replace("_mask_1", "_mask"), mask)

    # Special case
    mask_2 = cv2.imread(special_case, cv2.IMREAD_GRAYSCALE)
    mask_1 = cv2.imread(
        special_case.replace("_mask_2", "_mask_1"), cv2.IMREAD_GRAYSCALE
    )
    if os.path.exists(special_case.replace("_mask_2", "_mask")):
        os.rename(
            special_case.replace("_mask_2", "_mask"),
            special_case.replace("_mask_2", "_mask_0"),
        )
    mask = cv2.imread(special_case.replace("_mask_2", "_mask_0"), cv2.IMREAD_GRAYSCALE)
    mask = np.where(mask_2 == 255, 255, mask)
    mask = np.where(mask_1 == 255, 255, mask)
    cv2.imwrite(special_case.replace("_mask_2", "_mask"), mask)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        help="Path to data root",
        default="../working/dataset/BUSI/",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = arg_parser()
    main(args)
