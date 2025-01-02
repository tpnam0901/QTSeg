import os
import glob
import argparse


def create_fold(input_paths, target_paths, fold_root, num_test_samples, num_samples):
    for i in range(args.num_folds):
        sub_fold_root = os.path.join(fold_root, "fold_{}".format(i))
        val_indices = list(range(i * num_test_samples, (i + 1) * num_test_samples))
        train_indices = list(set(range(num_samples)) - set(val_indices))
        val_input_paths = [input_paths[i] for i in val_indices]
        val_target_paths = [target_paths[i] for i in val_indices]
        train_input_paths = [input_paths[i] for i in train_indices]
        train_target_paths = [target_paths[i] for i in train_indices]
        os.makedirs(os.path.join(sub_fold_root, "train", "inputs"), exist_ok=True)
        os.makedirs(os.path.join(sub_fold_root, "train", "targets"), exist_ok=True)
        os.makedirs(os.path.join(sub_fold_root, "val", "inputs"), exist_ok=True)
        os.makedirs(os.path.join(sub_fold_root, "val", "targets"), exist_ok=True)
        for input_path, target_path in zip(train_input_paths, train_target_paths):
            os.symlink(
                os.path.abspath(input_path),
                os.path.join(
                    sub_fold_root, "train", "inputs", os.path.basename(input_path)
                ),
            )
            os.symlink(
                os.path.abspath(target_path),
                os.path.join(
                    sub_fold_root, "train", "targets", os.path.basename(target_path)
                ),
            )
        for input_path, target_path in zip(val_input_paths, val_target_paths):
            os.symlink(
                os.path.abspath(input_path),
                os.path.join(
                    sub_fold_root, "val", "inputs", os.path.basename(input_path)
                ),
            )
            os.symlink(
                os.path.abspath(target_path),
                os.path.join(
                    sub_fold_root, "val", "targets", os.path.basename(target_path)
                ),
            )
        print(
            "Fold {} created with {} train and {} val samples".format(
                i, len(train_input_paths), len(val_input_paths)
            )
        )


def main(args):
    fold_root = os.path.abspath(os.path.join(args.data_root, "folds"))
    if os.path.exists(fold_root):
        raise ValueError("Folds already exist in {}".format(fold_root))
    os.mkdir(fold_root)

    target_paths = glob.glob(os.path.join(args.data_root, "preprocessed/targets/*"))
    target_paths.sort()
    sample_paths = [path.replace("targets", "inputs") for path in target_paths]
    for i in range(len(sample_paths)):
        assert os.path.exists(sample_paths[i]), "Sample path does not exist"
    num_samples = len(sample_paths)
    num_test_samples = num_samples // args.num_folds
    create_fold(sample_paths, target_paths, fold_root, num_test_samples, num_samples)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        help="Path to data root",
        default="../working/dataset/DSB2018/",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        help="Number of folds",
        default=5,
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = arg_parser()
    # 2 last samples of benign is not used in test set
    assert (
        435 % args.num_folds == 0 and 210 % args.num_folds == 0
    ), "Number of samples (435, 210) must be divisible by number of folds"
    main(args)
