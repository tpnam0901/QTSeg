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
    fold_binary = os.path.abspath(os.path.join(args.data_root, "folds_binary"))
    fold_multiclass = os.path.abspath(os.path.join(args.data_root, "folds_multiclass"))
    if os.path.exists(fold_binary):
        raise ValueError("Folds already exist in {}".format(fold_binary))
    os.mkdir(fold_binary)
    if os.path.exists(fold_multiclass):
        raise ValueError("Folds already exist in {}".format(fold_multiclass))
    os.mkdir(fold_multiclass)

    targets_binary = glob.glob(
        os.path.join(args.data_root, "train_gt/train_gt_binary/*.npy")
    )
    targets_binary.sort()
    targets_multiclass = glob.glob(
        os.path.join(args.data_root, "train_gt/train_gt_multi/*.npy")
    )
    targets_multiclass.sort()
    sample_paths = [
        path.replace("train_gt_binary", "train")
        .replace("train_gt", "train")
        .replace(".npy", ".jpeg")
        for path in targets_binary
    ]
    assert len(targets_binary) > 0, "No binary targets found"
    assert len(targets_multiclass) > 0, "No multiclass targets found"
    assert (
        len(targets_binary) == len(targets_multiclass) == 1000
    ), "Incorrect number of samples"
    for i in range(len(sample_paths)):
        assert os.path.exists(sample_paths[i]), "Sample path does not exist"
    num_samples = len(sample_paths)
    num_test_samples = num_samples // args.num_folds
    create_fold(
        sample_paths, targets_binary, fold_binary, num_test_samples, num_samples
    )
    create_fold(
        sample_paths, targets_multiclass, fold_multiclass, num_test_samples, num_samples
    )


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        help="Path to data root",
        default="../working/dataset/BKAI/",
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
    assert (
        1000 % args.num_folds == 0
    ), "Number of samples (1000) must be divisible by number of folds"
    main(args)
