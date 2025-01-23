import os
import logging

from configs.base import Config
from glob import glob
from .dataset import BaseDataset, AugDataset, TestDataset
from .utils import get_dataloader


def _load_BKAI(root_path, mode):
    sample_paths = glob(os.path.join(root_path, mode, "inputs", "*.jpeg"))
    target_paths = []
    for path in sample_paths:
        path = path.replace("inputs", "targets")
        path = path.replace("jpeg", "npy")
        target_paths.append(path)

    for path in target_paths:
        assert os.path.exists(path), f"Target not found: {path}"

    assert len(sample_paths) > 0 and len(target_paths) > 0
    return sample_paths, target_paths


def _build_dataloader(
    cfg: Config,
    mode: str,
    logger=logging.getLogger(),
    batch_size: int = -1,
    aug: bool = False,
    test: bool = False,
):
    load_fn = {
        "BKAI": _load_BKAI,
        "BKAIA": _load_BKAI,
    }
    sample_paths, target_paths = load_fn[cfg.dataloader](cfg.data_root, mode)
    assert len(sample_paths) > 0, "No samples found in {}".format(
        os.path.join(cfg.data_root, mode, "targets")
    )

    logger.info(
        "Found {} samples in {}".format(
            len(target_paths), os.path.join(cfg.data_root, mode)
        )
    )

    batch_size = cfg.batch_size if batch_size < 1 else batch_size
    dataset_fn = AugDataset if aug else BaseDataset
    if test:
        dataset_fn = TestDataset
    dataset = dataset_fn(
        sample_paths,
        target_paths,
        cfg.img_size,
        cfg.num_classes,
        cvtColor=cfg.cvtColor,
    )

    dataloader = get_dataloader(
        dataset, cfg.val_epoch_freq, cfg.num_workers, mode, batch_size
    )

    return dataloader


def build_dataloader(
    cfg: Config,
    mode: str,
    logger=logging.getLogger(),
    batch_size: int = -1,
):
    return _build_dataloader(cfg, mode, logger, batch_size, False)


def build_aug_dataloader(
    cfg: Config,
    mode: str,
    logger=logging.getLogger(),
    batch_size: int = -1,
):
    return _build_dataloader(cfg, mode, logger, batch_size, True)


def build_test_dataloader(
    cfg: Config,
    mode: str,
    logger=logging.getLogger(),
    batch_size: int = -1,
):
    return _build_dataloader(cfg, mode, logger, batch_size, test=True)
