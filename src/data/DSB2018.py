import os
import logging

from configs.base import Config
from glob import glob
from .dataset import BaseDataset, AugDataset
from .utils import (
    get_dataloader,
    load_img,
    preprocess,
    resize_mask,
    resize_longest_side,
    pad_to_square,
)
from typing import Tuple
import cv2
import numpy as np
import torch


class DSB2018Dataset(BaseDataset):
    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = load_img(self.X[index])
        y = load_img(self.y[index]).astype(np.int64)
        if x.shape[2] == 4:
            x = cv2.cvtColor(x, cv2.COLOR_RGBA2RGB)

        x_raw_rgb = x = resize_longest_side(
            x, self.img_size, interpolation=cv2.INTER_AREA
        )
        if self.cvtColor is not None:
            x = cv2.cvtColor(x, self.cvtColor)
        y = resize_mask(y, (x.shape[1], x.shape[0]))
        if np.max(y) > 1 and self.num_classes <= 2:
            y = y / np.max(y)
            if len(y.shape) == 3:
                y = y[:, :, 0]

        x = pad_to_square(x)
        x_raw_rgb = pad_to_square(x_raw_rgb)
        y = pad_to_square(y)

        if getattr(self, "augment_seg", None) is not None:
            x, y = self.augment_seg(x, y)
        # Standardization
        x = preprocess(x)
        assert len(y.shape) == 2
        return (
            torch.from_numpy(x),
            torch.from_numpy(y.astype(np.int64)),
            torch.from_numpy(x_raw_rgb.astype(np.float32)),
        )


class DSB2018TestDataset(BaseDataset):
    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_raw_rgb = x = load_img(self.X[index])
        y = load_img(self.y[index]).astype(np.int64)
        if x.shape[2] == 4:
            x = cv2.cvtColor(x, cv2.COLOR_RGBA2RGB)

        x = resize_longest_side(x, self.img_size, interpolation=cv2.INTER_AREA)
        if self.cvtColor is not None:
            x = cv2.cvtColor(x, self.cvtColor)
        if np.max(y) > 1 and self.num_classes <= 2:
            y = y / np.max(y)
            if len(y.shape) == 3:
                y = y[:, :, 0]

        x = pad_to_square(x)

        # Standardization
        x = preprocess(x)
        assert len(y.shape) == 2
        return (
            torch.from_numpy(x),
            torch.from_numpy(y.astype(np.int64)),
            torch.from_numpy(x_raw_rgb.astype(np.float32)),
        )


class DSB2018AugDataset(AugDataset, DSB2018Dataset):
    pass


def _load_DSB2018(root_path, mode):
    sample_paths = glob(os.path.join(root_path, mode, "inputs", "*.png"))
    target_paths = []
    for path in sample_paths:
        filename = os.path.basename(path)
        path = os.path.join(root_path, mode, "targets", filename)
        target_paths.append(path)

    assert len(sample_paths) == len(target_paths)
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
        "DSB2018": _load_DSB2018,
        "DSB2018A": _load_DSB2018,
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
    dataset_fn = DSB2018AugDataset if aug else DSB2018Dataset
    if test:
        dataset_fn = DSB2018TestDataset
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
