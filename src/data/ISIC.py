import os
import logging

from configs.base import Config
from glob import glob
from .dataset import BaseDataset, AugDataset
from .utils import get_dataloader


def _load_ISIC(root_path, mode, prefix="_segmentation.png"):
    target_paths = glob(os.path.join(root_path, mode, "targets/*.png"))
    sample_paths = []
    for path in target_paths:
        path = path.replace("targets", "inputs")
        path = path.replace(prefix, ".jpg")
        sample_paths.append(path)

    for path, t_path in zip(sample_paths, target_paths):
        assert os.path.exists(path), "File not found: {} for the target {}".format(
            path, t_path
        )
    return sample_paths, target_paths


def _build_dataloader(
    cfg: Config,
    mode: str,
    logger=logging.getLogger(),
    batch_size: int = -1,
    aug: bool = False,
):
    load_fn = {
        "ISIC2016": _load_ISIC,
        "ISIC2016A": _load_ISIC,
        "ISIC2017": _load_ISIC,
        "ISIC2017A": _load_ISIC,
        "ISIC2018": _load_ISIC,
        "ISIC2018A": _load_ISIC,
    }
    prefixes = {
        "ISIC2016": "_Segmentation.png",
        "ISIC2016A": "_Segmentation.png",
        "ISIC2017": "_segmentation.png",
        "ISIC2017A": "_segmentation.png",
        "ISIC2018": "_segmentation.png",
        "ISIC2018A": "_segmentation.png",
    }
    sample_paths, target_paths = load_fn[cfg.dataloader](
        cfg.data_root, mode, prefixes[cfg.dataloader]
    )
    assert len(sample_paths) > 0, "No samples found in {}".format(
        os.path.join(cfg.data_root, mode, "targets")
    )

    logger.info(
        "Found {} samples in {}".format(
            len(target_paths), os.path.join(cfg.data_root, mode)
        )
    )

    batch_size = cfg.batch_size if batch_size < 1 else batch_size
    if aug:
        dataset = AugDataset(
            sample_paths,
            target_paths,
            cfg.img_size,
            cfg.num_classes,
            cvtColor=cfg.cvtColor,
        )
    else:
        dataset = BaseDataset(
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
    return _build_dataloader(cfg, mode, logger, batch_size, aug=False)


def build_aug_dataloader(
    cfg: Config,
    mode: str,
    logger=logging.getLogger(),
    batch_size: int = -1,
):
    return _build_dataloader(cfg, mode, logger, batch_size, aug=True)
