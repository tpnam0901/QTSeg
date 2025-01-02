import os
import logging

from configs.base import Config
from glob import glob
from .dataset import BaseDataset, AugDataset
from .utils import get_dataloader


def _load_BUSI(root_path, mode):
    sample_paths = glob(os.path.join(root_path, mode, "inputs", "*.png"))
    target_paths = []
    for path in sample_paths:
        filename = os.path.basename(path).replace(".png", "_mask.png")
        path = os.path.join(root_path, mode, "targets", filename)
        target_paths.append(path)

    for path, t_path in zip(target_paths.copy(), sample_paths.copy()):
        if not os.path.exists(path):
            target_paths.remove(path)
            sample_paths.remove(t_path)

    assert len(sample_paths) == len(target_paths)
    assert len(sample_paths) > 0 and len(target_paths) > 0
    return sample_paths, target_paths


def _build_dataloader(
    cfg: Config,
    mode: str,
    logger=logging.getLogger(),
    batch_size: int = -1,
    aug: bool = False,
):
    load_fn = {
        "BUSI": _load_BUSI,
        "BUSIA": _load_BUSI,
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
    return _build_dataloader(cfg, mode, logger, batch_size, False)


def build_aug_dataloader(
    cfg: Config,
    mode: str,
    logger=logging.getLogger(),
    batch_size: int = -1,
):
    return _build_dataloader(cfg, mode, logger, batch_size, True)
