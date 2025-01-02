import logging

from configs.base import Config
from .ISIC import (
    build_dataloader as _build_ISIC,
    build_test_dataloader as _build_ISIC_test,
    build_aug_dataloader as _build_ISIC_aug,
)

from .BUSI import (
    build_dataloader as _build_BUSI,
    build_test_dataloader as _build_BUSI_test,
    build_aug_dataloader as _build_BUSI_aug,
)


from .BKAI import (
    build_dataloader as _build_BKAI,
    build_test_dataloader as _build_BKAI_test,
    build_aug_dataloader as _build_BKAI_aug,
)

from .KvasirSEG import (
    build_dataloader as _build_KvasirSeg,
    build_test_dataloader as _build_KvasirSeg_test,
    build_aug_dataloader as _build_KvasirSeg_aug,
)

from .DSB2018 import (
    build_dataloader as _build_DSB2018,
    build_test_dataloader as _build_DSB2018_test,
    build_aug_dataloader as _build_DSB2018_aug,
)


def build_dataloader(
    cfg: Config,
    mode: str,
    logger=logging.getLogger(),
    batch_size: int = -1,
):
    build_fn = {
        "ISIC2016": _build_ISIC,
        "ISIC2016A": _build_ISIC_aug,
        "ISIC2017": _build_ISIC,
        "ISIC2017A": _build_ISIC_aug,
        "ISIC2018": _build_ISIC,
        "ISIC2018A": _build_ISIC_aug,
        "BUSI": _build_BUSI,
        "BUSIA": _build_BUSI_aug,
        "BKAI": _build_BKAI,
        "BKAIA": _build_BKAI_aug,
        "KvasirSEG": _build_KvasirSeg,
        "KvasirSEGA": _build_KvasirSeg_aug,
        "DSB2018": _build_DSB2018,
        "DSB2018A": _build_DSB2018_aug,
    }

    if mode != "train":
        build_fn.update(
            {
                "ISIC2016A": _build_ISIC,
                "ISIC2017A": _build_ISIC,
                "ISIC2018A": _build_ISIC,
                "BUSIA": _build_BUSI,
                "BKAIA": _build_BKAI,
                "KvasirSEGA": _build_KvasirSeg,
                "DSB2018A": _build_DSB2018,
            }
        )

    return build_fn[cfg.dataloader](cfg, mode, logger, batch_size)


def build_test_dataloader(
    cfg: Config,
    mode: str,
    logger=logging.getLogger(),
):
    build_fn = {
        "ISIC2016": _build_ISIC_test,
        "ISIC2017": _build_ISIC_test,
        "ISIC2018": _build_ISIC_test,
        "BUSI": _build_BUSI_test,
        "BKAI": _build_BKAI_test,
        "KvasirSEG": _build_KvasirSeg_test,
        "DSB2018": _build_DSB2018_test,
    }
    if cfg.dataloader not in list(build_fn.keys()):
        cfg.dataloader = cfg.dataloader[:-1]
    return build_fn[cfg.dataloader](cfg, mode, logger)
