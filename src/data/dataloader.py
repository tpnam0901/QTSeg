import logging

from configs.base import Config
from .ISIC import (
    build_dataloader as _build_ISIC,
    build_aug_dataloader as _build_ISIC_aug,
)

from .BUSI import (
    build_dataloader as _build_BUSI,
    build_aug_dataloader as _build_BUSI_aug,
)


from .BKAI import (
    build_dataloader as _build_BKAI,
    build_aug_dataloader as _build_BKAI_aug,
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
    }

    if mode != "train":
        build_fn.update(
            {
                "ISIC2016A": _build_ISIC,
                "ISIC2017A": _build_ISIC,
                "ISIC2018A": _build_ISIC,
                "BUSIA": _build_BUSI,
                "BKAIA": _build_BKAI,
            }
        )

    return build_fn[cfg.dataloader](cfg, mode, logger, batch_size)
