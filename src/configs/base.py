import logging
import os
from abc import ABC, abstractmethod
from typing import Tuple, List, Union
import importlib
import sys
import json


class Base(ABC):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def show(self):
        pass

    @abstractmethod
    def save(self, save_folder: str):
        pass

    @abstractmethod
    def load(self, cfg_path: str):
        pass


class BaseConfig(Base):
    def __init__(self, **kwargs):
        super(BaseConfig, self).__init__(**kwargs)

    def show(self):
        for key, value in self.__dict__.items():
            logging.info(f"{key}: {value}")

    def save(self, save_folder: str):
        os.makedirs(os.path.join(save_folder), exist_ok=True)
        out_cfg = os.path.join(save_folder, "cfg.json")
        with open(out_cfg, "w") as f:
            json.dump(self.get_params(), f, indent=4)

    def get_params(self):
        return self.__dict__

    def load(self, cfg_path: str):
        with open(cfg_path, "r") as f:
            data_dict = json.load(f)
        for key, value in data_dict.items():
            setattr(self, key, value)


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.name = "default"
        self.set_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_args(self, **kwargs):
        # --------------------------------- Training settings
        self.epochs: int = 350
        self.val_epoch_freq: int = 1
        self.transfer_epochs: int = 50
        self.batch_size: int = 32
        self.log_freq: int = 40
        self.checkpoint_dir: str = "working/checkpoints"
        self.ckpt_save_fred: int = 5000
        self.use_amp: bool = False

        # --------------------------------- Optim settings
        # sgd, adamw
        self.optimizer: str = "adamw"
        self.momentum: float = 0.99
        self.betas: Tuple[float, float] = (0.9, 0.999)
        self.eps: float = 1e-08
        self.amsgard: bool = False
        self.nesterov: bool = True

        # --------------------------------- Scheduler settings
        # StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts, IdentityScheduler, PolyLR
        self.scheduler: str = "StepLR"
        self.learning_rate: float = 0.001
        self.learning_rate_min: float = 0.00001
        self.weight_decay: float = 3e-05
        self.scheduler_last_epoch: int = -1
        # StepLR
        self.lr_step_size: int = 50
        self.lr_step_gamma: float = 0.5
        # MultiStepLR
        self.lr_milestones: List[int] = [50, 100, 150, 200]
        self.lr_multistep_gamma: float = 0.1
        # ExponentialLR
        self.lr_exp_gamma: float = 0.99
        # CosineAnnealingLR
        self.lr_T_max: int = 50
        self.lr_eta_min: float = 0.00001
        # ReduceLROnPlateau
        self.lr_plateau_mode: str = "min"
        self.lr_plateau_factor: float = 0.1
        self.lr_plateau_patience: int = 10
        self.lr_plateau_threshold: float = 0.0001
        self.lr_plateau_threshold_mode: str = "rel"
        self.lr_plateau_cooldown: int = 0
        self.lr_plateau_min_lr: float = 0
        self.lr_plateau_eps: float = 1e-08
        # CosineAnnealingWarmRestarts
        self.lr_T_0: int = 50
        self.lr_T_mult: int = 2
        self.lr_eta_min: float = 1e-6
        # IdentityScheduler - No params, update every step

        # --------------------------------- Model settings
        self.model_type: str = "QTSeg"
        self.model_pretrained: str = ""
        self.img_size: int = 512
        self.image_embedding_size: Tuple[int, int] = (
            self.img_size // 16,
            self.img_size // 16,
        )

        # ----------------- Encoder settings
        self.encoder_model: str = "FPNEncoder"
        self.encoder_pretrained: str = "networks/pretrained/fpn-nano.pth"
        self.encoder_out_features: List[int] = [64, 128, 256]
        self.image_channel: int = 3
        self.n_channel: int = 16

        # ----------------- Bridge settings
        self.bridge_model: str = "MLFD"

        # ----------------- Decoder settings
        self.num_classes: int = 2  # Num classes
        self.decoder_model: str = "MaskDecoder"
        self.decoder_pretrained: str = ""
        self.mask_depths: List[int] = [1, 2, 3]
        self.mask_num_head: int = 8
        self.mask_mlp_dim: int = 2048

        # --------------------------------- Loss & Metric settings
        # Binary, MultiBinary
        self.metric = "Binary"
        # BCELoss, FocalLoss, CrossEntropyLoss, BinaryDiceLoss, CategoricalDiceLoss
        self.loss_type: List[str] = ["CrossEntropyLoss", "BinaryDiceLoss"]
        self.loss_weight: List[float] = [1.0, 1.0]

        self.focal_alpha: float = 0.25
        self.focal_gamma: float = 2
        self.lambda_value: float = 0.5
        self.dice_smooth: float = 1e-6

        # --------------------------------- Dataset
        # ISIC2016, BUSI, BKAI
        self.scale_value: float = 255.0
        self.cvtColor: Union[int, None] = None
        self.data_root: str = "working/dataset/ISIC2016"
        self.dataloader: str = "ISIC2016"
        self.valid_type: str = "test"
        self.num_workers: int = 8
        # Only used in BKAI for determining the location of the mask
        self.mask_type: str = ""

        # This SEED will be replaced at runtime and saved in the checkpoint
        self.SEED: int = 42

        self.name = self.model_type + "/" + self.encoder_model + self.decoder_model
        for key, value in kwargs.items():
            setattr(self, key, value)


def import_config(
    path: str,
):
    """Get arguments for training and evaluate
    Returns:
        cfg: ArgumentParser
    """
    # Import config from path
    spec = importlib.util.spec_from_file_location("config", path)
    config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config
    spec.loader.exec_module(config)
    cfg = config.Config()
    return cfg
