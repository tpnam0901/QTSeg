from configs.base import Config as BaseConfig


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.add_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_args(self, **kwargs):

        # fold_0, fold_1, fold_2, fold_3, fold_4
        self.num_classes = 2  # 2 or 3
        self.mask_type = "multiclass" if self.num_classes > 2 else "binary"

        self.data_fold = "fold_0"
        self.data_root: str = (
            f"working/dataset/BKAI/folds_{self.mask_type}/{self.data_fold}"
        )

        # BCELoss, FocalLoss, CrossEntropyLoss, BinaryDiceLoss
        self.loss_type = [
            "CrossEntropyLoss",
            "BinaryDiceLoss",
        ]  # BinaryDiceLoss for 2 masks, CategoricalDiceLoss for 3 masks

        # 2 - Binary, 3 - MultiBinary
        self.metric = "Binary"

        self.dataloader: str = "BKAIA"
        self.valid_type: str = "val"
        self.cvtColor = None

        self.name = (
            self.model_type
            + f"_n/{self.mask_type}/"
            + self.encoder_model
            + self.decoder_model
        )

        for key, value in kwargs.items():
            setattr(self, key, value)
