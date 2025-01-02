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
        self.data_fold = "fold_0"
        self.data_root: str = f"working/dataset/Kvasir-SEG/folds/{self.data_fold}"
        self.dataloader: str = "KvasirSEGA"
        self.valid_type: str = "val"
        self.cvtColor = None

        self.name = self.model_type + "/" + self.encoder_model + self.decoder_model

        for key, value in kwargs.items():
            setattr(self, key, value)
