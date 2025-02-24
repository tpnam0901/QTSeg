from configs.base import Config as BaseConfig


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.add_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_args(self, **kwargs):

        self.batch_size = 4

        self.optimizer: str = "sgd"
        self.scheduler: str = "PolyLR"
        self.learning_rate: float = 0.01
        self.weight_decay: float = 3e-05
        self.nesterov: bool = True

        self.img_size: int = 1024
        self.data_root: str = f"working/dataset/FIVES"
        self.dataloader: str = "FIVESA"
        self.valid_type: str = "test"
        self.cvtColor = None

        self.name = self.model_type + "/" + self.encoder_model + self.decoder_model

        for key, value in kwargs.items():
            setattr(self, key, value)
