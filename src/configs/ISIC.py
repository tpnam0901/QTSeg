import cv2
from configs.base import Config as BaseConfig


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.add_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_args(self, **kwargs):

        # ISIC2016, ISIC2017, ISIC2018
        self.data_root = "working/dataset/ISIC2016"
        self.dataloader = "ISIC2016"
        self.valid_type = "test"  # test for ISIC2016, val for ISIC2017, ISIC2018
        self.scale_value = 255.0
        self.cvtColor = cv2.COLOR_RGB2YCrCb

        self.name = (
            self.model_type + "_n" + "/" + self.encoder_model + self.decoder_model
        )

        for key, value in kwargs.items():
            setattr(self, key, value)
