import torch
import torch.nn as nn
import torch.nn.functional as F

from . import modules
from .encoder import models as encoder
from .decoder import models as decoder
from configs.base import Config


class BaseNet(nn.Module):
    def freeze_encoder(self):
        pass

    def unfreeze_encoder(self):
        pass

    def freeze_decoder(self):
        pass

    def unfreeze_decoder(self):
        pass

    def freeze_model(self):
        pass

    def unfreeze_model(self):
        pass


class QTSeg(BaseNet):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = getattr(encoder, cfg.encoder_model)(cfg)
        self.image_encoder.to("cpu")

        if cfg.encoder_pretrained:
            self.image_encoder.load_state_dict(
                torch.load(
                    cfg.encoder_pretrained,
                    map_location="cpu",
                ),
                strict=False,
            )

        self.mask_decoder = getattr(decoder, cfg.decoder_model)(cfg)
        self.mask_decoder.to("cpu")

        if cfg.decoder_pretrained:
            self.mask_decoder.load_state_dict(
                torch.load(
                    cfg.decoder_pretrained,
                    map_location="cpu",
                ),
                strict=False,
            )
        self.bridge = getattr(modules, cfg.bridge_model)(cfg)

    def forward(self, inputs):
        image_features = self.bridge(self.image_encoder(inputs))
        low_res_masks, mask_embeddings = self.mask_decoder(image_features)
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(inputs.shape[2], inputs.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        return ori_res_masks, mask_embeddings

    def freeze_encoder(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = True

    def freeze_decoder(self):
        for param in self.mask_decoder.parameters():
            param.requires_grad = False

    def unfreeze_decoder(self):
        for param in self.mask_decoder.parameters():
            param.requires_grad = True
