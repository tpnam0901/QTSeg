import torch.nn as nn
from .modules import Conv, C2f, Concat, SPPF
from configs.base import Config


class FPNEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        n = cfg.n_channel

        self.conv_0 = Conv(cfg.image_channel, n, 3, 2)
        self.conv_1 = Conv(n, n * 2, 3, 2)
        self.c2f_2 = C2f(n * 2, n * 2, 1, True)
        self.conv_3 = Conv(n * 2, n * 4, 3, 2)
        self.c2f_4 = C2f(n * 4, n * 4, 2, True)
        self.conv_5 = Conv(n * 4, n * 8, 3, 2)
        self.c2f_6 = C2f(n * 8, n * 8, 2, True)
        self.conv_7 = Conv(n * 8, n * 16, 3, 2)
        self.c2f_8 = C2f(n * 16, n * 16, 1, True)
        self.sppf_9 = SPPF(n * 16, n * 16, 5)
        self.upsample_10 = nn.modules.upsampling.Upsample(None, 2, "nearest")
        self.concat_11 = Concat(1)
        self.c2f_12 = C2f(n * 24, n * 8, 1)
        self.upsample_13 = nn.modules.upsampling.Upsample(None, 2, "nearest")
        self.concat_14 = Concat(1)
        self.c2f_15 = C2f(n * 12, n * 4, 1)
        self.conv_16 = Conv(n * 4, n * 4, 3, 2)
        self.concat_17 = Concat(1)
        self.c2f_18 = C2f(n * 12, n * 8, 1)
        self.conv_19 = Conv(n * 8, n * 8, 3, 2)
        self.concat_20 = Concat(1)
        self.c2f_21 = C2f(n * 24, n * 16, 1)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.c2f_2(x)
        x = self.conv_3(x)
        c2f_4 = self.c2f_4(x)
        x = self.conv_5(c2f_4)
        c2f_6 = self.c2f_6(x)
        x = self.conv_7(c2f_6)
        x = self.c2f_8(x)
        sppf_9 = self.sppf_9(x)
        x = self.upsample_10(sppf_9)
        x = self.concat_11([x, c2f_6])
        c2f_12 = self.c2f_12(x)
        x = self.upsample_13(c2f_12)
        x = self.concat_14([x, c2f_4])
        c2f_15 = self.c2f_15(x)
        x = self.conv_16(c2f_15)
        x = self.concat_17([x, c2f_12])
        c2f_18 = self.c2f_18(x)
        x = self.conv_19(c2f_18)
        x = self.concat_20([x, sppf_9])
        c2f_21 = self.c2f_21(x)

        return c2f_15, c2f_18, c2f_21
