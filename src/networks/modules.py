import torch.nn as nn
from .encoder.modules import Conv, Concat, ConvTranspose


class Identity(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x):
        return x


class MLFF(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        n = cfg.n_channel
        self.neck_0_0 = Conv(n * 4, n * 2, 1, 1)
        self.neck_0_1 = ConvTranspose(n * 8, n, 2, 2)
        self.neck_0_2 = ConvTranspose(n * 16, n, 4, 4)
        self.neck_0_concat = Concat(1)

        self.neck_1_0 = Conv(n * 4, n * 2, 3, 2)
        self.neck_1_1 = Conv(n * 8, n * 4, 1, 1)
        self.neck_1_2 = ConvTranspose(n * 16, n * 2, 2, 2)
        self.neck_1_concat = Concat(1)

        self.neck_2_0 = Conv(n * 4, n * 4, 3, 4)
        self.neck_2_1 = Conv(n * 8, n * 4, 3, 2)
        self.neck_2_2 = Conv(n * 16, n * 8, 1, 1)
        self.neck_2_concat = Concat(1)

    def forward(self, inputs):
        c2f_15, c2f_18, c2f_21 = inputs
        high_level_features = self.neck_0_concat(
            [
                self.neck_0_0(c2f_15),
                self.neck_0_1(c2f_18),
                self.neck_0_2(c2f_21),
            ]
        )

        mid_level_features = self.neck_1_concat(
            [
                self.neck_1_0(c2f_15),
                self.neck_1_1(c2f_18),
                self.neck_1_2(c2f_21),
            ]
        )
        low_level_features = self.neck_2_concat(
            [
                self.neck_2_0(c2f_15),
                self.neck_2_1(c2f_18),
                self.neck_2_2(c2f_21),
            ]
        )

        return high_level_features, mid_level_features, low_level_features


class MLFD(MLFF):
    """
    Rename MLFF to MLFD
    """

    pass
