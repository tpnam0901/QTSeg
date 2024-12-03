import torch
import torch.nn as nn

from configs.base import Config
from torchvision.ops import sigmoid_focal_loss
from configs.base import Config


class BCELoss(nn.Module):
    def __init__(self, cfg: Config):
        super(BCELoss, self).__init__()
        self.ce = nn.BCELoss()
        self.act = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        inputs = inputs[0]
        return self.ce(self.act(inputs.squeeze(1)), targets.float())


class CrossEntropyLoss(nn.Module):
    def __init__(self, cfg: Config):
        super(CrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        inputs = inputs[0]
        return self.ce(
            inputs.view(inputs.size(0), inputs.size(1), -1),
            targets.view(targets.size(0), -1),
        )


class BinaryDiceLoss(nn.Module):
    def __init__(self, cfg: Config):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = cfg.dice_smooth
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        inputs = inputs[0]
        # B, 2, H, W -> B, H, W
        if len(inputs.size()) == 4 and inputs.size(1) != 1:
            inputs = self.softmax(inputs)
            inputs = inputs[:, 1, :, :]
        elif inputs.min() < 0 or inputs.max() > 1:
            inputs = self.sigmoid(inputs)
        inputs = inputs.squeeze(1).view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1).float()
        intersection = (inputs * targets).sum(dim=1)
        union = inputs.sum(dim=1) + targets.sum(dim=1)
        dice = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
        dice = dice.mean()
        return dice


class CategoricalDiceLoss(nn.Module):
    def __init__(self, cfg: Config):
        super(CategoricalDiceLoss, self).__init__()
        self.smooth = cfg.dice_smooth
        self.softmax = nn.Softmax(dim=1)

    def _one_hot(self, targets: torch.Tensor, num_classes: int):
        target_one_hot = torch.zeros(
            targets.size(0), num_classes, *targets.size()[1:]
        ).to(targets.device)
        for i in range(num_classes):
            target_one_hot[:, i] = targets == i
        return target_one_hot

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        inputs = inputs[0]
        inputs = self.softmax(inputs)
        targets = self._one_hot(targets, inputs.size(1))

        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1).float()

        intersection = (inputs * targets).sum(dim=2)
        union = inputs.sum(dim=2) + targets.sum(dim=2)

        dice = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
        dice = dice.sum(dim=1)
        dice = dice.mean()
        return dice


class FocalLoss(nn.Module):
    def __init__(self, cfg: Config):
        super(FocalLoss, self).__init__()
        self.alpha: float = cfg.focal_alpha
        self.gamma: float = cfg.focal_gamma
        self.reduction: str = "mean"

    def forward(self, inputs, targets):
        inputs = inputs[0]
        inputs = inputs.squeeze(1).view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1).float()
        return sigmoid_focal_loss(
            inputs,
            targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )


class Losses(nn.Module):
    def __init__(self, cfg: Config):
        super(Losses, self).__init__()
        losses = {
            "BCELoss": BCELoss,
            "CrossEntropyLoss": CrossEntropyLoss,
            "BinaryDiceLoss": BinaryDiceLoss,
            "CategoricalDiceLoss": CategoricalDiceLoss,
            "FocalLoss": FocalLoss,
        }
        self.criterions = {loss: losses[loss](cfg) for loss in cfg.loss_type}
        self.criterion_weights = cfg.loss_weight

    def forward(self, inputs, targets):
        loss = 0
        for i, criterion in enumerate(self.criterions.values()):
            loss += self.criterion_weights[i] * criterion(inputs, targets)
        return loss

    def get_params(self):
        return {k: v.parameters() for k, v in self.criterions.items()}

    def to(self, device):
        for criterion in self.criterions.values():
            criterion.to(device)
