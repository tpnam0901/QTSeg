import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from configs.base import Config


class Binary(nn.Module):
    def __init__(self, cfg: Config):
        super(Binary, self).__init__()
        self.MAE = list()
        self.Recall = list()
        self.Precision = list()
        self.Accuracy = list()
        self.Dice = list()
        self.IoU_polyp = list()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)
        self.device = torch.device("cpu")

    def evaluate(self, pred, gt, use_act=True):
        if use_act:
            if pred.size(0) != 1:
                pred = self.softmax(pred)
                pred_binary = pred.argmax(0).float().to(self.device)
                pred_binary_inverse = 1 - pred_binary
            else:
                pred = pred.squeeze(0)
                if pred.min() < 0 or pred.max() > 1:
                    pred = self.sigmoid(pred)
                pred_binary = (pred > 0.5).float().to(self.device)
                pred_binary_inverse = 1 - pred_binary
        else:
            pred_binary = (pred > 0.5).float().to(self.device)
            pred_binary_inverse = 1 - pred_binary

        gt_binary = gt.float().to(self.device)
        gt_binary_inverse = (gt_binary == 0).float().to(self.device)

        MAE = torch.abs(pred_binary - gt_binary).mean().to(self.device)
        TP = pred_binary.mul(gt_binary).sum().to(self.device)
        FP = pred_binary.mul(gt_binary_inverse).sum().to(self.device)
        TN = pred_binary_inverse.mul(gt_binary_inverse).sum().to(self.device)
        FN = pred_binary_inverse.mul(gt_binary).sum().to(self.device)

        if TP.item() == 0:
            TP = torch.Tensor([1]).to(self.device)
        # recall
        Recall = TP / (TP + FN)
        # Precision or positive predictive value
        Precision = TP / (TP + FP)
        # Specificity = TN / (TN + FP)
        # F1 score = Dice
        Dice = 2 * Precision * Recall / (Precision + Recall)
        # Overall accuracy
        Accuracy = (TP + TN) / (TP + FP + FN + TN)
        # IoU for poly
        IoU_polyp = TP / (TP + FP + FN)

        return (
            MAE.data.cpu().numpy().squeeze(),
            Recall.data.cpu().numpy().squeeze(),
            Precision.data.cpu().numpy().squeeze(),
            Accuracy.data.cpu().numpy().squeeze(),
            Dice.data.cpu().numpy().squeeze(),
            IoU_polyp.data.cpu().numpy().squeeze(),
        )

    def forward(self, pred, gt, use_act=True):
        pred = pred[0]
        running_mae = []
        running_recall = []
        running_precision = []
        running_accuracy = []
        running_dice = []
        running_iou_polyp = []
        for i in range(gt.size(0)):
            mae, recall, precision, accuracy, dice, ioU_polyp = self.evaluate(
                pred[i], gt[i], use_act
            )
            self.MAE.append(mae)
            running_mae.append(mae)
            self.Recall.append(recall)
            running_recall.append(recall)
            self.Precision.append(precision)
            running_precision.append(precision)
            self.Accuracy.append(accuracy)
            running_accuracy.append(accuracy)
            self.Dice.append(dice)
            running_dice.append(dice)
            self.IoU_polyp.append(ioU_polyp)
            running_iou_polyp.append(ioU_polyp)
        return {
            "mae": np.mean(running_mae) * 100,
            # "recall": np.mean(running_recall) * 100,
            # "precision": np.mean(running_precision) * 100,
            "acc": np.mean(running_accuracy) * 100,
            "dice": np.mean(running_dice) * 100,
            "iou": np.mean(running_iou_polyp) * 100,
        }

    def compute(self):
        return {
            "mae": np.mean(self.MAE) * 100,
            # "recall": np.mean(self.Recall) * 100,
            # "precision": np.mean(self.Precision) * 100,
            "acc": np.mean(self.Accuracy) * 100,
            "dice": np.mean(self.Dice) * 100,
            "iou": np.mean(self.IoU_polyp) * 100,
        }

    def reset(self):
        self.MAE = list()
        self.Recall = list()
        self.Precision = list()
        self.Accuracy = list()
        self.Dice = list()
        self.IoU_polyp = list()

    def to(self, device):
        super().to(device)
        self.device = device


class MultiBinary(nn.Module):
    def __init__(self, cfg: Config):
        super(MultiBinary, self).__init__()
        self.num_masks = cfg.num_masks
        for i in range(1, cfg.num_masks):
            setattr(self, f"metric_{i}", Binary(cfg))

    def forward(self, pred, gt):
        ret = {}
        pred = pred[0]
        pred = torch.argmax(pred, dim=1)
        pred = F.one_hot(pred.long(), num_classes=self.num_masks)
        new_labels = F.one_hot(gt.long(), num_classes=self.num_masks)
        mae = []
        accuracy = []
        dice = []
        iou_polyp = []
        for i in range(1, self.num_masks):
            metrics = getattr(self, f"metric_{i}")(
                [pred[..., i]], new_labels[..., i].float(), use_act=False
            )
            mae.append(metrics["mae"])
            ret[f"mae_{i}"] = metrics["mae"]
            accuracy.append(metrics["acc"])
            ret[f"acc_{i}"] = metrics["acc"]
            dice.append(metrics["dice"])
            ret[f"dice_{i}"] = metrics["dice"]
            iou_polyp.append(metrics["iou"])
            ret[f"iou_{i}"] = metrics["iou"]
        ret.update(
            {
                "mae": np.mean(mae),
                "acc": np.mean(accuracy),
                "dice": np.mean(dice),
                "iou": np.mean(iou_polyp),
            }
        )
        return ret

    def compute(self):
        ret = {}
        mae = []
        accuracy = []
        dice = []
        iou_polyp = []
        for i in range(1, self.num_masks):
            metrics = getattr(self, f"metric_{i}").compute()
            mae.append(metrics["mae"])
            ret[f"mae_{i}"] = metrics["mae"]
            accuracy.append(metrics["acc"])
            ret[f"acc_{i}"] = metrics["acc"]
            dice.append(metrics["dice"])
            ret[f"dice_{i}"] = metrics["dice"]
            iou_polyp.append(metrics["iou"])
            ret[f"iou_{i}"] = metrics["iou"]

        ret.update(
            {
                "mae": np.mean(mae),
                "acc": np.mean(accuracy),
                "dice": np.mean(dice),
                "iou": np.mean(iou_polyp),
            }
        )
        return ret

    def reset(self):
        for i in range(1, self.num_masks):
            getattr(self, f"metric_{i}").reset()

    def to(self, device):
        super().to(device)
        for i in range(1, self.num_masks):
            getattr(self, f"metric_{i}").to(device)


class Metrics(nn.Module):
    def __init__(self, cfg: Config):
        super(Metrics, self).__init__()
        metrics = {
            "MultiBinary": MultiBinary,
            "Binary": Binary,
        }
        self.metric = metrics[cfg.metric](cfg)

    def forward(self, outputs, targets):

        return self.metric(outputs, targets)

    def compute(self):
        return self.metric.compute()

    def reset(self):
        self.metric.reset()

    def to(self, device):
        super().to(device)
        self.metric.to(device)
