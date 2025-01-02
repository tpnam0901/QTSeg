import cv2
import torch
import numpy as np
from typing import Tuple
from torch.utils.data import Dataset

from .utils import load_img, preprocess, resize_mask, resize_longest_side, pad_to_square
import imgaug as ia
import imgaug.augmenters as iaa


class BaseDataset(Dataset):
    def __init__(
        self,
        X,
        y,
        img_size: int = 512,
        num_classes: int = 2,
        cvtColor=None,
    ):
        super(BaseDataset, self).__init__()
        self.X = X
        self.y = y
        self.img_size = img_size
        self.num_classes = num_classes
        self.img_aug = None
        self.cvtColor = cvtColor

    def repeat(self, n: int):
        # Repeat the dataset n times
        self.X = self.X * n
        self.y = self.y * n

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = load_img(self.X[index])
        y = load_img(self.y[index]).astype(np.int64)

        x_raw_rgb = x = resize_longest_side(
            x, self.img_size, interpolation=cv2.INTER_AREA
        )
        if self.cvtColor is not None:
            x = cv2.cvtColor(x, self.cvtColor)
        y = resize_mask(y, (x.shape[1], x.shape[0]))
        if np.max(y) > 1 and self.num_classes <= 2:
            y = y / np.max(y)
            if len(y.shape) == 3:
                y = y[:, :, 0]

        x = pad_to_square(x)
        x_raw_rgb = pad_to_square(x_raw_rgb)
        y = pad_to_square(y)

        x, y = self.augment_seg(x, y)
        # Standardization
        x = preprocess(x)
        assert len(y.shape) == 2
        return (
            torch.from_numpy(x),
            torch.from_numpy(y.astype(np.int64)),
            torch.from_numpy(x_raw_rgb.astype(np.float32)),
        )

    def __len__(self):
        return len(self.X)

    def augment_seg(self, img, seg):
        return img, seg


class AugDataset(BaseDataset):
    def __init__(
        self,
        X,
        y,
        img_size: 512,
        num_classes: int = 2,
        cvtColor=None,
    ):
        super(AugDataset, self).__init__(X, y, img_size, num_classes, cvtColor)
        self.img_aug = iaa.SomeOf(
            (0, 4),
            [
                iaa.Flipud(0.5, name="Flipud"),
                iaa.Fliplr(0.5, name="Fliplr"),
                iaa.AdditiveGaussianNoise(scale=0.005 * 255),
                iaa.GaussianBlur(sigma=(1.0)),
                iaa.LinearContrast((0.5, 1.5), per_channel=0.5),
                iaa.Affine(scale={"x": (0.5, 2), "y": (0.5, 2)}),
                iaa.Affine(rotate=(-40, 40)),
                iaa.Affine(shear=(-16, 16)),
                iaa.PiecewiseAffine(scale=(0.008, 0.03)),
                iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
            ],
            random_order=True,
        )

    def mask_to_onehot(self, mask):
        """
        Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
        hot encoding vector, C is usually 1 or 3, and K is the number of class.
        """
        semantic_map = []
        mask = np.expand_dims(mask, -1)
        for colour in range(9):
            equality = np.equal(mask, colour)
            class_map = np.all(equality, axis=-1)
            semantic_map.append(class_map)
        semantic_map = np.stack(semantic_map, axis=-1).astype(np.int32)
        return semantic_map

    def augment_seg(self, img, seg):
        seg = self.mask_to_onehot(seg)
        aug_det = self.img_aug.to_deterministic()
        image_aug = aug_det.augment_image(img)

        segmap = ia.SegmentationMapOnImage(
            seg, nb_classes=np.max(seg) + 1, shape=img.shape
        )
        segmap_aug = aug_det.augment_segmentation_maps(segmap)
        segmap_aug = segmap_aug.get_arr_int()
        segmap_aug = np.argmax(segmap_aug, axis=-1).astype(np.float32)
        return image_aug, segmap_aug
