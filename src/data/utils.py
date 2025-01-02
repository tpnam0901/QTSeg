import cv2
import os
import numpy as np

from PIL import Image
from typing import Tuple
from torch.utils.data import DataLoader


def preprocess(
    array: np.ndarray,
) -> np.ndarray:
    """
    Convert HxWxC to CxHxW, scale to [0, 1], standardize, expand dims and convert to float32
    Args:
        array (np.ndarray): HxWxC numpy array to preprocess
        scale_value (float): scale value to divide the array. Usually 255.0 for 8-bit images, 4095.0 for 12-bit images and 65535.0 for 16-bit images
    """
    # Convert HxW to CxHxW
    if len(array.shape) == 2:
        array = np.expand_dims(array, axis=-1)
    # Convert HxWxC to CxHxW
    x = np.moveaxis(array, -1, 0)
    # Min-max scaling
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    # Convert to [-1, 1]
    x = x * 2 - 1
    # Convert to float32
    x = x.astype(np.float32)
    return x


def load_img(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    return np.array(Image.open(path))


def resize(
    array: np.ndarray, size: Tuple[int, int], interpolation=cv2.INTER_AREA
) -> np.ndarray:
    return cv2.resize(array, size, interpolation=interpolation)


def resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return resize(mask, size, interpolation=cv2.INTER_NEAREST)


def get_dataloader(
    dataset,
    repeat: int,
    num_workers: int,
    mode: str,
    batch_size: int = -1,
    collate_fn=None,
):
    def worker_init_fn(worker_id):
        os.sched_setaffinity(0, list(range(os.cpu_count())))

    if mode == "train":
        dataset.repeat(repeat)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if mode == "train" else False,
        pin_memory=True if mode == "train" else False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        drop_last=True if mode == "train" else False,
    )

    return dataloader
