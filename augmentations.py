from random import random

import numpy as np
import torch
import torchvision.transforms.functional as transforms_functional
from numpy.random import normal as gaussian_noise
from torch import Tensor
from torchvision import transforms


class GaussianNoise(torch.nn.Module):
    def __init__(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.loc = loc
        self.scale = scale

    def noise(self, size) -> np.ndarray:
        return gaussian_noise(self.loc, self.scale, size=size)

    def forward(self, image: Tensor) -> Tensor:
        return image + self.noise(image.size())


class SaltAndPepper(torch.nn.Module):
    def __init__(
        self,
        p: float = 0.01,
    ) -> None:
        super().__init__()
        self.p = p

    def forward(self, image: Tensor) -> Tensor:
        dtype_info: torch.finfo | torch.iinfo = (
            torch.finfo(image.dtype)
            if image.dtype.is_floating_point
            else torch.iinfo(image.dtype)
        )
        salt_value = dtype_info.max
        pepper_value = dtype_info.min

        w, h, _c = image.size()
        for i in range(w):
            for j in range(h):
                r = random()
                if r < self.p / 2:
                    image[i, j, :] = salt_value
                elif r < self.p:
                    image[i, j, :] = pepper_value
        return image


class PerPixelChannelPermutation(torch.nn.Module):
    def forward(self, image: Tensor) -> Tensor:
        w, h, c = image.size()
        for i in range(w):
            for j in range(h):
                image[i, j, :] = image[i, j, np.random.permutation(c)]
        return image


class ChannelPermutation(torch.nn.Module):
    def forward(self, image: Tensor) -> Tensor:
        _w, _h, c = image.size()
        image[:, :, :] = image[:, :, np.random.permutation(c)]
        return image


class Invert(torch.nn.Module):
    def forward(self, image: Tensor) -> Tensor:
        return transforms_functional.invert(image)


class HueShift(torch.nn.Module):
    def forward(self, image: Tensor) -> Tensor:
        # TODO determine hue factor
        return transforms_functional.adjust_hue(image, self.hue_factor)


augmentations = {
    "gaussian_noise": [
        transforms.ToTensor(),
        GaussianNoise(),
    ],
    "gaussian_blur": [
        transforms.ToTensor(),
        transforms.GaussianBlur(kernel_size=(3, 3)),
    ],
    "color_jitter": [
        transforms.ToTensor(),
        transforms.ColorJitter(),
    ],
    "salt_and_pepper": [
        transforms.ToTensor(),
        SaltAndPepper(),
    ],
    "per_pixel_channel_permutation": [
        transforms.ToTensor(),
        PerPixelChannelPermutation(),
    ],
    "channel_permutation": [
        transforms.ToTensor(),
        ChannelPermutation(),
    ],
    "invert": [
        transforms.ToTensor(),
        Invert(),
    ],
    "hue_shift": [
        transforms.ToTensor(),
        HueShift(),
    ],
    "grayscale": [
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=3),
    ],
}
