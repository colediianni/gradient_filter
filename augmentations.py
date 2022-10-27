import numpy as np
import torch
import torchvision.transforms.functional as transforms_functional
from torch import Tensor
from torchvision import transforms


class RandomBased(torch.nn.Module):
    def __init__(self, *args, seed: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(seed)


class GaussianNoise(RandomBased):
    def __init__(
        self, *args, loc: float = 0.0, scale: float = 1.0, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loc = loc
        self.scale = scale

    def noise(self, size) -> np.ndarray:
        return self.rng.normal(self.loc, self.scale, size=size)

    def forward(self, image: Tensor) -> Tensor:
        return image + self.noise(image.size())


class SaltAndPepper(RandomBased):
    def __init__(self, *args, p: float = 0.01, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.p = p

    def forward(self, image: Tensor) -> Tensor:
        dtype_info: torch.finfo | torch.iinfo = (
            torch.finfo(image.dtype)
            if image.dtype.is_floating_point
            else torch.iinfo(image.dtype)
        )
        salt_value = 1
        pepper_value = 0

        (
            _c,
            w,
            h,
        ) = image.size()
        for i in range(w):
            for j in range(h):
                r = self.rng.uniform()
                if r < self.p / 2:
                    image[:, i, j] = salt_value
                elif r < self.p:
                    image[:, i, j] = pepper_value
        return image


class PerPixelChannelPermutation(RandomBased):
    def forward(self, image: Tensor) -> Tensor:
        c, w, h = image.size()
        for i in range(w):
            for j in range(h):
                image[:, i, j] = image[self.rng.permutation(c), i, j]
        return image


class ChannelPermutation(RandomBased):
    def forward(self, image: Tensor) -> Tensor:
        c, _w, _h = image.size()
        image[:, :, :] = image[self.rng.permutation(c), :, :]
        return image


class Invert(torch.nn.Module):
    def forward(self, image: Tensor) -> Tensor:
        return transforms_functional.invert(image)


class HueShift(RandomBased):
    def forward(self, image: Tensor) -> Tensor:
        hue_factor = self.rng.uniform(-0.5, 0.5)
        return transforms_functional.adjust_hue(image, hue_factor)


augmentations_dict = {
    "none": [],
    "gaussian_noise": [
        GaussianNoise(scale=0.002),
    ],
    "gaussian_blur": [
        transforms.GaussianBlur(kernel_size=(3, 3)),
    ],
    "color_jitter": [
        transforms.ColorJitter(),
    ],
    "salt_and_pepper": [
        SaltAndPepper(),
    ],
    "per_pixel_channel_permutation": [
        PerPixelChannelPermutation(),
    ],
    "channel_permutation": [
        ChannelPermutation(),
    ],
    "invert": [
        Invert(),
    ],
    "hue_shift": [
        HueShift(),
    ],
    "grayscale": [
        transforms.Grayscale(num_output_channels=3),
    ],
}