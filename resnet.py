import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torchvision.transforms import functional as F, InterpolationMode
from torchvision.models._api import Weights
from torchvision.models._meta import _IMAGENET_CATEGORIES
from custom_resnet_preprocessing import NoNormalizationImageClassification

from layers import EuclideanColorInvariantConv2d, LearnedColorInvariantConv2d, GrayscaleConv2d, GrayscaleEuclideanColorInvariantConv2d

_COMMON_META = {
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES,
}

no_normalization_weights = Weights(
        url="https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        transforms=partial(NoNormalizationImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 25557032,
            "recipe": "https://github.com/pytorch/vision/issues/3995#issuecomment-1013906621",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 80.858,
                    "acc@5": 95.434,
                }
            },
            "_ops": 4.089,
            "_weight_size": 97.79,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )

#Defining the convolutional neural network
class ResNet(nn.Module):
    def __init__(self, model_type, num_classes=10):
        super().__init__()

        self.resnet = torchvision.models.resnet50(weights=no_normalization_weights)
        if model_type == "normal_resnet":
            self.resnet.conv1 = torch.nn.Conv2d(
                input_channels,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
        elif model_type == "euclidean_diff_ci_resnet":
            self.resnet.conv1 = EuclideanColorInvariantConv2d(
                input_channels,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
        elif model_type == "learned_diff_ci_resnet":
            self.resnet.conv1 = LearnedColorInvariantConv2d(
                input_channels,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
        elif model_type == "grayscale_normal_resnet":
            self.resnet.conv1 = GrayscaleConv2d(
                input_channels,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
        elif model_type == "grayscale_euclidean_diff_ci_resnet":
            self.resnet.conv1 = GrayscaleEuclideanColorInvariantConv2d(
                input_channels,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )

        self.relu = nn.ReLU()
        self.fc = nn.Linear(1000, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.resnet(x)
        out = self.relu(out)
        out = self.fc(out)
        out = self.softmax(out)
        return out
