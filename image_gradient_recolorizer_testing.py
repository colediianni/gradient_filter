import matplotlib.pyplot as plt
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

from data import load_data_gan
from image_gradient_recolorizer import remove_color, remove_infs, colorize_gradient_image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
colorspace="rgb"

# load dataset
dataloader, nc = load_data_gan(
    dataset="cifar",
    colorspace="rgb",
    batch_size=64,
    train_prop=1,
    test_augmentation="none"
)

image = next(iter(dataloader))[0]
print(image.shape)
gradient_image = transforms.Compose([remove_color(4, "absolute")])(image)
print(gradient_image.shape)

# image = (image * 255).type(torch.int)
gradient_image = (gradient_image * 255).type(torch.int)

plt.imshow(image[0].permute([1, 2, 0]))
plt.show()

generated_image = colorize_gradient_image(image, device, bias_color_location=[], weighted=False, receptive_field=4, lr=2)
bias = [[255, 0, 0], "all"]
generated_image = colorize_gradient_image(image, device, bias_color_location=bias, weighted=False, receptive_field=4, lr=2)
bias = [[[100, 0, 0], [200, 200, 200]], [[10, 10], [20, 20]]]
generated_image = colorize_gradient_image(image, device, bias_color_location=bias, weighted=False, receptive_field=4, lr=2)
