import logging
from pathlib import Path

import matplotlib.pyplot as plt
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torchvision

from data import load_data
from layers import EuclideanColorInvariantConv2d, LearnedColorInvariantConv2d
from test_cases import test


# This is a normal generator with a ci discriminator
def normal_ci_gan():
    pass


# What if we add a term to the normal GAN's loss function which encourages it to have a variety of colors!?!?
# L = normal_gan_loss + (lambda * pixel_variation_of_generated_images) <- sum of pixel distance to all other pixels? covariance matrix? KL divergence between rgb cube and actual pixels?
def gan():
    pass


# This is a generator which generates inputs directly into a ci discriminator
# output painting is then needed to make a plottable image
# proposed output painting: randomly choose border colors, then fill others based on distance to top, diag, and left pixels
# will have problems since distances aren't guarenteed to be physically possible
def total_ci_gan():
    pass
