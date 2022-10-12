import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair


class RGBColorInvariantConv2d(torch.nn.modules.conv._ConvNd):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(RGBColorInvariantConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
        )
        # print("HERE", self.weight.shape) # [10, 3, 28, 28])
        self.weight = torch.nn.Parameter(
            torch.rand(
                size=[
                    self.weight.shape[0],
                    self.weight.shape[2] * self.weight.shape[3],
                    self.weight.shape[2],
                    self.weight.shape[3],
                ],
                device="cuda",
            ),
            requires_grad=True,
        )  # add to initialize weights at mean pixel value

    def conv2d_forward(self, input, weight):
        return rgb_conv2d(
            input,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, input):
        return self.conv2d_forward(input, self.weight)


def rgb_conv2d(
    input,
    weight,
    bias=None,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
):
    batch_size, in_channels, in_h, in_w = input.shape
    out_channels, kern_in_channels, kh, kw = weight.shape
    out_h = int((in_h - kh + 2 * padding[0]) / stride[0] + 1)
    out_w = int((in_w - kw + 2 * padding[1]) / stride[1] + 1)
    unfold = torch.nn.Unfold(
        kernel_size=(kh, kw), dilation=dilation, padding=padding, stride=stride
    )
    inp_unf = unfold(input)
    w_ = weight.view(weight.size(0), -1).t()

    image = inp_unf.transpose(1, 2)
    kern = w_.transpose(1, 0)
    # print("image", image.shape) # [4, 784, 75]
    # print("kern", kern.shape) # [6, 625]

    image = image.reshape((image.shape[0], -1, in_channels, kh * kw))
    # print("image", image.shape) # [128, 961, 3, 16]
    kern = kern.reshape((out_channels, kern_in_channels, kh * kw))
    # print("kern", kern.shape) # [64, 16, 16]

    comparison_image = torch.zeros(
        size=(image.shape[0], image.shape[1], image.shape[3], image.shape[3]),
        device="cuda",
    )
    # print("comparison_image", comparison_image.shape) # [128, 961, 16, 16]

    image = image.permute([0, 1, 3, 2])
    # print("rearranged", rearranged.shape) # [128, 961, 16, 3]
    # colors = torch.matmul(rearranged, normalized_colors)
    # print("colors", colors.shape) # [128, 961, 16, 4]

    # print("comparison_image", comparison_image.shape) # [128, 961, 16, 16]
    for pixel in range(image.shape[2]):
        testing = torch.abs(
            torch.sub(image, image[:, :, pixel : pixel + 1, :])
        )
        # print("testing", testing.shape) # [128, 961, 16, 8]
        comparison_image[:, :, :, pixel] = testing.sum(dim=3)

    # print("comparison_image", comparison_image.shape) # [128, 961, 16, 16]

    first = comparison_image.flatten(2)
    # print("first", first.shape) # [128, 961, 256]
    second = kern.flatten(1)
    # print("second", second.shape) # [64, 256]
    mul = first.matmul(second.t())
    # print("mul", mul.shape) # [128, 961, 64]

    if bias is None:
        out_unf = mul.transpose(1, 2)
    else:
        out_unf = (mul + bias).transpose(1, 2)
    out = out_unf.view(batch_size, out_channels, out_h, out_w)
    return out.float()


class LearnedColorInvariantConv2d(torch.nn.modules.conv._ConvNd):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(LearnedColorInvariantConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
        )
        self.weight = torch.nn.Parameter(
            torch.rand(
                size=[
                    self.weight.shape[0],
                    self.weight.shape[2] * self.weight.shape[3],
                    self.weight.shape[2],
                    self.weight.shape[3],
                ],
                device="cuda",
            ),
            requires_grad=True,
        )  # add to initialize weights at mean pixel value
        self.color_mapping_model = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 25),
            nn.ReLU(inplace=True),
            nn.Linear(25, 25),
            nn.ReLU(inplace=True),
            nn.Linear(25, 10),
        )  # TODO: Initialize color mapping model

    def conv2d_forward(self, input, weight, color_mapping_model):
        return learned_ci_conv2d(
            input,
            color_mapping_model,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, input):
        return self.conv2d_forward(
            input, self.weight, self.color_mapping_model
        )


# Clean Convolution
def learned_ci_conv2d(
    input,
    color_mapping_model,
    weight,
    bias=None,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
):
    batch_size, in_channels, in_h, in_w = input.shape
    out_channels, kern_in_channels, kh, kw = weight.shape
    out_h = int((in_h - kh + 2 * padding[0]) / stride[0] + 1)
    out_w = int((in_w - kw + 2 * padding[1]) / stride[1] + 1)
    unfold = torch.nn.Unfold(
        kernel_size=(kh, kw), dilation=dilation, padding=padding, stride=stride
    )
    inp_unf = unfold(input)
    w_ = weight.view(weight.size(0), -1).t()

    image = inp_unf.transpose(1, 2)
    kern = w_.transpose(1, 0)
    # print("image", image.shape) # [4, 784, 75]
    # print("kern", kern.shape) # [6, 625]

    image = image.reshape((image.shape[0], -1, in_channels, kh * kw))
    # print("image", image.shape) # [128, 961, 3, 16]
    kern = kern.reshape((out_channels, kern_in_channels, kh * kw))
    # print("kern", kern.shape) # [64, 16, 16]

    comparison_image = torch.zeros(
        size=(image.shape[0], image.shape[1], image.shape[3], image.shape[3]),
        device="cuda",
    )
    # print("comparison_image", comparison_image.shape) # [128, 961, 16, 16]

    image = image.permute([0, 1, 3, 2])
    # print("rearranged", rearranged.shape) # [128, 961, 16, 3]
    # colors = torch.matmul(rearranged, normalized_colors)
    # print("colors", colors.shape) # [128, 961, 16, 4]

    # I don't think we need location information since the locations will be specific to the matrix index

    # print("comparison_image", comparison_image.shape) # [128, 961, 16, 16]
    for pixel in range(image.shape[2]):
        # print(color_mapping_model(image).shape) # torch.Size([16, 256, 49, 10])
        # print(color_mapping_model(image[:, :, pixel:pixel + 1, :]).shape) # torch.Size([16, 256, 1, 10])

        testing = torch.mul(
            color_mapping_model(image),
            color_mapping_model(image[:, :, pixel : pixel + 1, :]),
        )
        # print(testing.shape)
        # print(testing.sum(dim=3).shape)
        # print("testing", testing.shape) # [128, 961, 16, 8]
        comparison_image[:, :, :, pixel] = testing.sum(dim=3)

    # print("comparison_image", comparison_image.shape) # [128, 961, 16, 16]

    first = comparison_image.flatten(2)
    # print("first", first.shape) # [128, 961, 256]
    second = kern.flatten(1)
    # print("second", second.shape) # [64, 256]
    mul = first.matmul(second.t())
    # print("mul", mul.shape) # [128, 961, 64]

    if bias is None:
        out_unf = mul.transpose(1, 2)
    else:
        out_unf = (mul + bias).transpose(1, 2)
    out = out_unf.view(batch_size, out_channels, out_h, out_w)
    return out.float()
