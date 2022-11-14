import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import time


class EuclideanColorInvariantConv2d(torch.nn.modules.conv._ConvNd):
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
        super(EuclideanColorInvariantConv2d, self).__init__(
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
        return euclidean_conv2d(
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


def euclidean_conv2d(
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
        testing = torch.square(
            torch.sub(image, image[:, :, pixel : pixel + 1, :])
        )
        # print("testing", testing.shape) # [128, 961, 16, 8]
        comparison_image[:, :, :, pixel] = torch.sqrt(testing.sum(dim=3))

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
        self.cs = torch.nn.CosineSimilarity(dim=3, eps=1e-08)
        self.mapping_model = nn.Sequential(
                    nn.Conv2d(in_channels, 10, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(10, 10, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(10, 10, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(10, 5, 1, 1),
                )

    def conv2d_forward(self, input, weight):
        return learned_conv2d(
            input,
            weight,
            self.cs,
            self.mapping_model,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )

    def forward(self, input):
        return self.conv2d_forward(input, self.weight)


def learned_conv2d(
    input,
    weight,
    cs,
    mapping_model,
    bias=None,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
):
    # print(input.shape)
    start = time.time()
    input = mapping_model(input)
    end = time.time()
    print("mapping_model", end - start)
    # print(input.shape) # torch.Size([128, 10, 32, 32])
    # print(weight.shape) # torch.Size([64, 49, 7, 7])


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

    start = time.time()
    # print("comparison_image", comparison_image.shape) # [128, 961, 16, 16]
    for pixel in range(image.shape[2]):
        # testing = torch.mul(image, image[:, :, pixel : pixel + 1, :]).sum(dim=3)
        # print(testing.sum(dim=3).shape)
        testing = cs(image, image[:, :, pixel : pixel + 1, :])
        # print(testing.shape)
        # print("testing", testing.shape) # [128, 961, 16, 8]
        comparison_image[:, :, :, pixel] = testing

    end = time.time()
    print("for loop", end - start)
    # print("comparison_image", comparison_image.shape) # [128, 961, 16, 16]
    # print(comparison_image.min(), comparison_image.max())

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


class SquaredEuclideanColorInvariantConv2d(torch.nn.modules.conv._ConvNd):
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
        super(SquaredEuclideanColorInvariantConv2d, self).__init__(
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
        return squared_euclidean_conv2d(
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


def squared_euclidean_conv2d(
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
        testing = torch.square(
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


class AbsColorInvariantConv2d(torch.nn.modules.conv._ConvNd):
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
        super(AbsColorInvariantConv2d, self).__init__(
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
        return abs_conv2d(
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


def abs_conv2d(
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
