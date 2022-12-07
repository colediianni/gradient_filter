import logging
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import numpy as np

from data import load_data_gan
from image_gradient_recolorizer import GanDecolorizer, colorize_gradient_image

cudnn.benchmark = True

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc, receptive_field, device):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        # self.mult = nc // 3
        self.mult = 1
        self.device = device
        self.decolorizer = GanDecolorizer(receptive_field, distance_metric="euclidean")

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(
                self.nz, self.ngf * 8 * self.mult, 4, 1, 0, bias=False
            ),
            nn.BatchNorm2d(self.ngf * 8 * self.mult),
            nn.ReLU(True),
            # state size. (self.ngf*8) x 4 x 4
            nn.ConvTranspose2d(
                self.ngf * 8 * self.mult,
                self.ngf * 4 * self.mult,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(self.ngf * 4 * self.mult),
            nn.ReLU(True),
            # state size. (self.ngf*4) x 8 x 8
            nn.ConvTranspose2d(
                self.ngf * 4 * self.mult,
                self.ngf * 2 * self.mult,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(self.ngf * 2 * self.mult),
            nn.ReLU(True),
            # state size. (self.ngf*2) x 16 x 16
            nn.ConvTranspose2d(
                self.ngf * 2 * self.mult,
                self.ngf * self.mult,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(self.ngf * self.mult),
            nn.ReLU(True),
            # state size. (self.ngf) x 32 x 32
            nn.ConvTranspose2d(
                self.ngf * self.mult, self.nc, 4, 2, 1, bias=False
            ),
            nn.ReLU(True),
            # nn.Conv2d(self.nc * self.mult, self.nc, int(np.sqrt(self.nc)), 1, padding=int((np.sqrt(self.nc) - 1)/2), bias=False)
            # state size. (self.nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(
                self.main, input, range(self.ngpu)
            )
        else:
            output = self.main(input)
        noise = torch.randn(output.shape, device=self.device) / 100000
        output = output + noise
        return self.decolorizer(output).to(self.device)


class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, nc):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        # self.mult = nc // 3
        self.mult = 1

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf * self.mult, 4, 2, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(
                ndf * self.mult, ndf * 2 * self.mult, 4, 2, 2, bias=False
            ),
            nn.BatchNorm2d(ndf * 2 * self.mult),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(
                ndf * 2 * self.mult, ndf * 4 * self.mult, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(ndf * 4 * self.mult),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(
                ndf * 4 * self.mult, ndf * 8 * self.mult, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(ndf * 8 * self.mult),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8 * self.mult, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(
                self.main, input, range(self.ngpu)
            )
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)


# class Color_Discriminator(nn.Module):
#     def __init__(self, num_pixels, nc=3):
#         super(Color_Discriminator, self).__init__()
#         self.main = nn.Sequential(
#             # input is (nc) x 64 x 64
#             nn.Linear(num_pixels, 1000),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(1000, 1000),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(1000, 2),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, input):
#         print(input.shape)
#         input = torch.sort(input)
#         print("input", input)
#         if input.is_cuda and self.ngpu > 1:
#             output = nn.parallel.data_parallel(
#                 self.main, input, range(self.ngpu)
#             )
#         else:
#             output = self.main(input)
#
#         print("output", output.shape)
#         return output


def train_gan(
    base_path: Path,
    model_type,
    dataset_name,
    colorspace,
    device,
    D_criterion=nn.BCELoss(),
    G_criterion=nn.BCELoss(),
    receptive_field=2,
    epochs=25,
    batch_size=128,
    g_lr=0.0003,
    d_lr=0.0001,
    recolorizer_lr=0.01,
):

    if not isinstance(base_path, Path):
        base_path = Path(base_path)

    output_file = (
        base_path
        / "logs"
        / (
            "gan_"
            + model_type
            + "_"
            + dataset_name
            + "_"
            + colorspace
            + ".txt"
        )
    )
    logger = logging.root
    file_handler = logging.FileHandler(output_file, mode="w")
    stream_handler = logging.StreamHandler()

    logger.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # load dataset
    dataloader, nc = load_data_gan(
        dataset=dataset_name,
        colorspace=colorspace,
        batch_size=batch_size,
        train_prop=1,
    )

    # checking the availability of cuda devices
    device = "cuda" if torch.cuda.is_available() else "cpu"

    nc = (2 * receptive_field + 1) ** 2
    # number of gpu's available
    ngpu = 1
    # input noise dimension
    nz = 256 # 100
    # number of generator filters
    ngf = 64
    # number of discriminator filters
    ndf = 64

    netG = Generator(ngpu=ngpu, nz=nz, ngf=ngf, nc=3, receptive_field=receptive_field, device=device).to(device)
    netG.apply(weights_init)

    netD = Discriminator(ngpu, ndf, nc).to(device)
    netD.apply(weights_init)

    decolorizer = GanDecolorizer(receptive_field, distance_metric="euclidean")

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=d_lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=g_lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    g_loss = []
    d_loss = []
    h, w = 64, 64  # NOTE: h, w hardcoded
    mask = torch.ones([batch_size, nc, h, w], device=device)
    for y_val in range(h):
        for x_val in range(w):
            for direction in range(nc):
                neighbor_x_shift = (direction % int(np.sqrt(nc))) - receptive_field
                neighbor_y_shift = int(direction / int(np.sqrt(nc)))  - receptive_field

                if (
                    y_val + neighbor_y_shift < 0
                    or y_val + neighbor_y_shift >= h
                ):
                    mask[:, direction, y_val, x_val] = 0
                if (
                    x_val + neighbor_x_shift < 0
                    or x_val + neighbor_x_shift >= w
                ):
                    mask[:, direction, y_val, x_val] = 0

    for epoch in range(epochs):
        for i, (images, _) in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            images = images.to(device)
            decolorized_images = decolorizer(images).to(device)
            # print(decolorized_images.shape) # torch.Size([128, 25, 64, 64])
            # print("decolorized_images", decolorized_images.max())
            # batch_size = decolorized_images.size(0)
            label = torch.full(
                (batch_size,), real_label, device=device
            ).float()

            output = netD(decolorized_images)
            errD_real = D_criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)

            # Note: Output must be clipped to valid range because discriminator does not know valid range anymore (would accept negative pixel values if given)
            # fake = torch.clip(netG(noise), 0, 1)

            fake = netG(noise)
            # fake = fake * mask
            # print((fake*(1-mask)).sum())
            # print("fake", fake.max())
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = D_criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            consistency_output = netD(fake)
            consistency_loss = G_criterion(consistency_output, label)
            errG = consistency_loss
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # save the output
            if i % 300 == 0:
                print(
                    "[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f"
                    % (
                        epoch,
                        epochs,
                        i,
                        len(dataloader),
                        errD.item(),
                        errG.item(),
                        D_x,
                        D_G_z1,
                        D_G_z2,
                    )
                )
                logging.info(
                    "[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f",
                    epoch,
                    epochs,
                    i,
                    len(dataloader),
                    errD.item(),
                    errG.item(),
                    D_x,
                    D_G_z1,
                    D_G_z2,
                )
                save_image = (
                    netG(fixed_noise)
                    .detach()
                    .requires_grad_(requires_grad=False)
                )
                # print("save_image", save_image.max(), save_image.min())
                generated_images = colorize_gradient_image(
                    save_image,
                    device,
                    weighted=False,
                    bias_color_location=[[255, 255, 255], "all"],
                    receptive_field=receptive_field,
                    lr=recolorizer_lr,
                    image_is_rgb=False,
                )
                # print("generated_images", generated_images.max(), generated_images.min())
                save_sample_image(
                    generated_images,
                    base_path,
                    model_type,
                    dataset_name,
                    colorspace,
                    epoch,
                )

                save_real_image = decolorized_images.detach().requires_grad_(
                    requires_grad=False
                )
                # print("save_real_image", save_real_image.max(), save_real_image.min())
                real_images = colorize_gradient_image(
                    save_real_image,
                    device,
                    weighted=False,
                    bias_color_location=[[255, 255, 255], "all"],
                    receptive_field=receptive_field,
                    lr=recolorizer_lr,
                    image_is_rgb=False,
                )
                # print("real_images", real_images.max(), real_images.min())
                save_sample_image(
                    real_images,
                    base_path,
                    "real",
                    dataset_name,
                    colorspace,
                    "real",
                )

        # save latest
        if epoch % 5 == 0:
            save_gan(
                base_path,
                model_type,
                dataset_name,
                colorspace,
                epoch,
                netG,
                netD,
            )
        save_gan(
            base_path,
            model_type,
            dataset_name,
            colorspace,
            "latest",
            netG,
            netD,
        )


def save_gan(
    base_path, model_type, dataset_name, colorspace, epoch, netG, netD
):
    g_model_save_path = (
        base_path
        / "models"
        / (
            "gan_"
            + model_type
            + "_"
            + dataset_name
            + "_"
            + colorspace
            + f"_g_{epoch}.pth"
        )
    )
    d_model_save_path = (
        base_path
        / "models"
        / (
            "gan_"
            + model_type
            + "_"
            + dataset_name
            + "_"
            + colorspace
            + f"_d_{epoch}.pth"
        )
    )
    # Check pointing for every epoch
    torch.save(netG.state_dict(), g_model_save_path)
    torch.save(netD.state_dict(), d_model_save_path)


def save_sample_image(
    images, base_path, model_type, dataset_name, colorspace, epoch
):
    image_path = (
        base_path
        / "images"
        / (f"gan_{model_type}_{dataset_name}_{colorspace}_{epoch}.png")
    )
    vutils.save_image(images.detach(), image_path, normalize=False)


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
