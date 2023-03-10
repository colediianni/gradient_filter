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
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

from data import load_data, load_data_gan
from layers import EuclideanColorInvariantConv2d, LearnedColorInvariantConv2d, SquaredEuclideanColorInvariantConv2d, AbsColorInvariantConv2d
from test_cases import test
cudnn.benchmark = True

def train_clustering_ci_gan(base_path: Path,
    model_type,
    dataset_name,
    colorspace,
    device,
    pixel_field_of_view = 2,
    D_criterion = nn.BCELoss(),
    G_criterion = nn.BCELoss(),
    epochs=25,
    batch_size=128,
    g_lr=0.0003,
    d_lr=0.0001):

    # assuming first kernel size of discriminator is 4x4
    # each pixel is converted to have distance between itself and neightbors up to pixel_field_of_view pixels away

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    class Generator(nn.Module):
        def __init__(self, ngpu, nz, ngf, nc):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.nz = nz
            self.ngf = ngf
            self.nc = nc
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(self.ngf * 8),
                nn.ReLU(True),
                # state size. (self.ngf*8) x 4 x 4
                nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ngf * 4),
                nn.ReLU(True),
                # state size. (self.ngf*4) x 8 x 8
                nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ngf * 2),
                nn.ReLU(True),
                # state size. (self.ngf*2) x 16 x 16
                nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ngf),
                nn.ReLU(True),
                # state size. (self.ngf) x 32 x 32
                nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (self.nc) x 64 x 64
            )

        def forward(self, input):
            if input.is_cuda and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                output = self.main(input)
                return output

    class Discriminator(nn.Module):
        def __init__(self, ngpu, ndf, nc):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            if input.is_cuda and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                output = self.main(input)

            return output.view(-1, 1).squeeze(1)

    class Color_Discriminator(nn.Module):
        def __init__(self, num_pixels, nc=3):
            super(Color_Discriminator, self).__init__()
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Linear(num_pixels, 1000),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(1000, 1000),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(1000, 2),
                nn.Sigmoid()
            )

        def forward(self, input):
            print(input.shape)
            input = torch.sort(input)
            print("input", input)
            if input.is_cuda and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                output = self.main(input)

            print("output", output.shape)
            return output

    if not isinstance(base_path, Path):
        base_path = Path(base_path)

    output_file = (
        base_path
        / "logs"
        / ("gan_" + model_type + "_" + dataset_name + "_" + colorspace + ".txt")
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
        test_augmentation="remove_color",
    )
    sample = next(iter(dataloader))
    sample_dims = sample[0].shape
    total_pixels_per_batch = sample_dims[0] * sample_dims[2] * sample_dims[3]

    #checking the availability of cuda devices
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    nc=(1+(2*pixel_field_of_view))*(1+(2*pixel_field_of_view))
    # number of gpu's available
    ngpu = 1
    # input noise dimension
    nz = 100
    # number of generator filters
    ngf = 64
    #number of discriminator filters
    ndf = 64

    netG = Generator(ngpu=ngpu, nz=nz, ngf=ngf, nc=nc).to(device)
    netG.apply(weights_init)

    netC = Color_Discriminator(num_pixels=total_pixels_per_batch, nc=nc).to(device)
    # netC.apply(weights_init)

    netD = Discriminator(ngpu, ndf, nc).to(device)
    netD.apply(weights_init)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=d_lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=g_lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(128, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    g_loss = []
    d_loss = []

    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device).float()

            output = netD(real_cpu)
            errD_real = D_criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)

            # Note: Output must be clipped to valid range because discriminator does not know valid range anymore (would accept negative pixel values if given)
            # fake = torch.clip(netG(noise), 0, 1)

            fake = netG(noise)
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
            errG = G_criterion(consistency_output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            #save the output
            if i % 100 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (epoch, epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                logging.info('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (epoch, epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                normal_image_path = (
                    base_path
                    / "images"
                    / (f"{dataset_name}.png")
                )
                print('saving the output')
                logging.info('saving the output')


                # vutils.save_image(real_cpu,normal_image_path,normalize=False)


                # fake = torch.clip(netG(fixed_noise), 0, 1)
                fake = netG(fixed_noise)
                fake = torch.clip((fake - fake.min()), 0, 1)
                generated_image_path = (
                    base_path
                    / "images"
                    / (f"gan_{model_type}_{dataset_name}_{colorspace}_{epoch}.png")
                )

                # vutils.save_image(fake.detach(),generated_image_path,normalize=False)

        # save latest
        if epoch % 5 == 0:
            g_model_save_path = (
                base_path
                / "models"
                / ("gan_" + model_type + "_" + dataset_name + "_" + colorspace + f"_g_{epoch}.pth")
            )
            d_model_save_path = (
                base_path
                / "models"
                / ("gan_" + model_type + "_" + dataset_name + "_" + colorspace + f"_d_{epoch}.pth")
            )
            # Check pointing for every epoch
            torch.save(netG.state_dict(), g_model_save_path)
            torch.save(netD.state_dict(), d_model_save_path)

        g_model_save_path = (
            base_path
            / "models"
            / ("gan_" + model_type + "_" + dataset_name + "_" + colorspace + f"_g_latest.pth")
        )
        d_model_save_path = (
            base_path
            / "models"
            / ("gan_" + model_type + "_" + dataset_name + "_" + colorspace + f"_d_latest.pth")
        )
        # Check pointing for every epoch
        torch.save(netG.state_dict(), g_model_save_path)
        torch.save(netD.state_dict(), d_model_save_path)
