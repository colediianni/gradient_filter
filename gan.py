import logging
from pathlib import Path

import matplotlib.pyplot as plt
import os
import torch
# import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

from data import load_data
from layers import EuclideanColorInvariantConv2d, LearnedColorInvariantConv2d
from test_cases import test

# cudnn.benchmark = True

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
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
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
            EuclideanColorInvariantConv2d(nc, ndf, 4, 2, 1, bias=False),
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



# This is a normal generator with a ci discriminator
def train_normal_ci_gan(base_path: Path,
    model_type,
    dataset_name,
    colorspace,
    device,
    D_criterion = nn.BCELoss(),
    G_criterion = nn.BCELoss(),
    epochs=25,
    lr=0.001):

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
    dataloader, nc = load_data(
        dataset=dataset_name,
        colorspace=colorspace,
        batch_size=128,
        train_prop=0.8,
        training_gan=True
    )

    #checking the availability of cuda devices
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # number of gpu's available
    ngpu = 1
    # input noise dimension
    nz = 100
    # number of generator filters
    ngf = 64
    #number of discriminator filters
    ndf = 64

    netG = Generator(ngpu, nz, ngf, nc).to(device)
    netG.apply(weights_init)

    netD = Discriminator(ngpu, ndf, nc).to(device)
    netD.apply(weights_init)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    fixed_noise = torch.randn(128, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    g_loss = []
    d_loss = []

    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            print("here1")
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

            print("here2")
            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = D_criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            print("here3")
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = G_criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (epoch, epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            logging.info('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (epoch, epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            #save the output
            if i % 100 == 0:
                normal_image_path = (
                    base_path
                    / "images"
                    / (f"{dataset_name}.png")
                )
                print('saving the output')
                logging.info('saving the output')
                vutils.save_image(real_cpu,normal_image_path,normalize=False)
                fake = netG(fixed_noise)
                generated_image_path = (
                    base_path
                    / "images"
                    / (f"gan_{model_type}_{dataset_name}_{colorspace}_{epoch}.png")
                )
                vutils.save_image(fake.detach(),generated_image_path,normalize=False)


        # save latest
        print("here4")
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
