# -*- coding: utf-8 -*-
# @Author: aaronlai

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from torch.autograd import Variable


class CondiGAN_Discriminator(nn.Module):

    def __init__(self, n_layer=3, condition=True, use_gpu=False,
                 featmap_dim=256, n_channel=1, condi_featmap_dim=256):
        """
        Conditional Discriminator.
        Architecture brought from DCGAN.
        """
        super(CondiGAN_Discriminator, self).__init__()
        self.n_layer = n_layer
        self.condition = condition

        # original Discriminator
        self.featmap_dim = featmap_dim
        self.convs = []
        self.BNs = []

        for layer in range(self.n_layer):
            if layer == (self.n_layer - 1):
                n_conv_in = n_channel
            else:
                n_conv_in = int(featmap_dim / (2**(layer + 1)))
            n_conv_out = int(featmap_dim / (2**layer))

            _conv = nn.Conv2d(n_conv_in, n_conv_out, kernel_size=5,
                              stride=2, padding=2)
            if use_gpu:
                _conv = _conv.cuda()
            self.convs.append(_conv)

            if layer != (self.n_layer - 1):
                _BN = nn.BatchNorm2d(n_conv_out)
                if use_gpu:
                    _BN = _BN.cuda()
                self.BNs.append(_BN)

        # extra image information to be conditioned on
        if self.condition:
            self.condi_featmap_dim = condi_featmap_dim
            self.convs_condi = []
            self.BNs_condi = []

            for layer in range(self.n_layer):
                if layer == (self.n_layer - 1):
                    n_conv_in = n_channel
                else:
                    n_conv_in = int(condi_featmap_dim / (2**(layer + 1)))
                n_conv_out = int(condi_featmap_dim / (2**layer))

                _conv = nn.Conv2d(n_conv_in, n_conv_out, kernel_size=5,
                                  stride=2, padding=2)
                if use_gpu:
                    _conv = _conv.cuda()
                self.convs_condi.append(_conv)

                if layer != (self.n_layer - 1):
                    _BN = nn.BatchNorm2d(n_conv_out)
                    if use_gpu:
                        _BN = _BN.cuda()
                    self.BNs_condi.append(_BN)

        # output layer
        if self.condition:
            n_hidden = int((featmap_dim + condi_featmap_dim) * 4 * 4)
        else:
            n_hidden = int(featmap_dim * 4 * 4)

        self.fc = nn.Linear(n_hidden, 1)

    def forward(self, x, condi_x=None):
        """
        Concatenate CNN-processed extra information vector at the last layer
        """
        for layer in range(self.n_layer):
            conv_layer = self.convs[self.n_layer - layer - 1]
            if layer == 0:
                x = F.leaky_relu(conv_layer(x), negative_slope=0.2)
            else:
                BN_layer = self.BNs[self.n_layer - layer - 1]
                x = F.leaky_relu(BN_layer(conv_layer(x)), negative_slope=0.2)
        x = x.view(-1, self.featmap_dim * 4 * 4)

        # calculate and concatenate extra information
        if self.condition:
            for layer in range(self.n_layer):
                _conv = self.convs_condi[self.n_layer - layer - 1]
                if layer == 0:
                    condi_x = F.leaky_relu(_conv(condi_x), negative_slope=0.2)
                else:
                    BN_layer = self.BNs_condi[self.n_layer - layer - 1]
                    condi_x = F.leaky_relu(BN_layer(_conv(condi_x)),
                                           negative_slope=0.2)

            condi_x = condi_x.view(-1, self.condi_featmap_dim * 4 * 4)
            x = torch.cat((x, condi_x), 1)

        # output layer
        x = F.sigmoid(self.fc(x))

        return x


class CondiGAN_Generator(nn.Module):

    def __init__(self, noise_dim=10, n_layer=3, condition=True, use_gpu=False,
                 featmap_dim=256, n_channel=1, condi_featmap_dim=256):
        """
        Conditional Generator.
        Architecture brought from DCGAN.
        """
        super(CondiGAN_Generator, self).__init__()
        self.n_layer = n_layer
        self.condition = condition

        # extra image information to be conditioned on
        if self.condition:
            self.condi_featmap_dim = condi_featmap_dim
            self.convs_condi = []
            self.BNs_condi = []

            for layer in range(self.n_layer):
                if layer == (self.n_layer - 1):
                    n_conv_in = n_channel
                else:
                    n_conv_in = int(condi_featmap_dim / (2**(layer + 1)))
                n_conv_out = int(condi_featmap_dim / (2**layer))

                _conv = nn.Conv2d(n_conv_in, n_conv_out, kernel_size=5,
                                  stride=2, padding=2)
                if use_gpu:
                    _conv = _conv.cuda()
                self.convs_condi.append(_conv)

                if layer != (self.n_layer - 1):
                    _BN = nn.BatchNorm2d(n_conv_out)
                    if use_gpu:
                        _BN = _BN.cuda()
                    self.BNs_condi.append(_BN)

        # calculate input dimension
        if self.condition:
            n_input = int(noise_dim + condi_featmap_dim * 4 * 4)
        else:
            n_input = noise_dim

        # Generator
        self.featmap_dim = featmap_dim
        self.fc1 = nn.Linear(n_input, int(featmap_dim * 4 * 4))
        self.convs = []
        self.BNs = []

        for layer in range(self.n_layer):
            if layer == 0:
                n_conv_out = n_channel
            else:
                n_conv_out = featmap_dim / (2 ** (self.n_layer - layer))
            n_conv_in = featmap_dim / (2 ** (self.n_layer - layer - 1))

            n_width = 5 if layer == (self.n_layer - 1) else 6
            _conv = nn.ConvTranspose2d(n_conv_in, n_conv_out, n_width,
                                       stride=2, padding=2)
            if use_gpu:
                _conv = _conv.cuda()
            self.convs.append(_conv)

            if layer != 0:
                _BN = nn.BatchNorm2d(n_conv_out)
                if use_gpu:
                    _BN = _BN.cuda()
                self.BNs.append(_BN)

    def forward(self, x, condi_x=None):
        """
        Concatenate CNN-processed extra information vector at the first layer
        """
        # calculate and concatenate extra information
        if self.condition:
            for layer in range(self.n_layer):
                _conv = self.convs_condi[self.n_layer - layer - 1]
                if layer == 0:
                    condi_x = F.leaky_relu(_conv(condi_x), negative_slope=0.2)
                else:
                    BN_layer = self.BNs_condi[self.n_layer - layer - 1]
                    condi_x = F.leaky_relu(BN_layer(_conv(condi_x)),
                                           negative_slope=0.2)

            condi_x = condi_x.view(-1, self.condi_featmap_dim * 4 * 4)
            x = torch.cat((x, condi_x), 1)

        x = self.fc1(x)
        x = x.view(-1, self.featmap_dim, 4, 4)

        for layer in range(self.n_layer):
            conv_layer = self.convs[self.n_layer - layer - 1]
            if layer == (self.n_layer - 1):
                x = F.tanh(conv_layer(x))
            else:
                BN_layer = self.BNs[self.n_layer - layer - 2]
                x = F.relu(BN_layer(conv_layer(x)))

        return x


class LAPGAN(object):

    def __init__(self, n_level, noise_dim=10, D_featmap_dim=64,
                 condi_D_featmap_dim=64, G_featmap_dim=128,
                 condi_G_featmap_dim=64, use_gpu=False, n_channel=1):
        """
        Initialize a group of discriminators and generators for LAPGAN
        n_level: number of levels in the Laplacian Pyramid
        noise_dim: dimension of random noise to feed into the last generator
        D_featmap_dim: discriminator, (#feature maps) in the last layer of CNN
        condi_D_featmap_dim: (#feature maps) of extra info CNN's last layer
        G_featmap_dim: generator, (#feature maps) of deconvNN's first layer
        condi_G_featmap_dim: (#feature maps) of extra info CNN's last layer
        use_gpu: to use GPU computation or not
        n_channel: number of channel for input images
        """
        self.n_level = n_level
        self.n_channel = n_channel
        self.use_gpu = use_gpu
        self.noise_dim = noise_dim
        self.Dis_models = []
        self.Gen_models = []

        for level in range(n_level):
            n_layer = n_level - level
            if level == (n_level - 1):
                condition = False
            else:
                condition = True

            Dis_model = CondiGAN_Discriminator(n_layer, condition, use_gpu,
                                               D_featmap_dim, n_channel,
                                               condi_D_featmap_dim)
            Gen_model = CondiGAN_Generator(noise_dim, n_layer, condition,
                                           use_gpu, G_featmap_dim, n_channel,
                                           condi_G_featmap_dim)

            if use_gpu:
                Dis_model = Dis_model.cuda()
                Gen_model = Gen_model.cuda()

            self.Dis_models.append(Dis_model)
            self.Gen_models.append(Gen_model)

    def generate(self, batchsize):
        """Generate images from LAPGAN generators"""
        for level in range(self.n_level):
            Gen_model = self.Gen_models[self.n_level - level - 1]

            # generate noise
            noise = Variable(gen_noise(batchsize, self.noise_dim))
            if self.use_gpu:
                noise = noise.cuda()

            if level == 0:
                # directly generate images
                output_imgs = Gen_model.forward(noise)
                if self.use_gpu:
                    output_imgs = output_imgs.cpu()
                output_imgs = output_imgs.data.numpy()
            else:
                # upsize
                input_imgs = np.array([[cv2.pyrUp(output_imgs[i, j, :])
                                      for j in range(self.n_channel)]
                                      for i in range(batchsize)])
                condi_imgs = Variable(torch.Tensor(input_imgs))
                if self.use_gpu:
                    condi_imgs = condi_imgs.cuda()

                # generate images with extra information
                residual_imgs = Gen_model.forward(noise, condi_imgs)
                if self.use_gpu:
                    residual_imgs = residual_imgs.cpu()
                output_imgs = residual_imgs.data.numpy() + input_imgs

        return output_imgs


def gen_noise(n_instance, n_dim=2):
    """generate 2-dim uniform random noise"""
    return torch.Tensor(np.random.uniform(low=-1.0, high=1.0,
                                          size=(n_instance, n_dim)))
