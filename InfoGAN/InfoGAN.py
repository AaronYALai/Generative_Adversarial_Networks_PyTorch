# -*- coding: utf-8 -*-
# @Author: aaronlai

import torch.nn as nn
import torch.nn.functional as F


class InfoGAN_Discriminator(nn.Module):

    def __init__(self, n_layer=3, n_conti=2, n_discrete=1,
                 num_category=10, use_gpu=False, featmap_dim=256,
                 n_channel=1):
        """
        InfoGAN Discriminator, have additional outputs for latent codes.
        Architecture brought from DCGAN.
        """
        super(InfoGAN_Discriminator, self).__init__()
        self.n_layer = n_layer
        self.n_conti = n_conti
        self.n_discrete = n_discrete
        self.num_category = num_category

        # Discriminator
        self.featmap_dim = featmap_dim

        convs = []
        BNs = []
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
            convs.append(_conv)

            if layer != (self.n_layer - 1):
                _BN = nn.BatchNorm2d(n_conv_out)
                if use_gpu:
                    _BN = _BN.cuda()
                BNs.append(_BN)

        # output layer - prob(real) and auxiliary distributions Q(c_j|x)
        n_hidden = featmap_dim * 4 * 4
        n_output = 1 + n_conti + n_discrete * num_category
        self.fc = nn.Linear(n_hidden, n_output)

        # register all nn modules
        self.convs = nn.ModuleList(convs)
        self.BNs = nn.ModuleList(BNs)

    def forward(self, x):
        """
        Output the probability of being in real dataset
        plus the conditional distributions of latent codes.
        """
        for layer in range(self.n_layer):
            conv_layer = self.convs[self.n_layer - layer - 1]

            if layer == 0:
                x = F.leaky_relu(conv_layer(x), negative_slope=0.2)
            else:
                BN_layer = self.BNs[self.n_layer - layer - 1]
                x = F.leaky_relu(BN_layer(conv_layer(x)), negative_slope=0.2)

        x = x.view(-1, self.featmap_dim * 4 * 4)

        # output layer
        x = self.fc(x)
        x[0] = F.sigmoid(x[0].clone())
        for j in range(self.n_discrete):
            start = 1 + self.n_conti + j * self.num_category
            end = start + self.num_category
            x[start:end] = F.softmax(x[start:end].clone())

        return x


class InfoGAN_Generator(nn.Module):

    def __init__(self, noise_dim=10, n_layer=3, n_conti=2, n_discrete=1,
                 num_category=10, use_gpu=False, featmap_dim=256, n_channel=1):
        """
        InfoGAN Generator, have an additional input branch for latent codes.
        Architecture brought from DCGAN.
        """
        super(InfoGAN_Generator, self).__init__()
        self.n_layer = n_layer
        self.n_conti = n_conti
        self.n_discrete = n_discrete
        self.num_category = num_category

        # calculate input dimension
        n_input = noise_dim + n_conti + n_discrete * num_category

        # Generator
        self.featmap_dim = featmap_dim
        self.fc_in = nn.Linear(n_input, featmap_dim * 4 * 4)

        convs = []
        BNs = []
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
            convs.append(_conv)

            if layer != 0:
                _BN = nn.BatchNorm2d(n_conv_out)
                if use_gpu:
                    _BN = _BN.cuda()
                BNs.append(_BN)

        # register all nn modules
        self.convs = nn.ModuleList(convs)
        self.BNs = nn.ModuleList(BNs)

    def forward(self, x):
        """
        Input the random noise plus latent codes to generate fake images.
        """
        x = self.fc_in(x)
        x = x.view(-1, self.featmap_dim, 4, 4)

        for layer in range(self.n_layer):
            conv_layer = self.convs[self.n_layer - layer - 1]
            if layer == (self.n_layer - 1):
                x = F.tanh(conv_layer(x))
            else:
                BN_layer = self.BNs[self.n_layer - layer - 2]
                x = F.relu(BN_layer(conv_layer(x)))

        return x
