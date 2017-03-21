# -*- coding: utf-8 -*-
# @Author: aaronlai

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedGAN_Discriminator(nn.Module):

    def __init__(self, featmap_dim=512, n_channel=1, use_gpu=False,
                 n_B=128, n_C=16):
        """
        Minibatch discrimination: learn a tensor to encode side information
        from other examples in the same minibatch.
        """
        super(ImprovedGAN_Discriminator, self).__init__()
        self.use_gpu = use_gpu
        self.n_B = n_B
        self.n_C = n_C
        self.featmap_dim = featmap_dim

        self.conv1 = nn.Conv2d(n_channel, featmap_dim / 4, 5,
                               stride=2, padding=2)

        self.conv2 = nn.Conv2d(featmap_dim / 4, featmap_dim / 2, 5,
                               stride=2, padding=2)
        self.BN2 = nn.BatchNorm2d(featmap_dim / 2)

        self.conv3 = nn.Conv2d(featmap_dim / 2, featmap_dim, 5,
                               stride=2, padding=2)
        self.BN3 = nn.BatchNorm2d(featmap_dim)

        T_ten_init = torch.randn(featmap_dim * 4 * 4, n_B * n_C) * 0.1
        self.T_tensor = nn.Parameter(T_ten_init, requires_grad=True)
        self.fc = nn.Linear(featmap_dim * 4 * 4 + n_B, 1)

    def forward(self, x):
        """
        Strided convulation layers,
        Batch Normalization after convulation but not at input layer,
        LeakyReLU activation function with slope 0.2.
        """
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.BN2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.BN3(self.conv3(x)), negative_slope=0.2)
        x = x.view(-1, self.featmap_dim * 4 * 4)

        T_tensor = self.T_tensor
        if self.use_gpu:
            T_tensor = T_tensor.cuda()

        Ms = x.mm(T_tensor)
        Ms = Ms.view(-1, self.n_B, self.n_C)

        out_tensor = []
        for i in range(Ms.size()[0]):

            out_i = None
            for j in range(Ms.size()[0]):
                o_i = torch.sum(torch.abs(Ms[i, :, :] - Ms[j, :, :]), 1)
                o_i = torch.exp(-o_i)
                if out_i is None:
                    out_i = o_i
                else:
                    out_i = out_i + o_i

            out_tensor.append(out_i)

        out_T = torch.cat(tuple(out_tensor)).view(Ms.size()[0], self.n_B)
        x = torch.cat((x, out_T), 1)

        x = F.sigmoid(self.fc(x))
        return x


class ImprovedGAN_Generator(nn.Module):

    def __init__(self, featmap_dim=1024, n_channel=1, noise_dim=100):
        super(ImprovedGAN_Generator, self).__init__()
        self.featmap_dim = featmap_dim
        self.fc1 = nn.Linear(noise_dim, 4 * 4 * featmap_dim)
        self.conv1 = nn.ConvTranspose2d(featmap_dim, (featmap_dim / 2), 5,
                                        stride=2, padding=2)

        self.BN1 = nn.BatchNorm2d(featmap_dim / 2)
        self.conv2 = nn.ConvTranspose2d(featmap_dim / 2, featmap_dim / 4, 6,
                                        stride=2, padding=2)

        self.BN2 = nn.BatchNorm2d(featmap_dim / 4)
        self.conv3 = nn.ConvTranspose2d(featmap_dim / 4, n_channel, 6,
                                        stride=2, padding=2)

    def forward(self, x):
        """
        Project noise to featureMap * width * height,
        Batch Normalization after convulation but not at output layer,
        ReLU activation function.
        """
        x = self.fc1(x)
        x = x.view(-1, self.featmap_dim, 4, 4)
        x = F.relu(self.BN1(self.conv1(x)))
        x = F.relu(self.BN2(self.conv2(x)))
        x = F.tanh(self.conv3(x))

        return x
