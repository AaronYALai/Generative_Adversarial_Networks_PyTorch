# -*- coding: utf-8 -*-
# @Author: aaronlai

import torch
import torch.nn as nn
import torch.nn.functional as F


class CondiGAN_Discriminator(nn.Module):
    """Conditional GAN Discriminator"""

    def __init__(self, featmap_dim=512, n_channel=1):
        super(DCGAN_Discriminator, self).__init__()
        self.featmap_dim = featmap_dim
        self.conv1 = nn.Conv2d(n_channel, featmap_dim / 8, 5,
                               stride=2, padding=2)
        self.conv_c = nn.Conv2d(n_channel, featmap_dim / 8, 5,
                                stride=2, padding=2)

        self.conv2 = nn.Conv2d(featmap_dim / 4, featmap_dim / 2, 5,
                               stride=2, padding=2)
        self.BN2 = nn.BatchNorm2d(featmap_dim / 2)

        self.conv3 = nn.Conv2d(featmap_dim / 2, featmap_dim, 5,
                               stride=2, padding=2)
        self.BN3 = nn.BatchNorm2d(featmap_dim)

        self.fc = nn.Linear(featmap_dim * 4 * 4, 1)

    def forward(self, x, condi_x):
        """
        Strided convulation layers,
        Batch Normalization after convulation but not at input layer,
        LeakyReLU activation function with slope 0.2.
        """
        x1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x2 = F.leaky_relu(self.conv_c(condi_x), negative_slope=0.2)
        x = torch.cat(tuple(x1.view(-1, np.prod(x1.size()[1:])), x2.view(-1, np.prod(x2.size()[1:]))), 1)
        x = x.view(-1, x1.size()[1]*2, x1.size()[2], x1.size()[3])
        
        x = F.leaky_relu(self.BN2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.BN3(self.conv3(x)), negative_slope=0.2)
        x = x.view(-1, self.featmap_dim * 4 * 4)
        x = F.sigmoid(self.fc(x))
        return x


class CondiGAN_Generator(nn.Module):
    """Conditional GAN Generator"""

    def __init__(self, featmap_dim=1024, n_channel=1, noise_dim=10):
        super(DCGAN_Generator, self).__init__()
        self.featmap_dim = featmap_dim
        self.fc1 = nn.Linear(noise_dim, 4 * 4 * featmap_dim)
        self.conv1 = nn.ConvTranspose2d(featmap_dim, (featmap_dim / 4), 5,
                                        stride=2, padding=2)
        self.conv_c = nn.Conv2d(n_channel, featmap_dim / 4, 5,
                                stride=2, padding=2)

        self.BN1 = nn.BatchNorm2d(featmap_dim / 2)
        self.conv2 = nn.ConvTranspose2d(featmap_dim / 2, featmap_dim / 4, 6,
                                        stride=2, padding=2)

        self.BN2 = nn.BatchNorm2d(featmap_dim / 4)
        self.conv3 = nn.ConvTranspose2d(featmap_dim / 4, n_channel, 6,
                                        stride=2, padding=2)

    def forward(self, x, condi_x):
        """
        Project noise to featureMap * width * height,
        Batch Normalization after convulation but not at output layer,
        ReLU activation function.
        """
        x = self.fc1(x)
        x = x.view(-1, self.featmap_dim, 4, 4)
        x1 = self.conv1(x)
        x2 = self.conv_c(condi_x)
        x = ???
        
        x = F.relu(self.BN1(x))
        x = F.relu(self.BN2(self.conv2(x)))
        x = F.tanh(self.conv3(x))

        return x


class LAPGAN(object):
    """Laplacian Generative Adversarial Networks"""

    def __init__(self, n_level, D_featmap_dim=64, G_featmap_dim=128, use_gpu=False, noise_dim=10):
        self.Dis_models = []
        self.Gen_models = []
        for i in range(n_level):
            self.Dis_models.append(CondiGAN_Discriminator(D_featmap_dim).cuda())
            self.Gen_models.append(CondiGAN_Generator(G_featmap_dim, noise_dim=noise_dim).cuda())

    def reconstruct(self, num):
        pass
