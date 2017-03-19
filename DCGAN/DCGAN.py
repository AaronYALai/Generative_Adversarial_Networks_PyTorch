# -*- coding: utf-8 -*-
# @Author: aaronlai

import torch.nn as nn
import torch.nn.functional as F


class DCGAN_Discriminator(nn.Module):

    def __init__(self):
        super(DCGAN_Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.BN2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, 5, stride=2, padding=2)
        self.BN3 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512 * 4 * 4, 1)

    def forward(self, x):
        """
        Strided convulation layers,
        Batch Normalization after convulation but not at input layer,
        LeakyReLU activation function with slope 0.2.
        """
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.BN2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.BN3(self.conv3(x)), negative_slope=0.2)
        x = x.view(-1, 512 * 4 * 4)
        x = F.sigmoid(self.fc(x))
        return x


class DCGAN_Generator(nn.Module):

    def __init__(self, noise_dim=100):
        super(DCGAN_Generator, self).__init__()
        self.fc1 = nn.Linear(noise_dim, 4 * 4 * 1024)
        self.conv1 = nn.ConvTranspose2d(1024, 512, 5, stride=2, padding=2)
        self.BN1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 256, 6, stride=2, padding=2)
        self.BN2 = nn.BatchNorm2d(256)
        self.conv3 = nn.ConvTranspose2d(256, 1, 6, stride=2, padding=2)

    def forward(self, x):
        """
        Project noise to featureMap * width * height,
        Batch Normalization after convulation but not at output layer,
        ReLU activation function.
        """
        x = self.fc1(x)
        x = x.view(-1, 1024, 4, 4)
        x = F.relu(self.BN1(self.conv1(x)))
        x = F.relu(self.BN2(self.conv2(x)))
        x = F.tanh(self.conv3(x))

        return x
