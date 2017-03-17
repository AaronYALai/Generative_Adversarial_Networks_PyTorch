# -*- coding: utf-8 -*-
# @Author: aaronlai

import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(1 * 28 * 28, 256)
        self.drop1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.drop1(F.leaky_relu(self.fc1(x)))
        x = self.drop2(F.leaky_relu(self.fc2(x)))
        x = F.sigmoid(self.fc3(x))
        return x


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1 * 28 * 28)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = F.tanh(self.fc4(x))
        return x
