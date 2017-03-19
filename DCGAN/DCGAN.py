# -*- coding: utf-8 -*-
# @Author: aaronlai

import torch.nn as nn
import torch.nn.functional as F


class DCGAN_Discriminator(nn.Module):

    def __init__(self):
        super(DCGAN_Discriminator, self).__init__()

    def forward(self, x):
        return x


class DCGAN_Generator(nn.Module):

    def __init__(self):
        super(DCGAN_Generator, self).__init__()

    def forward(self, x):
        return x
