# -*- coding: utf-8 -*-
# @Author: aaronlai

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2

from torch.autograd import Variable
from LAPGAN import LAPGAN, gen_noise


def load_dataset(batch_size=10, download=True):
    """
    The output of torchvision datasets are PILImage images of range [0, 1].
    Transform them to Tensors of normalized range [-1, 1]
    """
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.MNIST(root='../data', train=True,
                                          download=download,
                                          transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='../data', train=False,
                                         download=download,
                                         transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    return trainloader, testloader


def train_LAPGAN(LapGan_model, n_level, D_criterions, G_criterions,
                 D_optimizers, G_optimizers, trainloader, n_epoch,
                 batch_size, noise_dim, n_update_dis=1, n_update_gen=1,
                 use_gpu=False, print_every=10, update_max=None):
    """train LAPGAN and print out the losses for Ds and Gs"""
    for epoch in range(n_epoch):

        D_running_losses = [0.0 for i in range(n_level)]
        G_running_losses = [0.0 for i in range(n_level)]

        for ind, data in enumerate(trainloader, 0):
            # get the inputs from true distribution
            true_inputs, lab = data
            down_imgs = true_inputs.numpy()
            n_minibatch, n_channel, _, _ = down_imgs.shape

            for l in range(n_level):
                # calculate input images for models at the particular level
                if l == (n_level - 1):
                    condi_inputs = None
                    true_inputs = Variable(torch.Tensor(down_imgs))
                    if use_gpu:
                        true_inputs = true_inputs.cuda()
                else:
                    new_down_imgs = []
                    up_imgs = []
                    residual_imgs = []

                    # compute a Laplacian Pyramid
                    for i in range(n_minibatch):
                        down_img = []
                        up_img = []
                        residual_img = []

                        for j in range(n_channel):
                            previous = down_imgs[i, j, :]
                            down_img.append(cv2.pyrDown(previous))
                            up_img.append(cv2.pyrUp(down_img[-1]))
                            residual_img.append(previous - up_img[-1])

                        new_down_imgs.append(down_img)
                        up_imgs.append(up_img)
                        residual_imgs.append(residual_img)

                    down_imgs = np.array(new_down_imgs)
                    up_imgs = np.array(up_imgs)
                    residual_imgs = np.array(residual_imgs)

                    condi_inputs = Variable(torch.Tensor(up_imgs))
                    true_inputs = Variable(torch.Tensor(residual_imgs))
                    if use_gpu:
                        condi_inputs = condi_inputs.cuda()
                        true_inputs = true_inputs.cuda()

                # get inputs for discriminators from generators and real data
                noise = Variable(gen_noise(batch_size, noise_dim))
                if use_gpu:
                    noise = noise.cuda()
                fake_inputs = LapGan_model.Gen_models[l](noise, condi_inputs)
                inputs = torch.cat([true_inputs, fake_inputs])
                labels = np.zeros(2 * batch_size)
                labels[:batch_size] = 1
                labels = Variable(torch.from_numpy(labels.astype(np.float32)))
                if use_gpu:
                    labels = labels.cuda()

                # Discriminator
                D_optimizers[l].zero_grad()
                if condi_inputs:
                    condi_inputs = torch.cat((condi_inputs, condi_inputs))
                outputs = LapGan_model.Dis_models[l](inputs, condi_inputs)
                D_loss = D_criterions[l](outputs[:, 0], labels)

                if ind % n_update_dis == 0:
                    D_loss.backward(retain_variables=True)
                    D_optimizers[l].step()

                # Generator
                if ind % n_update_gen == 0:
                    G_optimizers[l].zero_grad()
                    G_loss = G_criterions[l](outputs[batch_size:, 0],
                                             labels[:batch_size])
                    G_loss.backward()
                    G_optimizers[l].step()

                # print statistics
                D_running_losses[l] += D_loss.data[0]
                G_running_losses[l] += G_loss.data[0]
                if ind % print_every == (print_every - 1):
                    print('[%d, %5d, %d] D loss: %.3f ; G loss: %.3f' %
                          (epoch+1, ind+1, l+1,
                           D_running_losses[l] / print_every,
                           G_running_losses[l] / print_every))
                    D_running_losses[l] = 0.0
                    G_running_losses[l] = 0.0

            if update_max and ind > update_max:
                break

    print('Finished Training')


def run_LAPGAN(n_level=3, n_epoch=2, batch_size=20, use_gpu=False,
               dis_lrs=None, gen_lrs=None, n_update_dis=1, n_update_gen=1,
               noise_dim=10, n_condition=100, D_featmap_dim=128,
               condi_D_featmap_dim=128, G_featmap_dim=256,
               condi_G_featmap_dim=128, n_channel=1, n_sample=25,
               update_max=None):
    # loading data
    trainloader, testloader = load_dataset(batch_size=batch_size)

    # initialize models
    LapGan_model = LAPGAN(n_level, noise_dim, n_condition, D_featmap_dim,
                          condi_D_featmap_dim, G_featmap_dim,
                          condi_G_featmap_dim, use_gpu, n_channel)

    # assign loss function and optimizer (Adam) to D and G
    D_criterions = []
    G_criterions = []

    D_optimizers = []
    G_optimizers = []

    if not dis_lrs:
        dis_lrs = [0.0002, 0.0003, 0.001]

    if not gen_lrs:
        gen_lrs = [0.001, 0.005, 0.01]

    for l in range(n_level):
        D_criterions.append(nn.BCELoss())
        D_optim = optim.Adam(LapGan_model.Dis_models[l].parameters(),
                             lr=dis_lrs[l], betas=(0.5, 0.999))
        D_optimizers.append(D_optim)

        G_criterions.append(nn.BCELoss())
        G_optim = optim.Adam(LapGan_model.Gen_models[l].parameters(),
                             lr=gen_lrs[l], betas=(0.5, 0.999))
        G_optimizers.append(G_optim)

    train_LAPGAN(LapGan_model, n_level, D_criterions, G_criterions,
                 D_optimizers, G_optimizers, trainloader, n_epoch,
                 batch_size, noise_dim, n_update_dis, n_update_gen,
                 update_max=update_max)

    return LapGan_model.generate(n_sample)


if __name__ == '__main__':
    run_LAPGAN(n_epoch=1, D_featmap_dim=64, condi_D_featmap_dim=64,
               G_featmap_dim=128, condi_G_featmap_dim=64, update_max=50)
