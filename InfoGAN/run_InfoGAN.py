# -*- coding: utf-8 -*-
# @Author: aaronlai

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

from torch.autograd import Variable
from InfoGAN import InfoGAN_Discriminator, InfoGAN_Generator


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


def gen_noise(n_instance, n_dim=2):
    """generate n-dim uniform random noise"""
    return torch.Tensor(np.random.uniform(low=-1.0, high=1.0,
                                          size=(n_instance, n_dim)))


def gen_conti_codes(n_instance, n_conti, mean=0, std=1):
    """generate gaussian continuous codes with specified mean and std"""
    codes = np.random.randn(n_instance, n_conti) * std + mean
    return torch.Tensor(codes)


def gen_discrete_code(n_instance, n_discrete, num_category=10):
    """generate discrete codes with n categories"""
    codes = []
    for i in range(n_discrete):
        code = np.zeros((n_instance, num_category))
        random_cate = np.random.randint(0, num_category, n_instance)
        code[range(n_instance), random_cate] = 1
        codes.append(code)

    codes = np.concatenate(codes, 1)
    return torch.Tensor(codes)


def train_InfoGAN(InfoGAN_Dis, InfoGAN_Gen, D_criterion, G_criterion,
                  D_optimizer, G_optimizer, info_reg_discrete, info_reg_conti,
                  n_conti, n_discrete, mean, std, num_category, trainloader,
                  n_epoch, batch_size, noise_dim,
                  n_update_dis=1, n_update_gen=1, use_gpu=False,
                  print_every=50, update_max=None):
    """train InfoGAN and print out the losses for D and G"""

    for epoch in range(n_epoch):

        D_running_loss = 0.0
        G_running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs from true distribution
            true_inputs, lab = data
            true_inputs = Variable(true_inputs)
            if use_gpu:
                true_inputs = true_inputs.cuda()

            # get inputs (noises and codes) for Generator
            noises = Variable(gen_noise(batch_size, n_dim=noise_dim))
            conti_codes = Variable(gen_conti_codes(batch_size, n_conti,
                                                   mean, std))
            discr_codes = Variable(gen_discrete_code(batch_size, n_discrete,
                                                     num_category))
            if use_gpu:
                noises = noises.cuda()
                conti_codes = conti_codes.cuda()
                discr_codes = discr_codes.cuda()

            # generate fake images
            gen_inputs = torch.cat((noises, conti_codes, discr_codes), 1)
            fake_inputs = InfoGAN_Gen(gen_inputs)
            inputs = torch.cat([true_inputs, fake_inputs])

            # make a minibatch of labels
            labels = np.zeros(2 * batch_size)
            labels[:batch_size] = 1
            labels = torch.from_numpy(labels.astype(np.float32))
            if use_gpu:
                labels = labels.cuda()
            labels = Variable(labels)

            # Discriminator
            D_optimizer.zero_grad()
            outputs = InfoGAN_Dis(inputs)

            # calculate mutual information lower bound L(G, Q)
            for j in range(n_discrete):
                shift = (j * num_category)
                start = 1 + n_conti + shift
                end = start + num_category
                Q_cx_discr = outputs[batch_size:, start:end]
                codes = discr_codes[:, shift:(shift+num_category)]
                condi_entro = -torch.mean(torch.sum(Q_cx_discr * codes, 1))

                if j == 0:
                    L_discrete = -condi_entro
                else:
                    L_discrete -= condi_entro
            L_discrete /= n_discrete

            Q_cx_conti = outputs[batch_size:, 1:(1 + n_conti)]
            L_conti = torch.mean(-(((Q_cx_conti - mean) / std) ** 2))

            # Update Discriminator
            D_loss = D_criterion(outputs[:, 0], labels)
            if n_discrete > 0:
                D_loss = D_loss - info_reg_discrete * L_discrete

            if n_conti > 0:
                D_loss = D_loss - info_reg_conti * L_conti

            if i % n_update_dis == 0:
                D_loss.backward(retain_variables=True)
                D_optimizer.step()

            # Update Generator
            if i % n_update_gen == 0:
                G_optimizer.zero_grad()
                G_loss = G_criterion(outputs[batch_size:, 0],
                                     labels[:batch_size])

                if n_discrete > 0:
                    G_loss = G_loss - info_reg_discrete * L_discrete

                if n_conti > 0:
                    G_loss = G_loss - info_reg_conti * L_conti

                G_loss.backward()
                G_optimizer.step()

            # print statistics
            D_running_loss += D_loss.data[0]
            G_running_loss += G_loss.data[0]
            if i % print_every == (print_every - 1):
                print('[%d, %5d] D loss: %.3f ; G loss: %.3f' %
                      (epoch+1, i+1, D_running_loss / print_every,
                       G_running_loss / print_every))
                D_running_loss = 0.0
                G_running_loss = 0.0

            if update_max and i > update_max:
                break

    print('Finished Training')


def run_InfoGAN(info_reg_discrete=1.0, info_reg_conti=0.5, noise_dim=10,
                n_conti=2, n_discrete=1, mean=0.0, std=0.5, num_category=10,
                n_layer=3, n_channel=1, D_featmap_dim=256, G_featmap_dim=1024,
                n_epoch=2, batch_size=50, use_gpu=False, dis_lr=1e-4,
                gen_lr=1e-3, n_update_dis=1, n_update_gen=1, update_max=None):
    # loading data
    trainloader, testloader = load_dataset(batch_size=batch_size)

    # initialize models
    InfoGAN_Dis = InfoGAN_Discriminator(n_layer, n_conti, n_discrete,
                                        num_category, use_gpu, D_featmap_dim,
                                        n_channel)

    InfoGAN_Gen = InfoGAN_Generator(noise_dim, n_layer, n_conti, n_discrete,
                                    num_category, use_gpu, G_featmap_dim,
                                    n_channel)

    if use_gpu:
        InfoGAN_Dis = InfoGAN_Dis.cuda()
        InfoGAN_Gen = InfoGAN_Gen.cuda()

    # assign loss function and optimizer (Adam) to D and G
    D_criterion = torch.nn.BCELoss()
    D_optimizer = optim.Adam(InfoGAN_Dis.parameters(), lr=dis_lr,
                             betas=(0.5, 0.999))

    G_criterion = torch.nn.BCELoss()
    G_optimizer = optim.Adam(InfoGAN_Gen.parameters(), lr=gen_lr,
                             betas=(0.5, 0.999))

    train_InfoGAN(InfoGAN_Dis, InfoGAN_Gen, D_criterion, G_criterion,
                  D_optimizer, G_optimizer, info_reg_discrete, info_reg_conti,
                  n_conti, n_discrete, mean, std, num_category, trainloader,
                  n_epoch, batch_size, noise_dim,
                  n_update_dis, n_update_gen, use_gpu, update_max=update_max)


if __name__ == '__main__':
    run_InfoGAN(n_conti=2, n_discrete=1, D_featmap_dim=64, G_featmap_dim=128,
                n_epoch=1, batch_size=10, update_max=200)
