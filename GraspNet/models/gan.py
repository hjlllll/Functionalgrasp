#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Define network blocks
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Zhutq & Wrina - 15/11/2020
#


import time
import math
import torch
from torch import nn,optim,autograd
from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_uniform_
from kernels.kernel_points import load_kernels
import numpy as np

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(128+4+3+17, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, input):
        out = self.net(input)
        return out


def compute_gradient_penalty(cuda, D, real_samples, fake_samples):
    alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1))).cuda()
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = autograd.Variable(torch.FloatTensor(real_samples.shape[0], 1).cuda().fill_(1.0), requires_grad=False)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) -1)**2).mean()
    return gradient_penalty