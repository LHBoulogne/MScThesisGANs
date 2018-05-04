import torch
import torch.nn as nn
from gan.model.helper.weight_init import *
from gan.model.helper.layers import *

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.auxclas = config.auxclas
        self.coupled = config.coupled
        c_len = 0
        c_len = 0
        if config.auxclas:
            if config.dataname == 'CelebA' and config.labeltype == 'bool':
                c_len = 1
            else :
                c_len = config.categories

        mult = 2**config.blocks
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh())

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return (y,)