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
        if config.auxclas:
            c_len = sum(config.categories)

        mult = 2**config.blocks
        first_fm_size = 4
        layers = (
            nn.Linear(config.z_len+c_len, config.g_dim*mult*first_fm_size*first_fm_size, bias=False),
            Norm1D(config.g_dim*mult*first_fm_size*first_fm_size, config.g_norm),
            Activation(config.g_act),
            Reshape(-1, config.g_dim*mult, first_fm_size, first_fm_size)
            )

        for it in range(config.blocks-1, -1, -1):
            mult = 2**it
            layers += (
                nn.ConvTranspose2d(config.g_dim*mult*2, config.g_dim*mult, 5, 2,
                    padding=2, output_padding=1, bias=False),
                Norm2d(config.g_dim*mult, config.g_norm),
                Activation(config.g_act)
                )

        self.main = nn.Sequential(*layers)

        self.last_a = self.get_last_layers(config.g_dim)
        if config.coupled:
            self.last_b = self.get_last_layers(config.g_dim)

        weight_init(self, config.weight_init)

    def get_last_layers(self, in_dim):
        last = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh()
        )
        return last

    def forward(self, z, c_a=None, c_b=None): #for accogans and cogans
        if self.auxclas:
            inp_a = torch.cat((z, c_a), 1)
            if self.coupled:
                inp_b = torch.cat((z, c_b), 1)
        else:
            inp_a = z
            if self.coupled:
                inp_b = z

        features_a = self.main(inp_a)
        out_a = self.last_a(features_a)

        if self.coupled:
            features_b = self.main(inp_b)
            out_b = self.last_b(features_b)
            return out_a, out_b

        return (out_a,)
        
