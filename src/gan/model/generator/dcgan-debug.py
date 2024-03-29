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
            c_len = config.categories

        mult = 2**config.blocks
        layers = (
            Vector2FeatureMaps(config.z_len+c_len, config.g_dim*mult, config.g_first_layer),
            Norm2d(config.g_dim*mult, config.g_norm),
            nn.ReLU()
            )

        for it in range(config.blocks-1, -1, -1):
            mult = 2**it
            layers += (
                nn.ConvTranspose2d(config.g_dim*mult*2, config.g_dim*mult, 4, stride=2, padding=1),
                Norm2d(config.g_dim*mult, config.g_norm),
                nn.ReLU()
                )

        self.main = nn.Sequential(*layers)

        self.last_a = self.get_last_layers(config)
        if config.coupled:
            self.last_b = self.get_last_layers(config)

        weight_init(self, config.weight_init)

    def get_last_layers(self, config):
        if config.g_extra_conv:
            last = nn.Sequential(
                nn.ConvTranspose2d(config.g_dim, config.g_dim//2, 4, stride=2, padding=1),
                Norm2d(config.g_dim//2, config.g_norm),
                nn.ReLU(),

                nn.Conv2d(config.g_dim//2, config.imgch, 3, stride=1, padding=1),
                nn.Tanh()
            )
        else:
            last = nn.Sequential(
                nn.ConvTranspose2d(config.g_dim, config.imgch, 4, stride=2, padding=1),
                nn.Tanh()
            )
        return last

    def forward(self, z, c_a=None, c_b=None): #for accogans and cogans
        print('WWWWWWWWW  GENERATOR WWWWWWWWW')
        if self.auxclas:
            inp_a = torch.cat((z, c_a), 1)
            if self.coupled:
                inp_b = torch.cat((z, c_b), 1)
        else:
            inp_a = z
            if self.coupled:
                inp_b = z
        print(type(z))
        print(z)

        features_a = self.main(inp_a)

        print(type(features_a))
        print(features_a)

        out_a = self.last_a(features_a)

        print(type(out_a))
        print(out_a)
        
        if self.coupled:
            features_b = self.main(inp_b)
            out_b = self.last_b(features_b)
            return out_a, out_b
        
        return (out_a,)
        
