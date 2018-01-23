import torch
import torch.nn as nn
from gan.model.helper.weight_init import *
from gan.model.helper.layers import *

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__() 
        self.auxclas = config.auxclas

        self.first_a = nn.Sequential(
            nn.Conv2d(config.imgch, config.d_dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(config.d_dim),
            nn.LeakyReLU(0.2)
        )
        
        if config.coupled:
            self.first_b = nn.Sequential(
                nn.Conv2d(config.imgch, config.d_dim, 4, stride=2, padding=1),
                nn.BatchNorm2d(config.d_dim),
                nn.LeakyReLU(0.2)
            )
        
        self.main = nn.Sequential(
            nn.Conv2d(config.d_dim,  config.d_dim*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(config.d_dim*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(config.d_dim*2, config.d_dim*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(config.d_dim*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(config.d_dim*4, config.d_dim*8, 4, stride=2, padding=1),
            nn.BatchNorm2d(config.d_dim*8),
            nn.LeakyReLU(0.2)
        )

        self.predict_src = nn.Sequential(
            nn.Conv2d(config.d_dim*8, 1, 4, stride=1, padding=0),
            nn.Sigmoid()
        )

        if self.auxclas:
            self.predict_class = nn.Sequential(
                nn.Conv2d(config.d_dim*8, config.categories, 4, stride=1, padding=0),
                Reshape(-1, config.categories)
                )

    def single_forward(self, inp, first):
        hidden = first(inp)
        hidden = self.main(hidden)
        s = self.predict_src(hidden)
        if self.auxclas:
            c = self.predict_class(hidden)
            return s,c
        return s

    def forward(self, inp_a, inp_b=None):
        out_a = self.single_forward(inp_a, self.first_a)
        if not inp_b is None:
            out_b = self.single_forward(inp_b, self.first_b)
            return (out_a, out_b)
        return out_a