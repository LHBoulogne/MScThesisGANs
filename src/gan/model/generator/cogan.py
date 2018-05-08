
import torch
import torch.nn as nn
from gan.model.helper.weight_init import *
from gan.model.helper.layers import *
# Augmented code from https://github.com/mingyuliutw/CoGAN/blob/master/cogan_pytorch/src/net_cogan_mnistedge.py

# Generator Model
# Original paper parameters: 
# g_dim = 128
# g_first_layer = 'conv'
class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.auxclas = config.auxclas
        self.coupled = config.coupled
        c_len = 0
        if config.auxclas:
            c_len = sum(config.categories)

        self.dconv0 = Vector2FeatureMaps(config.z_len+c_len, config.g_dim*8, mode=config.g_first_layer)
        self.bn0 = nn.BatchNorm2d(config.g_dim*8, affine=False)
        self.prelu0 = nn.PReLU()
        self.dconv1 = nn.ConvTranspose2d(config.g_dim*8, config.g_dim*4, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(config.g_dim*4, affine=False)
        self.prelu1 = nn.PReLU()
        self.dconv2 = nn.ConvTranspose2d(config.g_dim*4, config.g_dim*2, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(config.g_dim*2, affine=False)
        self.prelu2 = nn.PReLU()
        self.dconv3 = nn.ConvTranspose2d(config.g_dim*2, config.g_dim, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(config.g_dim, affine=False)
        self.prelu3 = nn.PReLU()
        self.dconv4_a = nn.ConvTranspose2d(config.g_dim, 1, kernel_size=6, stride=1, padding=1)
        if self.coupled:
            self.dconv4_b = nn.ConvTranspose2d(config.g_dim, 1, kernel_size=6, stride=1, padding=1)
        self.sig4 = nn.Tanh()
        
        weight_init(self, config.weight_init)
        

    def singleForward(self, z, c=None):        
        if self.auxclas:
            inp = torch.cat((z, c), 1)
        else:
            inp = z

        inp = inp.view(inp.size(0), inp.size(1), 1, 1)

        h0 = self.prelu0(self.bn0(self.dconv0(inp)))
        h1 = self.prelu1(self.bn1(self.dconv1(h0)))
        h2 = self.prelu2(self.bn2(self.dconv2(h1)))
        h3 = self.prelu3(self.bn3(self.dconv3(h2)))
        return h3
        
    def forward(self, z, c_a=None, c_b=None):
        h3_a = self.singleForward(z, c_a)
        out_a = self.sig4(self.dconv4_a(h3_a))
        if self.coupled:
            h3_b = self.singleForward(z, c_b)
            out_b = self.sig4(self.dconv4_b(h3_b))
            return out_a, out_b
        return (out_a,)