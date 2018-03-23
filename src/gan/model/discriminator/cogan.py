import torch
import torch.nn as nn
from gan.model.helper.layers import *
# Augmented code from https://github.com/mingyuliutw/CoGAN/blob/master/cogan_pytorch/src/net_cogan_mnistedge.py

# Discriminator Model
# original paper params: 
# d_dim = 10
# imgch = 1
# d_last_layer = 'conv'
class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.auxclas = config.auxclas
        self.conv0_a = nn.Conv2d(1, config.d_dim*2, kernel_size=5, stride=1, padding=0)
        self.conv0_b = nn.Conv2d(1, config.d_dim*2, kernel_size=5, stride=1, padding=0)
        #24x24
        self.pool0 = nn.MaxPool2d(kernel_size=2)
        #12x12
        self.conv1 = nn.Conv2d(config.d_dim*2, config.d_dim*5, kernel_size=5, stride=1, padding=0)
        #8x8
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        #4x4
        self.conv2 = nn.Conv2d(config.d_dim*5, config.d_dim*50, kernel_size=4, stride=1, padding=0)
        #1x1
        self.prelu2 = nn.PReLU()
        self.conv3 = FeatureMaps2Vector(config.d_dim*50, 1, config.d_last_layer, kernel_size=1)
        self.sigm = nn.Sigmoid()
        if config.auxclas:
            self.conv3c = FeatureMaps2Vector(config.d_dim*50, config.categories, config.d_last_layer, kernel_size=1)


    def single_forward(self, inp, conv0):
        h0 = self.pool0(conv0(inp))
        h1 = self.pool1(self.conv1(h0))
        h2 = self.prelu2(self.conv2(h1))
        h3 = self.conv3(h2)
        out = self.sigm(h3)
        if self.auxclas:
            out_c = self.conv3c(h2)
            return (out, out_c)
        return (out,)
        
    def forward(self, inp_a=None, inp_b=None):
        if not inp_a is None:
            out_a = self.single_forward(inp_a, self.conv0_a)

        if not inp_b is None:
            out_b = self.single_forward(inp_b, self.conv0_b)            
            
            if not inp_a is None:            
                return (out_a, out_b)
            return (out_b,)

        return (out_a,)