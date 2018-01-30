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
    def forward(self, x_a, x_b):
        
        h0_a = self.pool0(self.conv0_a(x_a))
        h1_a = self.pool1(self.conv1(h0_a))
        h2_a = self.prelu2(self.conv2(h1_a))
        h3_a = self.conv3(h2_a)
        out_a = self.sigm(h3_a)

        h0_b = self.pool0(self.conv0_b(x_b))
        h1_b = self.pool1(self.conv1(h0_b))
        h2_b = self.prelu2(self.conv2(h1_b))
        h3_b = self.conv3(h2_b)
        out_b = self.sigm(h3_b)

        if self.auxclas:
            out_a_c = self.conv3c(h2_a)
            out_b_c = self.conv3c(h2_b)
            return (out_a, out_a_c), (out_b, out_b_c)
        return (out_a, out_b)