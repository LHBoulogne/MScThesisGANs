import torch
import torch.nn as nn
from gan.model.helper.layers import *
# Augmented code from https://github.com/mingyuliutw/CoGAN/blob/master/cogan_pytorch/src/net_cogan_mnistedge.py

# Discriminator Model
# original paper params: 
# dim = 10
# imgch = 1
class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()

        clen = 0
        self.numcats = len(config.categories)

        self.conv0 = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=0)
        #24x24
        self.pool0 = nn.MaxPool2d(kernel_size=2)
        #12x12
        self.conv1 = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=0)
        #8x8
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        #4x4
        self.conv2 = nn.Conv2d(1, 1, kernel_size=4, stride=1, padding=0)
        #1x1
        self.prelu2 = nn.PReLU()
        self.conv3 = ()
        for it in range(self.numcats):
            self.conv3 += (FeatureMaps2Vector(1, config.categories[it], 'conv', kernel_size=1),)

    def forward(self, inp):
        h0 = self.pool0(self.conv0(inp))
        h1 = self.pool1(self.conv1(h0))
        h2 = self.prelu2(self.conv2(h1))
        out_c = ()
        for it in range(self.numcats):
            out_c += (self.conv3[it](h2),)
        return out_c
        