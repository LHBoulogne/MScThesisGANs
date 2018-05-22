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
        self.use_dropout = config.dropout
        self.relu = nn.ReLU()
        if self.use_dropout:
            self.dropout1d = nn.Dropout()
            self.dropout2d = nn.Dropout2d()

        self.conv0 = nn.Conv2d(1, config.dim*2, kernel_size=5, stride=1, padding=0)
        #24x24
        self.pool0 = nn.MaxPool2d(kernel_size=2)
        #12x12
        self.conv1 = nn.Conv2d(config.dim*2, config.dim*5, kernel_size=5, stride=1, padding=0)
        #8x8
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        #4x4
        self.flatten = Reshape(-1, 4*4*config.dim*5)
        self.fc1 = nn.Linear(4*4*config.dim*5, config.dim*50)
        self.fc2 = ()
        for it in range(self.numcats):
            self.fc2 += (nn.Linear(config.dim*50, config.categories[it]),)

    def forward(self, inp):
        h0 = self.relu(self.pool0(self.conv0(inp)))
        h1 = self.relu(self.pool1(self.conv1(h0)))
        if self.use_dropout:
            h1 = self.dropout2d(h1)
        h2 = self.relu(self.fc1(self.flatten(h1)))
        if self.use_dropout:
            h2 = self.dropout1d(h2)
        out_c = ()
        for it in range(self.numcats):
            out_c += (self.fc2[it](h2),)
        return out_c
        