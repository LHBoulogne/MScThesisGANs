import torch
import torch.nn as nn

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.conv0_a = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0)
        self.conv0_b = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0)
        self.pool0 = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(50, 500, kernel_size=4, stride=1, padding=0)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(500, 1, kernel_size=1, stride=1, padding=0)
        self.sigm = nn.Sigmoid()
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
        return (out_a, out_b)