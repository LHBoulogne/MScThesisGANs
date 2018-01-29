
import torch
import torch.nn as nn

# Generator Model
class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.dconv0 = nn.ConvTranspose2d(config.z_len, 1024, kernel_size=4, stride=1)
        self.bn0 = nn.BatchNorm2d(1024, affine=False)
        self.prelu0 = nn.PReLU()
        self.dconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(512, affine=False)
        self.prelu1 = nn.PReLU()
        self.dconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256, affine=False)
        self.prelu2 = nn.PReLU()
        self.dconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128, affine=False)
        self.prelu3 = nn.PReLU()
        self.dconv4_a = nn.ConvTranspose2d(128, 1, kernel_size=6, stride=1, padding=1)
        self.dconv4_b = nn.ConvTranspose2d(128, 1, kernel_size=6, stride=1, padding=1)
        self.sig4_a = nn.Tanh()
        self.sig4_b = nn.Tanh()

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        h0 = self.prelu0(self.bn0(self.dconv0(z)))
        h1 = self.prelu1(self.bn1(self.dconv1(h0)))
        h2 = self.prelu2(self.bn2(self.dconv2(h1)))
        h3 = self.prelu3(self.bn3(self.dconv3(h2)))
        out_a = self.sig4_a(self.dconv4_a(h3))
        out_b = self.sig4_b(self.dconv4_b(h3))
        return out_a, out_b