import os, time, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np

# G(z)
class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, inp):
        return inp.view(self.shape)

class Generator(nn.Module):
    def __init__(self, fm, categories=0):
        z_len = 100
        imgch = 3
        super(Generator, self).__init__()
        self.is_conditional = categories > 0

        if self.is_conditional:
            noise_channels = 4
            cond_channels = 4
            self.encode_c = nn.Sequential(
                nn.Linear(categories, (fm*cond_channels)*4*4),
                nn.BatchNorm2d((fm*cond_channels)*4*4),
                nn.ReLU(),
                Reshape(-1, fm*cond_channels, 4, 4)
                )
        else :
            noise_channels = 8

        self.encode_z = nn.Sequential(
            nn.Linear(z_len, (fm*noise_channels)*4*4),
            nn.BatchNorm2d((fm*noise_channels)*4*4),
            nn.ReLU(),
            Reshape(-1, fm*noise_channels, 4, 4)
            )


        self.main = nn.Sequential(
            nn.ConvTranspose2d(fm*8, fm*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(fm*4),
            nn.ReLU(),

            nn.ConvTranspose2d(fm*4, fm*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(fm*2),
            nn.ReLU(),

            nn.ConvTranspose2d(fm*2, fm, 4, stride=2, padding=1),
            nn.BatchNorm2d(fm),
            nn.ReLU(),

            nn.ConvTranspose2d(fm, imgch, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, noise, cond=None):
        noise = noise.view(-1, 100)
        x = self.encode_z(noise)
        
        if self.is_conditional:
            c = self.encode_c(cond)
            x = torch.cat((x, c), 1)

        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, fm, categories=0):
        super(Discriminator, self).__init__() 
        self.is_conditional = categories > 0
        imgch = 3
        self.main = nn.Sequential(
            nn.Conv2d(imgch, fm, 4, stride=2, padding=1),
            nn.BatchNorm2d(fm),
            nn.LeakyReLU(0.2),

            nn.Conv2d(fm,  fm*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(fm*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(fm*2, fm*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(fm*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(fm*4, fm*8, 4, stride=2, padding=1),
            nn.BatchNorm2d(fm*8),
            nn.LeakyReLU(0.2),

            Reshape(-1, (fm*8)*4*4)
        )

        self.predict_src = nn.Sequential(
            nn.Linear((fm*8)*4*4, 1),
            nn.Sigmoid()
        )

        if self.is_conditional:
            self.predict_class = nn.Linear((fm*8)*4*4, categories)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, inp):
        x = self.main(inp)
        out = self.predict_src(x)
        if self.is_conditional:
            c = self.predict_class(x)
            return out, c
        return out


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

fixed_z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)    # fixed noise
fixed_z_ = Variable(fixed_z_, volatile=True)

# training parameters
batch_size = 128
lr = 0.0002
train_epoch = 20

# data_loader
img_size = 64
isCrop = False
if isCrop:
    transform = transforms.Compose([
        transforms.Scale(108),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
else:
    transform = transforms.Compose([
        transforms.Scale(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
transform = transforms.Compose([
        transforms.Scale(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
data_dir = '../../acgan-pytorch/data/celeba/'          # this path depends on your computer
dset = datasets.ImageFolder(data_dir, transform)
train_loader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=True, num_workers=3)

# network
#G = generator(128) #TODO PUT THIS BACK
#D = discriminator(128)

G = Generator(64)
D = Discriminator(64)

G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# results save folder
if not os.path.isdir('CelebA_DCGAN_results'):
    os.mkdir('CelebA_DCGAN_results')
if not os.path.isdir('CelebA_DCGAN_results/Random_results'):
    os.mkdir('CelebA_DCGAN_results/Random_results')
if not os.path.isdir('CelebA_DCGAN_results/Fixed_results'):
    os.mkdir('CelebA_DCGAN_results/Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []




z_len = 100
vis_dim = 10
vis_step = 50
vis_noise_len = vis_dim*vis_dim
visualization_noise = torch.FloatTensor(vis_noise_len, z_len)
#visualization_noise.uniform_(-1,1)
visualization_noise = torch.randn((vis_noise_len, z_len)).view(-1, z_len, 1, 1)
visualization_noise = Variable(visualization_noise)
savefolder = 'savedata/'
if not os.path.exists(savefolder):
    os.makedirs(savefolder)

def save_images(G, D, noise, dim, epoch, batch, savefolder) :
    generator_input = (noise,)

    fake = G(*generator_input).data.numpy()
        
    
    x = fake.shape[2] 
    y = fake.shape[3]

    image = np.empty((dim*x, dim*y, 3))
    
    for ity in range(dim):
        for itx in range(dim):
            xstart = itx*x
            ystart = ity*y
            image[xstart:xstart+x,ystart:ystart+y] = np.swapaxes(fake[itx+dim*ity,:,:,:],0,2)
    np.save(savefolder + '/' + str(epoch) + '_' + str(batch).zfill(7), image)

def save(G, D, savefolder) :
    torch.save(G.state_dict(), os.path.join(savefolder, 'generator.h5'))
    torch.save(D.state_dict(), os.path.join(savefolder, 'discriminator.h5'))



print('Training...')
start_time = time.time()
for epoch in range(train_epoch):
    D_losses = []
    G_losses = []

    # learning rate decay
    if (epoch+1) == 11:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    if (epoch+1) == 16:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    epoch_start_time = time.time()
    for num_iter, (x_, _) in enumerate(train_loader):
        print("\rEpoch: "+str(epoch+1)+ "/" + str(train_epoch)+", Batch " + str(num_iter+1), end='\r')
        # train discriminator D
        D.zero_grad()
        
        if isCrop:
            x_ = x_[:, :, 22:86, 22:86]

        mini_batch = x_.size()[0]

        y_real_ = torch.ones(mini_batch)
        y_fake_ = torch.zeros(mini_batch)

        x_, y_real_, y_fake_ = Variable(x_), Variable(y_real_), Variable(y_fake_)
        D_result = D(x_).squeeze()
        D_real_loss = BCE_loss(D_result, y_real_)

        z_ = torch.randn((mini_batch, z_len)).view(-1, z_len, 1, 1)
        z_ = Variable(z_)
        G_result = G(z_)

        D_result = D(G_result).squeeze()
        D_fake_loss = BCE_loss(D_result, y_fake_)
        D_fake_score = D_result.data.mean()

        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        D_optimizer.step()

        D_losses.append(D_train_loss.data[0])

        # train generator G
        G.zero_grad()

        z_ = torch.randn((mini_batch, z_len)).view(-1, z_len, 1, 1)
        z_ = Variable(z_)

        G_result = G(z_)
        D_result = D(G_result).squeeze()
        G_train_loss = BCE_loss(D_result, y_real_)
        G_train_loss.backward()
        G_optimizer.step()

        G_losses.append(G_train_loss.data[0])

        if num_iter%vis_step == 0:
            save_images(G, D, visualization_noise, vis_dim, epoch, num_iter, savefolder=savefolder)
            save(G, D, savefolder)

save_images(G, D, visualization_noise, vis_dim, epoch, num_iter, savefolder=savefolder)
save(G, D, savefolder)

