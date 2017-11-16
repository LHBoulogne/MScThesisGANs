import os

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

import torchvision.datasets as dset
import torchvision.transforms as transforms

import numpy as np

from argreader import *

class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, inp):
        return inp.view(self.shape)

def int_to_one_hot(categories, y):
    y = np.array([y])
    y = torch.from_numpy(y)
    return to_one_hot(categories, y)


def to_one_hot(categories, y):
    if categories == 0 or categories is None:
        return []

    batch_size = len(y)

    y = y.view(-1,1)
    onehot = torch.FloatTensor(batch_size, categories)
    torch.zeros(batch_size, categories, out=onehot)
    onehot.scatter_(1, y, 1)
    return onehot

def rescale(t):
    return t.div_(127.5).add_(-1)


def xavier_weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0.1)

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, categories):
        super(Discriminator, self).__init__()
        self.is_conditional = categories > 0
        self.categories = categories

        self.main = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(50, 500, kernel_size=4, stride=1, padding=0),
            nn.PReLU(),
        )
        self.predict_src = nn.Sequential(
            nn.Conv2d(500, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        if self.is_conditional:
            self.predict_class = nn.Conv2d(500, categories, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.main(x)
        out = self.predict_src(x)
        out = out.view(-1)
        if self.is_conditional:
            c = self.predict_class(x)
            c = c.view(-1, self.categories)
            return out, c
        return out


# Generator Model
class Generator(nn.Module):
    def __init__(self, z_len, categories):
        super(Generator, self).__init__()
        self.is_conditional = categories > 0
        self.z_len = z_len
        self.categories = categories

        if self.is_conditional:
            noise_channels = 512
            cond_channels = 512
            self.encode_c = nn.Sequential(
                nn.ConvTranspose2d(categories, cond_channels, kernel_size=4, stride=1),
                nn.BatchNorm2d(cond_channels, affine=False),
                nn.PReLU()
                )
        else :
            noise_channels = 1024

        self.encode_z = nn.Sequential(
            nn.ConvTranspose2d(z_len, noise_channels, kernel_size=4, stride=1),
            nn.BatchNorm2d(cond_channels, affine=False),
            nn.PReLU()
            )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512, affine=False),
            nn.PReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256, affine=False),
            nn.PReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128, affine=False),
            nn.PReLU(),
            nn.ConvTranspose2d(128, 1, kernel_size=6, stride=1, padding=1),
            nn.Tanh())

    def forward(self, noise, cond):
        noise = noise.view(-1, self.z_len, 1, 1)
        cond = cond.view(-1, self.categories, 1, 1)
        x = self.encode_z(noise)
        
        if self.is_conditional:
            c = self.encode_c(cond)
            x = torch.cat((x, c), 1)

        return self.main(x)
        return out


# ACGAN for 64/64 imgss

#fm: feature multiple
#imgch: color chanels in produced image
class ACGAN():

    def __init__(self, categories=0, z_len=100, g_fm=128, d_fm=128, imgch=1, loadfolder=None):
        self.categories = categories
        self.z_len = z_len
        self.imgch = imgch
        self.D = Discriminator(categories)
        self.G = Generator(z_len, categories)
        
        self.G.apply(xavier_weights_init)
        self.D.apply(xavier_weights_init)

    def generate(self, inp):
        return self.G(inp)

    def save_images(self, noise, dim, epoch, batch, savefolder) :
        if self.categories:
            dim = noise.size()[0]
            c = np.repeat(range(dim), dim)
            c_tensor = torch.from_numpy(c)
            one_hot = to_one_hot(self.categories, c_tensor)
            generator_input = noise.repeat(dim,1)
            generator_input = (generator_input, Variable(one_hot))
        else :
            generator_input = (noise,)

        fake = self.G(*generator_input).data.numpy()
            
        
        x = fake.shape[2] 
        y = fake.shape[3]

        image = np.empty((dim*x, dim*y, 3))
        
        for ity in range(dim):
            for itx in range(dim):
                xstart = itx*x
                ystart = ity*y
                image[xstart:xstart+x,ystart:ystart+y] = np.swapaxes(fake[itx+dim*ity,:,:,:],0,2)
        np.save(savefolder + '/' + str(epoch) + '_' + str(batch).zfill(7), image)

    def save(self, savefolder) :
        torch.save(self.G.state_dict(), os.path.join(savefolder, 'generator.h5'))
        torch.save(self.D.state_dict(), os.path.join(savefolder, 'discriminator.h5'))
        
    def fake_conditional_tensors(self, batch_size) :
        rands = np.random.randint(self.categories, size=(batch_size,))
        rands = torch.from_numpy(rands)
        return rands
        
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def train(self, dataname="MNIST", mini_batch_size=128, k=1, nr_epochs=50, vis_step=10, vis_dim=10, savename=""):
        savefolder="save_data_diffarch_" + savename
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)

        #Get dataset ready
        if dataname == "MNIST":
            dataset = dset.MNIST(root='./data/mnist', download=True, 
                  transform=transforms.Compose([transforms.Scale(28),
                                                transforms.ToTensor(),
                                                transforms.Lambda(rescale)]))
        elif dataname == "CelebA":
            dataset = dset.ImageFolder(root='./data/celeba', 
                  transform=transforms.Compose([transforms.Scale(64),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]))

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=mini_batch_size, shuffle=True, num_workers=3)

        #Initialize Variables
        if vis_step != 0:
            if self.categories:
                vis_noise_len = vis_dim
            else :
                vis_noise_len = vis_dim*vis_dim
            visualization_noise = torch.FloatTensor(vis_noise_len, self.z_len)
            visualization_noise.uniform_(-1,1)
            visualization_noise = Variable(visualization_noise)

        x_real = torch.FloatTensor(mini_batch_size, self.imgch, 64, 64)
        x_real_v = Variable(x_real)
        
        y_real = torch.ones(mini_batch_size).mul_(0.9)
        y_real_v = Variable(y_real)
        y_fake = torch.zeros(mini_batch_size)
        y_fake_v = Variable(y_fake)

        z = torch.FloatTensor(mini_batch_size, self.z_len)
        z_v = Variable(z)

        # LongTensor with index for crossentropyloss function
        c_real = torch.LongTensor(mini_batch_size)
        c_real_v = Variable(c_real)
        c_fake = torch.LongTensor(mini_batch_size)
        c_fake_v = Variable(c_fake)

        #onehot vector for generator input
        c_fake_one_hot = torch.FloatTensor(mini_batch_size, self.categories)
        c_fake_one_hot_v = Variable(c_fake_one_hot)

        #init misc
        s_criterion = nn.BCELoss()
        c_criterion = nn.CrossEntropyLoss() #Includes the softmax function
        d_opt = optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        g_opt = optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))

        #Do actual training
        steps_without_G_update = 0
        print("Starting training:")
        for epoch in range(nr_epochs):
            for batch, data in enumerate(dataloader) :
                ### Print progress ###
                print("\rEpoch: "+str(epoch+1)+ "/" + str(nr_epochs)+", Batch " + str(batch+1), end='\r')

                ### Determine batch size for this batch (last batch size can be off):
                x_data, c_data = data
                this_batch_size = x_data.size(0)

                ### Discriminator training ####
                self.D.zero_grad()
                # Get input for G

                z.resize_(this_batch_size, self.z_len).uniform_(-1,1)

                generator_input = (z_v,)
                if self.categories: #for conditional input
                    c_fake.resize_(this_batch_size)
                    c_fake.copy_(self.fake_conditional_tensors(this_batch_size))
                    c_fake_one_hot.resize_(this_batch_size, self.categories)
                    c_fake_one_hot.copy_(to_one_hot(self.categories, c_fake))
                    generator_input += (c_fake_one_hot_v,) 
                    
                # Get input for D
                x_fake_v = self.G(*generator_input)
                x_real.resize_as_(x_data).copy_(x_data)
                
                discriminator_input_real = x_real_v
                discriminator_input_fake = x_fake_v.detach() #stop the backward pass at generator output
                    
                # Update D
                y_real.resize_(this_batch_size)
                y_fake.resize_(this_batch_size)


                if self.categories: #for conditional input
                    (verdict_real, class_probs_real) = self.D(discriminator_input_real)
                else :
                    verdict_real = self.D(discriminator_input_real)
                    
                error_D_real = s_criterion(verdict_real.view(-1), y_real_v)  #reshape verdict: [?,1] -> [?]
                
                if self.categories: # add loss for classc_criterion() prediction
                    c_real.resize_as_(c_data).copy_(c_data)
                    error_D_real += c_criterion(class_probs_real, c_real_v)

                error_D_real.backward()

                if self.categories: #for conditional input
                    (verdict_fake, class_probs_fake) = self.D(discriminator_input_fake)
                else :
                    verdict_fake = self.D(discriminator_input_fake)
                    
                error_D_fake = s_criterion(verdict_fake.view(-1), y_fake_v)
                if self.categories: # add loss for classc_criterion() prediction
                    error_D_fake += c_criterion(class_probs_fake, c_fake_v)

                error_D_fake.backward()

                error_D = error_D_real + error_D_fake #TODO save this error (maybe also save classification error by itself)
                d_opt.step()
                
                ### Generator training ####
                steps_without_G_update += 1
                if steps_without_G_update == k: #Allow for multiple updates of D with respect to G
                    steps_without_G_update = 0
                    self.G.zero_grad()
                    
                    # Get input for G
                    z.uniform_(-1,1)
                    generator_input = (z_v,)
                    if self.categories: #for conditional input
                        c_fake.copy_(self.fake_conditional_tensors(this_batch_size))
                        c_fake_one_hot.copy_(to_one_hot(self.categories, c_fake))
                        generator_input += (c_fake_one_hot_v,)

                    # Get input for D
                    x_fake_v = self.G(*generator_input)
                    discriminator_input_fake = x_fake_v #continue the backward pass through generator

                    # Update G   
                    if self.categories: #for conditional input
                        verdict_fake, class_probs_fake = self.D(discriminator_input_fake)
                    else :
                        verdict_fake = self.D(discriminator_input_fake)
                    
                    error_G = s_criterion(verdict_fake.view(-1), y_real_v) #use real labels
                    if self.categories:
                        error_G += c_criterion(class_probs_fake, c_fake_v)
                    
                    error_G.backward()
                    g_opt.step()

                ### Save training image and current G and D ###
                if vis_step!=0 and batch%vis_step == 0:
                    self.save_images(visualization_noise, vis_dim, epoch, batch, savefolder=savefolder)
                    self.save(savefolder)

            print("Epoch: "+str(epoch+1)+ "/" + str(nr_epochs)+", Batch " + str(batch+1))

        self.save_images(visualization_noise, vis_dim, epoch, batch, savefolder=savefolder)
        self.save(savefolder)


ar = ArgReader()
categorical = ar.next_arg()
if categorical == "True":
    categories = 10
else:
    categories = 0

dataname = ar.next_arg()
if dataname == "MNIST":
    imgch = 1
else :
    imgch = 3


print(ar.arg_string)
acgan = ACGAN(categories, imgch=imgch)
acgan.train(dataname=dataname, savename=ar.arg_string)
