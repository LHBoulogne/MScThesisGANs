import os, time, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets as dset
from torch.autograd import Variable

import numpy as np

from argreader import *

sys.path.append("./../../cogan")
from mnistedge import *

sys.path.append("./../../dataset")
from celeba_coupled import *

def rescale(t):
    return t.div_(127.5).add_(-1)

class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, inp):
        return inp.view(self.shape)

def to_one_hot(categories, y):
    if categories == 0 or categories is None:
        return []

    batch_size = len(y)

    y = y.view(-1,1)
    onehot = torch.FloatTensor(batch_size, categories)
    torch.zeros(batch_size, categories, out=onehot)
    onehot.scatter_(1, y, 1)
    return onehot

class Generator(nn.Module):
    def __init__(self, d, imgch=3, z_len=100, categories=0):
        super(Generator, self).__init__()
        self.is_conditional = categories > 0

        self.main = nn.Sequential(
            Reshape(-1, z_len+categories, 1, 1),
            nn.ConvTranspose2d(z_len+categories, d*8, 4, stride=1, padding=0),
            nn.BatchNorm2d(d*8),
            nn.ReLU(),

            nn.ConvTranspose2d(d*8, d*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(d*4),
            nn.ReLU(),

            nn.ConvTranspose2d(d*4, d*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(d*2),
            nn.ReLU(),

            nn.ConvTranspose2d(d*2, d, 4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU()
        )

        self.last_a = nn.Sequential(
            nn.ConvTranspose2d(d, imgch, 4, stride=2, padding=1),
            nn.Tanh()
        )

        self.last_b = nn.Sequential(
            nn.ConvTranspose2d(d, imgch, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z, c=None):
        if self.is_conditional:
            z = torch.cat((z, c), 1)
        
        features = self.main(z)

        return self.last_a(features), self.last_b(features)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class Discriminator(nn.Module):
    def __init__(self, d, imgch, categories=0):
        super(Discriminator, self).__init__() 
        self.is_conditional = categories > 0

        self.first_a = nn.Sequential(
            nn.Conv2d(imgch, d, 4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(0.2)
        )
        
        self.first_b = nn.Sequential(
            nn.Conv2d(imgch, d, 4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(0.2)
        )
        
        self.main = nn.Sequential(
            nn.Conv2d(d,  d*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(d*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(d*2, d*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(d*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(d*4, d*8, 4, stride=2, padding=1),
            nn.BatchNorm2d(d*8),
            nn.LeakyReLU(0.2)
        )

        self.predict_src = nn.Sequential(
            nn.Conv2d(d*8, 1, 4, stride=1, padding=0),
            nn.Sigmoid()
        )

        if self.is_conditional:
            self.predict_class = nn.Sequential(
                nn.Conv2d(d*8, categories, 4, stride=1, padding=0),
                Reshape(-1, categories)
                )


    def forward(self, inp_a, inp_b):
        features_a = self.first_a(inp_a)
        features_b = self.first_b(inp_b)
        features = torch.cat((features_a, features_b), 0)

        x = self.main(features)

        out = self.predict_src(x)
        if self.is_conditional:
            c = self.predict_class(x)
            return out, c
        return out

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
    if isinstance(m, nn.Sequential):
        for mod in m._modules:
            normal_init(m._modules[mod], mean, std)

class CoGAN():
        
    def __init__(self, categories=0, z_len=100, g_d=128, d_d=128, imgch=3, loadfolder=None):
        self.categories = categories
        self.z_len = z_len
        self.imgch = imgch
        self.D = Discriminator(d_d, imgch, categories=categories)
        self.G = Generator(g_d, imgch, categories=categories)
        
        self.G.weight_init(mean=0.0, std=0.02)
        self.D.weight_init(mean=0.0, std=0.02)

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

        fake1, fake2 = self.G(*generator_input)
        fake1 = fake1.data.numpy()
        fake2 = fake2.data.numpy()

        fake1 = np.moveaxis(fake1, 1, 3)
        fake2 = np.moveaxis(fake2, 1, 3)
        fake1 = (fake1+1)/2
        fake2 = (fake2+1)/2
        x = fake1.shape[1]
        y = fake1.shape[2]
        image = np.empty((dim*x*2, dim*y, self.imgch))

        for ity in range(dim):
            for itx in range(dim):
                xstart = itx*x*2
                ystart = ity*y
                image[xstart  :xstart+x  ,ystart:ystart+y] = fake1[itx+dim*ity]
                image[xstart+x:xstart+2*x,ystart:ystart+y] = fake2[itx+dim*ity]
        np.save(savefolder + '/' + str(epoch) + '_' + str(batch).zfill(7), image)

    def save(self, savefolder) :
        torch.save(self.G.state_dict(), os.path.join(savefolder, 'generator.h5'))
        torch.save(self.D.state_dict(), os.path.join(savefolder, 'discriminator.h5'))
        
    def fake_conditional_tensors(self, batch_size) :
        rands = np.random.randint(self.categories, size=(batch_size,))
        rands = torch.from_numpy(rands)
        return rands

    def train(self, dataname="MNIST", mini_batch_size=64, k=1, nr_epochs=20, vis_step=10, vis_dim=6, savename=""):
        savefolder="savedata_" + savename
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)

        #Get dataset ready
        if dataname == "MNIST":
            dataset = MNIST_edge(transform=transforms.Compose([transforms.Scale((64,64)),
                                                transforms.ToTensor(),
                                                transforms.Lambda(rescale)]))
        elif dataname == "CelebA":
            dataset = CelebA_dataset_coupled(colabelname='Male', root='../../../data/celeba/', 
                  transform=transforms.Compose([transforms.Scale((64,64)),
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
            #visualization_noise.uniform_(-1,1)
            visualization_noise = torch.randn((vis_noise_len, self.z_len))
            visualization_noise = Variable(visualization_noise)

        x1_real = torch.FloatTensor(mini_batch_size, self.imgch, 64, 64)
        x1_real_v = Variable(x1_real)
        x2_real = torch.FloatTensor(mini_batch_size, self.imgch, 64, 64)
        x2_real_v = Variable(x2_real)
        
        y_real = torch.ones(mini_batch_size*2).mul_(0.9)
        y_real_v = Variable(y_real)
        y_fake = torch.zeros(mini_batch_size*2)
        y_fake_v = Variable(y_fake)

        #z.resize_(this_batch_size, self.z_len).uniform_(-1,1)
        z = torch.FloatTensor(mini_batch_size, self.z_len)
        z_v = Variable(z)

        # LongTensor with index for crossentropyloss function
        c_real = torch.LongTensor(mini_batch_size*2)
        c_real_v = Variable(c_real)
        c_fake = torch.LongTensor(mini_batch_size*2)
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
                x1_data, x2_data, c_data = data
                this_batch_size = x1_data.size(0)

                ### Discriminator training ####
                self.D.zero_grad()
                # Get input for G

                z_temp = torch.randn((this_batch_size, self.z_len)).view(-1, self.z_len)
                z.resize_as_(z_temp).copy_(z_temp)

                generator_input = (z_v,)
                if self.categories: #for conditional input
                    c_fake_tmp = self.fake_conditional_tensors(this_batch_size)
                    c_fake.resize_(this_batch_size*2)
                    c_fake.copy_(torch.cat((c_fake_tmp,c_fake_tmp), 0))
                    c_fake_one_hot.resize_(this_batch_size, self.categories)
                    c_fake_one_hot.copy_(to_one_hot(self.categories, c_fake_tmp))
                    generator_input += (c_fake_one_hot_v,) 
                    
                # Get input for D
                x1_fake_v, x2_fake_v= self.G(*generator_input)

                x1_real.resize_as_(x1_data).copy_(x1_data)
                x2_real.resize_as_(x2_data).copy_(x2_data)
                
                discriminator_input_real = (x1_real_v, x2_real_v)
                discriminator_input_fake = (x1_fake_v.detach(), x2_fake_v.detach()) #stop the backward pass at generator output

                # Update D
                y_real.resize_(this_batch_size*2)
                y_fake.resize_(this_batch_size*2)


                if self.categories: #for conditional input
                    (verdict_real, class_probs_real) = self.D(*discriminator_input_real)
                else :
                    verdict_real = self.D(*discriminator_input_real)
                    
                error_D_real = s_criterion(verdict_real.view(-1), y_real_v)  #reshape verdict: [?,1] -> [?]
                
                if self.categories: # add loss for classc_criterion() prediction
                    c_real_tmp = torch.cat((c_data, c_data))
                    c_real.resize_as_(c_real_tmp).copy_(c_real_tmp)
                    error_D_real += c_criterion(class_probs_real, c_real_v)

                error_D_real.backward()

                if self.categories: #for conditional input
                    (verdict_fake, class_probs_fake) = self.D(*discriminator_input_fake)
                else :
                    verdict_fake = self.D(*discriminator_input_fake)

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
                    #z.uniform(-1,1)
                    z_temp = torch.randn((this_batch_size, self.z_len)).view(-1, self.z_len)
                    z.resize_as_(z_temp).copy_(z_temp)

                    generator_input = (z_v,)
                    if self.categories: #for conditional input
                        c_fake_tmp = self.fake_conditional_tensors(this_batch_size)
                        c_fake.copy_(torch.cat((c_fake_tmp,c_fake_tmp), 0))
                        c_fake_one_hot.resize_(this_batch_size, self.categories)
                        c_fake_one_hot.copy_(to_one_hot(self.categories, c_fake_tmp))
                        generator_input += (c_fake_one_hot_v,)

                    # Get input for D
                    x1_fake_v, x2_fake_v = self.G(*generator_input)
                    discriminator_input_fake = x1_fake_v, x2_fake_v #continue the backward pass through generator

                    # Update G   
                    if self.categories: #for conditional input
                        verdict_fake, class_probs_fake = self.D(*discriminator_input_fake)
                    else :
                        verdict_fake = self.D(*discriminator_input_fake)
                    
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
categories = int(ar.next_arg())
dataset = ar.next_arg()
if dataset == "MNIST":
    imgch = 1
else :
    imgch = 3

cogan = CoGAN(categories, imgch=imgch, g_d=2, d_d=2)
cogan.train(dataname=dataset, savename=ar.arg_string)
