import os, time, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets as dset
from torch.autograd import Variable

import numpy as np

sys.path.append("./../../cogan/")
from mnistedge import *

def rescale(t):
    return t.div_(127.5).add_(-1)

class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, inp):
        return inp.view(self.shape)


# Generator Model
class Generator(nn.Module):
    def __init__(self, d, imgch=3, z_len=100, categories=0):
        super(Generator, self).__init__()
        self.dconv0 = nn.ConvTranspose2d(z_len, d*8, kernel_size=4, stride=1)
        self.bn0 = nn.BatchNorm2d(d*8, affine=False)
        self.prelu0 = nn.PReLU()
        self.dconv1 = nn.ConvTranspose2d(d*8, d*4, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(d*4, affine=False)
        self.prelu1 = nn.PReLU()
        self.dconv2 = nn.ConvTranspose2d(d*4, d*2, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(d*2, affine=False)
        self.prelu2 = nn.PReLU()
        self.dconv3 = nn.ConvTranspose2d(d*2, d, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(d, affine=False)
        self.prelu3 = nn.PReLU()
        self.dconv4_a = nn.ConvTranspose2d(d, imgch, kernel_size=6, stride=1, padding=1)
        self.dconv4_b = nn.ConvTranspose2d(d, imgch, kernel_size=6, stride=1, padding=1)
        self.sig4_a = nn.Tanh()
        self.sig4_b = nn.Tanh()

    def forward(self, z, cond=None):
        z = z.view(z.size(0), z.size(1), 1, 1)
        h0 = self.prelu0(self.bn0(self.dconv0(z)))
        h1 = self.prelu1(self.bn1(self.dconv1(h0)))
        h2 = self.prelu2(self.bn2(self.dconv2(h1)))
        h3 = self.prelu3(self.bn3(self.dconv3(h2)))
        out_a = self.sig4_a(self.dconv4_a(h3))
        out_b = self.sig4_b(self.dconv4_b(h3))
        return out_a, out_b

    def weight_init(self,  method='xavier', mean=0.0, std=0.02):
        for m in self._modules:
            if method == 'normal':
                normal_init(self._modules[m], mean, std)
            elif method == 'xavier':
                xavier_weights_init(self.modules[m])

class Discriminator(nn.Module):
    def __init__(self, d, imgch, categories=0):
        super(Discriminator, self).__init__()
        self.conv0_a = nn.Conv2d(imgch, 2*d, kernel_size=5, stride=1, padding=0)
        self.conv0_b = nn.Conv2d(imgch, 2*d, kernel_size=5, stride=1, padding=0)
        self.pool0 = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(2*d, 5*d, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(5*d, 50*d, kernel_size=4, stride=1, padding=0)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(50*d, 1, kernel_size=1, stride=1, padding=0)
        self.sig3 = nn.Sigmoid()

    def forward(self, x_a, x_b):
        h0_a = self.pool0(self.conv0_a(x_a))
        h0_b = self.pool0(self.conv0_b(x_b))
        h0 = torch.cat((h0_a, h0_b), 0)
        h1 = self.pool1(self.conv1(h0))
        h2 = self.prelu2(self.conv2(h1))
        h3 = self.sig3(self.conv3(h2))
        return h3

    def weight_init(self, method='xavier', mean=0.0, std=0.02):
        for m in self._modules:
            if method == 'normal':
                normal_init(self._modules[m], mean, std)
            elif method == 'xavier':
                xavier_weights_init(self.modules[m])


def xavier_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0.1)
    if isinstance(m, nn.Sequential):
        for mod in m._modules:
            xavier_weights_init(m._modules[mod])

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
        self.D = Discriminator(d_d, imgch)
        self.G = Generator(g_d, imgch)
        
        self.G.weight_init(method='xavier')
        self.D.weight_init(method='xavier')

    def save_progress(self, noise, dim, epoch, batch, error_D, error_G, savefolder) :
        writ_style = "w" if epoch == 0 and batch == 0 else "a"

        with open(os.path.join(savefolder, "error_D.txt"), writ_style) as f:
            f.write(str(error_D.data.numpy()[0]) + " ")

        with open(os.path.join(savefolder, "error_G.txt"), writ_style) as f:
            f.write(str(error_G.data.numpy()[0]) + " ")


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

    def train(self,  mini_batch_size=128, k=1, nr_epochs=20, vis_step=10, vis_dim=6, savename=""):
        savefolder="savedata_" + savename
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)

        imgs_size = 28
        #Get dataset ready
        dataset = MNIST_edge(transform=transforms.Compose([transforms.Scale((imgs_size,imgs_size)),
                                                transforms.ToTensor(),
                                                transforms.Lambda(rescale)]))

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

        x1_real = torch.FloatTensor(mini_batch_size, self.imgch, imgs_size, imgs_size)
        x1_real_v = Variable(x1_real)
        x2_real = torch.FloatTensor(mini_batch_size, self.imgch, imgs_size, imgs_size)
        x2_real_v = Variable(x2_real)
        
        y_real = torch.ones(mini_batch_size*2).mul_(0.9)
        y_real_v = Variable(y_real)
        y_fake = torch.zeros(mini_batch_size*2)
        y_fake_v = Variable(y_fake)

        #z.resize_(this_batch_size, self.z_len).uniform_(-1,1)
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
        steps_without_G_update = k-1
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
                    c_fake.resize_(this_batch_size)
                    c_fake.copy_(self.fake_conditional_tensors(this_batch_size))
                    c_fake_one_hot.resize_(this_batch_size, self.categories)
                    c_fake_one_hot.copy_(to_one_hot(self.categories, c_fake))
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
                    c_real.resize_as_(c_data).copy_(c_data)
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
                        c_fake.copy_(self.fake_conditional_tensors(this_batch_size))
                        c_fake_one_hot.copy_(to_one_hot(self.categories, c_fake))
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
                    self.save_progress(visualization_noise, vis_dim, epoch, batch, error_D, error_G, savefolder=savefolder)
                    self.save(savefolder)

            print("Epoch: "+str(epoch+1)+ "/" + str(nr_epochs)+", Batch " + str(batch+1))

        self.save_progress(visualization_noise, vis_dim, epoch, batch, error_D, error_G, savefolder=savefolder)
        self.save(savefolder)

cogan = CoGAN(0, imgch=1, g_d=128, d_d=10)
cogan.train()
