import os, time, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets as dset
from torch.autograd import Variable

import numpy as np

def rescale(t):
    return t.div_(127.5).add_(-1)

class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, inp):
        return inp.view(self.shape)

def compute_same_padding(k) :
    p_oneside = (k-1)//2
    correction = 1 if k%2==0 else 0
    p_lt = p_oneside
    p_rb = p_oneside + correction
    return (p_lt, p_rb, p_lt, p_rb)


class Generator(nn.Module):
    def __init__(self, d, imgch=3, z_len=100, categories=0, k=4):
        super(Generator, self).__init__()
        self.is_conditional = categories > 0

        if self.is_conditional:
            noise_channels = 4
            cond_channels = 4
            self.encode_c = nn.Sequential(
                nn.Linear(categories, (d*cond_channels)*4*4),
                nn.BatchNorm2d((d*cond_channels)*4*4),
                nn.ReLU(),
                Reshape(-1, d*cond_channels, 4, 4)
                )
        else :
            noise_channels = 8

        self.encode_z = nn.Sequential(
            nn.Linear(z_len, (d*noise_channels)*4*4),
            nn.BatchNorm2d((d*noise_channels)*4*4),
            nn.ReLU(),
            Reshape(-1, d*noise_channels, 4, 4)
            )
        
        # Same padding for conv layers
        p = compute_same_padding(k)
        self.main = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ReplicationPad2d(p),
            nn.Conv2d(d*8, d*4, k),
            nn.BatchNorm2d(d*4),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.ReplicationPad2d(p),
            nn.Conv2d(d*4, d*2, k),
            nn.BatchNorm2d(d*2),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.ReplicationPad2d(p),
            nn.Conv2d(d*2, d, k),
            nn.BatchNorm2d(d),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.ReplicationPad2d(p),
            nn.Conv2d(d, imgch, k),
            nn.Tanh()
        )

    def forward(self, noise, cond=None):
        print(noise.data.numpy().shape)
        x = self.encode_z(noise)
        
        if self.is_conditional:
            c = self.encode_c(cond)
            x = torch.cat((x, c), 1)
        return self.main(x)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class Discriminator(nn.Module):
    def __init__(self, d, imgch, categories=0):
        super(Discriminator, self).__init__() 
        self.is_conditional = categories > 0

        self.main = nn.Sequential(
            nn.Conv2d(imgch, d, 4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(0.2),

            nn.Conv2d(d,  d*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(d*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(d*2, d*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(d*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(d*4, d*8, 4, stride=2, padding=1),
            nn.BatchNorm2d(d*8),
            nn.LeakyReLU(0.2),

            Reshape(-1, (d*8)*4*4)
        )

        self.predict_src = nn.Sequential(
            nn.Linear((d*8)*4*4, 1),
            nn.Sigmoid()
        )

        if self.is_conditional:
            self.predict_class = nn.Linear((d*8)*4*4, categories)


    def forward(self, inp):
        x = self.main(inp)
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

class ACGAN():
        
    def __init__(self, categories=0, z_len=100, g_d=128, d_d=128, imgch=3, loadfolder=None):
        self.categories = categories
        self.z_len = z_len
        self.imgch = imgch
        self.D = Discriminator(d_d, imgch)
        self.G = Generator(g_d, imgch)
        
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

        fake = self.G(*generator_input).data.numpy()
        
        fake = np.moveaxis(fake, 1, 3)
        fake = (fake+1)/2
        x = fake.shape[1]
        y = fake.shape[2]
        image = np.empty((dim*x, dim*y, self.imgch))

        for ity in range(dim):
            for itx in range(dim):
                xstart = itx*x
                ystart = ity*y
                image[xstart:xstart+x,ystart:ystart+y] = fake[itx+dim*ity]
        np.save(savefolder + '/' + str(epoch) + '_' + str(batch).zfill(7), image)

    def save(self, savefolder) :
        torch.save(self.G.state_dict(), os.path.join(savefolder, 'generator.h5'))
        torch.save(self.D.state_dict(), os.path.join(savefolder, 'discriminator.h5'))
        
    def fake_conditional_tensors(self, batch_size) :
        rands = np.random.randint(self.categories, size=(batch_size,))
        rands = torch.from_numpy(rands)
        return rands

    def train(self, dataname="MNIST", mini_batch_size=128, k=1, nr_epochs=20, vis_step=10, vis_dim=10, savename=""):
        savefolder="savedata_" + savename
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)

        #Get dataset ready
        if dataname == "MNIST":
            dataset = dset.MNIST(root='../../../data/mnist', download=True, 
                  transform=transforms.Compose([transforms.Scale((64,64)),
                                                transforms.ToTensor(),
                                                transforms.Lambda(rescale)]))
        elif dataname == "CelebA":
            dataset = dset.ImageFolder(root='../../../data/celeba/', 
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

        x_real = torch.FloatTensor(mini_batch_size, self.imgch, 64, 64)
        x_real_v = Variable(x_real)
        
        y_real = torch.ones(mini_batch_size).mul_(0.9)
        y_real_v = Variable(y_real)
        y_fake = torch.zeros(mini_batch_size)
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
                    #z.uniform(-1,1)
                    z_temp = torch.randn((this_batch_size, self.z_len)).view(-1, self.z_len)
                    z.resize_as_(z_temp).copy_(z_temp)

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

acgan = ACGAN(0, imgch=3, g_d=64, d_d=64)
acgan.train(dataname="CelebA")
