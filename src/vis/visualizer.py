import os

import numpy as np
import scipy.misc

import torch
from torch.autograd import Variable

from gan.auxiliary.sample import sample_z
from gan.auxiliary.auxiliary import to_one_hot

import utils

class Visualizer() :
    def __init__(self, config):
        self.config = config
        
        self.vis_noise_len = config.vis_dim*config.vis_dim
        if config.auxclas:
            self.vis_noise_len = config.vis_dim
        self.x_dim = config.vis_dim
        self.y_dim = config.vis_dim

        #init z
        z = torch.FloatTensor(self.vis_noise_len, config.z_len)
        sample_z(config.z_distribution, z)
        
        #init c if necessary
        if config.auxclas:
            if config.dataname == "CelebA" and config.labeltype == "bool": #Todo: make up to date for multiple categories
                c_len = 2
                c = []
                for n in range(c_len):
                    binary = bin(n)[2:].zfill(1) #unnecessarry

                    c += [[int(x) for x in binary]]
                c = np.array(c, dtype=np.float32)
                c = np.repeat(c, self.vis_noise_len, axis=0)
                c_g_input = torch.from_numpy(c)
            else:
                c_len = 1
                for cat in config.categories:
                    c_len *= cat

                category = config.categories[0]
                c = np.repeat(range(category), self.vis_noise_len)
                c = torch.from_numpy(c).float().unsqueeze(1)
                for category in config.categories[1:]:
                    new = np.repeat(range(category), len(c))
                    new = torch.from_numpy(new).float().unsqueeze(1)
                    c = torch.cat([c]*category, 0)
                    c = torch.cat([c, new], 1)
                c_g_input = to_one_hot(config.categories, c)
            z = z.repeat(c_len,1)
            #old self.x_dim = c_len
            self.y_dim = c_len
            

        #construct input
        self.generator_input = (utils.cuda(Variable(z)),)
        if config.auxclas:
            self.generator_input += (utils.cuda(Variable(c_g_input)),)
            if config.coupled:
                self.generator_input += (utils.cuda(Variable(c_g_input)),)

    def output_to_img(self, output):
        x = output.shape[2]
        y = output.shape[3]
        image = np.empty((self.config.imgch, self.x_dim*x, self.y_dim*y))
        for ity in range(self.y_dim):
            for itx in range(self.x_dim):
                xstart = itx*x
                ystart = ity*y
                image[:, xstart:xstart+x  ,ystart:ystart+y] = output[itx+self.x_dim*ity]
        return image

    def output_to_img_old(self, output):
        x = output.shape[2]
        y = output.shape[3]
        image = np.empty((self.config.imgch, self.x_dim*x, self.y_dim*y))
        for ity in range(self.y_dim):
            for itx in range(self.x_dim):
                xstart = itx*x
                ystart = ity*y
                image[:, xstart:xstart+x  ,ystart:ystart+y] = output[ity+self.y_dim*itx]
        return image

    def save_img(self, save_name, output):
        img = self.output_to_img(output)
        
        if self.config.imgch == 1:
            scipy.misc.toimage(img[0], cmin=-1, cmax=1).save(save_name)
        else:
            scipy.misc.toimage(img, cmin=-1, cmax=1, channel_axis=0).save(save_name)
        


    def save_training_img(self, output, epoch, batch, nr=None):
        save_name = str(epoch) + "_" + str(batch)
        if not nr is None:
            save_name += "_" + str(nr)
        save_name += ".png"
        path = os.path.join(self.config.savefolder, 'train_imgs')
        save_name = os.path.join(path, save_name)
        if not os.path.exists(path):
            os.mkdir(path)

        self.save_img(save_name, output)
        
    def save_training_imgs(self, epoch, batch, G):
        fake = G(*self.generator_input)
        if self.config.coupled:
            self.save_training_img(fake[0].data.cpu().numpy(), epoch, batch, 0) 
            self.save_training_img(fake[1].data.cpu().numpy(), epoch, batch, 1) 
        else:
            self.save_training_img(fake[0].data.cpu().numpy(), epoch, batch) 

    def save_test_img(self, output, nr=None):
        save_name = 'test'
        if not nr is None:
            save_name += "_" + str(nr)
        save_name += ".png"
        path = os.path.join(self.config.savefolder, 'test_imgs')
        save_name = os.path.join(path, save_name)
        if not os.path.exists(path):
            os.mkdir(path)
        self.save_img(save_name, output)

    def save_test_imgs(self, G):
        fake = G(*self.generator_input)
        if self.config.coupled:
            self.save_test_img(fake[0].data.cpu().numpy(), 0) 
            self.save_test_img(fake[1].data.cpu().numpy(), 1) 
        else:
            self.save_test_img(fake[0].data.cpu().numpy()) 
