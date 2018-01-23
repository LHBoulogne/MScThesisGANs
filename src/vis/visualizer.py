import os

import numpy as np
import scipy.misc

import torch
from torch.autograd import Variable

from gan.aux.sample import sample_z
from gan.aux.aux import to_one_hot

class Visualizer() :
    def __init__(self, dataset, config):
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
            c = np.repeat(range(config.categories), self.vis_noise_len)
            c_tensor = torch.from_numpy(c)
            one_hot = to_one_hot(config.categories, c_tensor)
            
            z = z.repeat(config.categories,1)
            self.x_dim = config.categories
            

        #construct input
        self.generator_input = (Variable(z),)
        if config.auxclas:
            self.generator_input += (Variable(one_hot),)
            if config.coupled:
                self.generator_input += (Variable(one_hot),)
            

    def output_to_img(self, output):
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
        
        if self.config.dataname == "MNIST":
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
            self.save_training_img(fake[0].data.numpy(), epoch, batch, 0) 
            self.save_training_img(fake[1].data.numpy(), epoch, batch, 1) 
        self.save_training_img(fake[0].data.numpy(), epoch, batch) 

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
            self.save_test_img(fake[0].data.numpy(), 0) 
            self.save_test_img(fake[1].data.numpy(), 1) 
        else:
            self.save_test_img(fake[0].data.numpy()) 
