
# Code for WGAN-GP was partially inspired by: https://github.com/LynnHo/WGAN-GP-DRAGAN-Celeba-Pytorch, Copyright (c) 2017 Zhenliang He

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import grad

import numpy as np

from gan.aux.aux import to_one_hot
from gan.aux.sample import sample_generator_input
from gan.errorstorage import *
import utils
    
class GANTrainer():
    def __init__(self, config, G, D):
        self.config = config
        self.error_storage = ErrorStorage(config)

        self.prob_data_is_real = 1.0
        if config.labelsmoothing :
            self.prob_data_is_real = 0.9
        
        self.y_real = Variable(torch.FloatTensor(config.mini_batch_size).fill_(self.prob_data_is_real))
        self.y_fake = Variable(torch.FloatTensor(config.mini_batch_size).fill_(0.0))

        self.c_fakes = self.real_fakes = (None, None)
        self.z = Variable(torch.FloatTensor(config.mini_batch_size, config.z_len))
        
        #init misc
        self.s_criterion = nn.BCELoss()
        
        self.c_criterion = nn.CrossEntropyLoss() #Includes the softmax function        

        self.g_opt = optim.Adam(G.parameters(), lr=config.g_lr, betas=(config.g_b1, config.g_b2), weight_decay=config.g_weight_decay)
        self.d_opt = optim.Adam(D.parameters(), lr=config.d_lr, betas=(config.d_b1, config.d_b2), weight_decay=config.d_weight_decay)
        
        self.cuda()
        
    def cuda(self):
        variables = [attr for attr in self.__dict__.keys() if type(self.__dict__[attr]) == Variable]

        for v in variables:
            self.__dict__[v] = utils.cuda(self.__dict__[v])

#loads the 'real' data into its torch.autograd Variables
    def next_step(self, data, c_fake_data=None): 
        c_fake_data = utils.cuda(c_fake_data)
        #put data into Variables

        if self.config.coupled:
            x1_data, x2_data, c1_data, c2_data = utils.cuda(data) #read out data tuple
            if self.config.auxclas:
                self.c_reals = (Variable(c1_data), Variable(c2_data))
                self.c_fakes = (Variable(c_fake_data[0]), Variable(c_fake_data[1]))
            self.x1_real = Variable(x1_data)
            self.x2_real = Variable(x2_data)
            
        else :
            x1_data, c1_data = utils.cuda(data) #read out data tuple
            if self.config.auxclas:
                #set c_real Variable to contain the class conditional vector as input
                self.c_reals = (Variable(c1_data),)
                self.c_fakes = (Variable(c_fake_data),)
            self.x1_real = Variable(x1_data)

        self.this_batch_size = x1_data.size(0)
        if self.config.mini_batch_size != self.this_batch_size:
            print('batch size is off: ' + str(self.this_batch_size))

    def get_error_storage(self):
        return self.error_storage

    def detach(self, tup):
        output = ()
        for t in tup:
            output += (t.detach(),)
        return output

    def save_error(self):
        self.error_storage.save_error()

    def load_error(self):
        self.error_storage.load_error()

    def src_error(self, d_out, y):
        return self.s_criterion(d_out.view(-1), y)

    def class_error(self, d_out, c):
        error = 0
        if len(c.size()) == 1:
            c=c.unsqueeze(1)
        for it in range(len(d_out)):
            error += self.c_criterion(d_out[it].view(self.config.mini_batch_size,-1), c[:,it])
        return error * self.config.c_error_weight

    def grad_penalty(self, inp_hat, out_hat):
        gradient = grad(out_hat, inp_hat,
            grad_outputs = utils.cuda(torch.ones(out_hat.size())), 
            create_graph = True)[0]
        gradient = gradient.view(self.config.mini_batch_size, -1)
        gp = ((gradient.norm(p=2, dim=1) - 1)**2).mean()
        
        return gp

    def WGAN_D_src_error(self, inp_real, inp_fake, inp_hat, out_real, out_fake, out_hat):
        gp = self.grad_penalty(inp_hat, out_hat)
        w1 = out_fake.mean() - out_real.mean() 
        err = w1 + self.config.gp_coef*gp
        return err, w1

    def WGAN_G_src_error(self, d_out):
        return -d_out.mean()


    def grad_penalty_old(self, inp_real, inp_fake, D):
        inp_hat = ()
        for idx in range(len(inp_fake)):
            e = utils.cuda(torch.rand(self.config.mini_batch_size, 1,1,1))

            x = inp_real[idx].data
            x_wave = inp_fake[idx].data

            x_hat = e*x + (1-e)*x_wave
            #x_hat = e*x_wave + (1-e)*x

            inp_hat += (utils.cuda(Variable(x_hat, requires_grad=True)),)

        out_hat = D(*inp_hat)

        gps = ()
        for idx in range(len(out_hat)):
            gradient = grad(out_hat[idx][0], inp_hat[idx],
                grad_outputs = utils.cuda(torch.ones(out_hat[idx][0].size())), 
                create_graph = True)[0]
            gradient = gradient.view(self.config.mini_batch_size, -1)
            gp = ((gradient.norm(p=2, dim=1) - 1)**2).mean()
            gps += (gp,)
            
        return gps


    def compute_D_error_WGAN_gp_old(self, inp_real, inp_fake, out_real, out_fake, D):
        gps = self.grad_penalty_old(inp_real, inp_fake, D)
        separate_errors = ()
        error = 0
        for idx in range(len(out_fake)):
            err = out_fake[idx][0].mean() - out_real[idx][0].mean() + self.config.gp_coef*gps[idx]
            separate_errors += (err,)
            error += err

        return error, separate_errors

    def compute_G_error_WGAN_gp_old(self, d_out):
        separate_errors = ()
        error = 0
        for idx in range(len(d_out)):
            err = -d_out[idx][0].mean()
            separate_errors += (err,)
            error += err

        return error, separate_errors


    def update_discriminator(self, G, D):
        if self.config.mini_batch_size != self.this_batch_size:
            return False
        D.zero_grad()

        #get fake discriminator output
        g_inp = sample_generator_input(self.config, self.config.mini_batch_size, self.z, *self.c_fakes)
        g_out = G(*g_inp)
        d_inp_fake_list = self.detach(g_out) #makes sure that the backward pass will stop at generator output
        d_out_fake_list = D(*d_inp_fake_list)

        #get real discriminator output
        d_inp_real_list = (self.x1_real,) 
        if self.config.coupled:
            d_inp_real_list += (self.x2_real,)
        d_out_real_list = D(*d_inp_real_list)

        #if necessary, get discriminator output of linearly interpolated points beween fake and real input 
        if self.config.algorithm == 'wgan_gp' or self.config.c_algorithm == 'wgan_gp':
            d_inp_hat_list = ()
            for idx in range(len(d_inp_fake_list)):
                e = utils.cuda(torch.rand(self.config.mini_batch_size, 1,1,1))
                x = d_inp_real_list[idx].data
                x_wave = d_inp_fake_list[idx].data
                x_hat = e*x + (1-e)*x_wave
                d_inp_hat_list += (utils.cuda(Variable(x_hat, requires_grad=True)),)
            d_out_hat_list = D(*d_inp_hat_list)

        if self.config.algorithm == 'wgan_gp_old':
            src_error, _ = self.compute_D_error_WGAN_gp_old(d_inp_real_list, d_inp_fake_list, d_out_real_list, d_out_fake_list, D)
            src_error.backward()
            self.d_opt.step()
        else:
            for idx in range(len(d_out_real_list)):
                d_out_real = d_out_real_list[idx]
                d_inp_real = d_inp_real_list[idx]
                d_out_fake = d_out_fake_list[idx]
                d_inp_fake = d_inp_fake_list[idx]

                if self.config.algorithm == 'default':
                    src_error_real = self.src_error(d_out_real[0], self.y_real)
                    src_error_fake = self.src_error(d_out_fake[0], self.y_fake)
                    src_error = (src_error_real, src_error_fake)
                elif self.config.algorithm == 'wgan_gp':
                    error, w_dist = self.WGAN_D_src_error(d_inp_real, d_inp_fake, d_inp_hat_list[idx], d_out_real[0], d_out_fake[0], d_out_hat_list[idx][0])
                    src_error = (error, w_dist)
                
                error = sum(src_error)
                if self.config.auxclas:
                    if self.config.c_algorithm == 'default':
                        class_error_real = self.class_error(d_out_real[1], self.c_reals[idx])
                        class_error_fake = self.class_error(d_out_fake[1], self.c_fakes[idx])
                        class_error = (class_error_real, class_error_fake)
                    elif self.config.c_algorithm == 'wgan_gp':
                        raise RuntimeError("c_algorithm wgan_gp: Not implemented")
                    error += sum(class_error)
                    
                    self.error_storage.store_errors('D', idx, src_error, class_error)
                else:
                    self.error_storage.store_errors('D', idx, src_error)
                
                error.backward()
            self.d_opt.step()

        return True


    def update_generator(self, G, D):
        if self.config.mini_batch_size != self.this_batch_size:
            return False
        
        # forward pass
        G.zero_grad()
        g_inp = sample_generator_input(self.config, self.config.mini_batch_size, self.z, *self.c_fakes)
        g_out = G(*g_inp)
        d_out_list = D(*g_out)

        # perform backward pass and update
        if self.config.algorithm == 'wgan_gp_old':
            error, separate_errors = self.compute_G_error_WGAN_gp_old(d_out_list)
            sum(error).backward()
            self.g_opt.step()
        else :
            for idx in range(len(d_out_list)):
                d_out = d_out_list[idx]
                
                if self.config.algorithm == 'default':
                    src_error = self.src_error(d_out[0], self.y_real)
                elif self.config.algorithm == 'wgan_gp':
                    src_error = self.WGAN_G_src_error(d_out[0])
                
                error = src_error
                if self.config.auxclas:
                    if self.config.c_algorithm == 'default':
                        class_error = self.class_error(d_out[1], self.c_reals[idx])
                    elif self.config.c_algorithm == 'wgan_gp':
                        raise RuntimeError("c_algorithm wgan_gp: Not implemented")
                    error += class_error
                    
                    self.error_storage.store_errors('G', idx, src_error, class_error)
                else:
                    self.error_storage.store_errors('G', idx, src_error)
                error.backward()
            
            self.g_opt.step()
        
        return True
