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

        #init Variables needed for training
        self.x1_real = torch.FloatTensor(config.mini_batch_size, config.imgch, config.imsize, config.imsize)
        self.x1_real_v = Variable(self.x1_real)
        self.x2_real = torch.FloatTensor(config.mini_batch_size, config.imgch, config.imsize, config.imsize)
        self.x2_real_v = Variable(self.x2_real)
        
        self.y_real = torch.FloatTensor(config.mini_batch_size)
        self.y_real_v = Variable(self.y_real)
        self.y_fake = torch.FloatTensor(config.mini_batch_size)
        self.y_fake_v = Variable(self.y_fake)

        self.z = torch.FloatTensor(config.mini_batch_size, config.z_len)
        self.z_v = Variable(self.z)

        # LongTensor with index for crossentropyloss function
        self.c_real1 = torch.LongTensor(config.mini_batch_size)
        self.c_real1_v = Variable(self.c_real1)
        self.c_fake1 = torch.LongTensor(config.mini_batch_size)
        self.c_fake1_v = Variable(self.c_fake1)

        self.c_real2 = torch.LongTensor(config.mini_batch_size)
        self.c_real2_v = Variable(self.c_real2)
        self.c_fake2 = torch.LongTensor(config.mini_batch_size)
        self.c_fake2_v = Variable(self.c_fake2)

        #onehot vectors for generator input
        self.c_fake_one_hot1 = torch.FloatTensor(config.mini_batch_size, config.categories)
        self.c_fake_one_hot1_v = Variable(self.c_fake_one_hot1)

        self.c_fake_one_hot2 = torch.FloatTensor(config.mini_batch_size, config.categories)
        self.c_fake_one_hot2_v = Variable(self.c_fake_one_hot2)

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


    def resize_vars(self):
        #resize Variables accordingly
        self.y_real.resize_(self.this_batch_size)
        self.y_fake.resize_(self.this_batch_size)
        
        self.z.resize_(self.this_batch_size, self.config.z_len)

        self.c_fake1.resize_(self.this_batch_size)
        self.c_fake2.resize_(self.this_batch_size)
        self.c_fake_one_hot1.resize_(self.this_batch_size, self.config.categories)
        self.c_fake_one_hot2.resize_(self.this_batch_size, self.config.categories)
            

    #loads the 'real' data into its torch.autograd Variables
    def next_step(self, data): 
        #put data into Variables
        if self.config.coupled:
            x1_data, x2_data, c1_data, c2_data = utils.cuda(data) #read out data tuple
            if self.config.auxclas:
                self.c_real1_v = Variable(c1_data)
                self.c_real2_v = Variable(c2_data)
            self.x1_real_v = Variable(x1_data)
            self.x2_real_v = Variable(x2_data)
            
        else :
            x1_data, c1_data = utils.cuda(data) #read out data tuple
            if self.config.auxclas:
                #set c_real Variable to contain the class conditional vector as input
                self.c_real1_v = Variable(c1_data)
            self.x1_real_v = Variable(x1_data)

        print('\n\n\n\n\nafter copying:')
        print('x1_data:')
        print(x1_data)
        print('x1_real:')
        print(self.x1_real)
        print('x1_real_v:')
        print(self.x1_real_v)

        # resize Variables to correct capacity
        self.this_batch_size = x1_data.size(0)
        self.resize_vars()

        # fill real/fake label Variables
        self.y_real.fill_(self.prob_data_is_real)
        self.y_fake.fill_(0.0)

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

    def compute_error_single_GAN(self, d_out, y_v, c_v):
        if self.config.auxclas: #for conditional input
            (verdict, class_probs) = d_out
        else :
            verdict = d_out[0]
        source_error = self.s_criterion(verdict.view(-1), y_v)
        if self.config.auxclas: # add loss for class_criterion() prediction
            classification_error = self.c_criterion(class_probs, c_v) * self.config.c_error_weight
            return (source_error, classification_error)

        return (source_error,)

    # returns a tuple containing:
    #   1 The total error 
    #   2 A tuple that contains all separate error. This tuple contains:
    #     - Per gan:
    #       - A tuple with the error on the source and 
    #         for an auxclas gan the classification error
    def compute_error_GAN(self, d_out, y_v, c1_v, c2_v):
        if self.config.coupled:
            separate_a = self.compute_error_single_GAN(d_out[0], y_v, c1_v)
            separate_b = self.compute_error_single_GAN(d_out[1], y_v, c2_v)
            total = sum(separate_a) + sum(separate_b)
            return (total, (separate_a, separate_b))
        separate = self.compute_error_single_GAN(d_out[0], y_v, c1_v)
        total = sum(separate)
        return (total, (separate,))

    def grad_penalty(self, inp_real, inp_fake, D):
        inp_hat = ()
        for idx in range(len(inp_fake)):
            e = utils.cuda(torch.rand(self.this_batch_size, 1,1,1))

            x = inp_real[idx].data
            x_wave = inp_fake[idx].data

            x_hat = e*x + (1-e)*x_wave
            inp_hat += (utils.cuda(Variable(x_hat, requires_grad=True)),)

        out_hat = D(*inp_hat)

        gps = ()
        for idx in range(len(out_hat)):
            gradient = grad(out_hat[idx][0], inp_hat[idx],
                grad_outputs = utils.cuda(torch.ones(out_hat[idx][0].size())), 
                create_graph = True)[0]
            gradient = gradient.view(self.this_batch_size, -1)
            gp = ((gradient.norm(p=2, dim=1) - 1)**2).mean()
            gps += (gp,)
            
        return gps


    def compute_D_error_WGAN_gp(self, inp_real, inp_fake, out_real, out_fake, D):
        gps = self.grad_penalty(inp_real, inp_fake, D)
        separate_errors = ()
        error = 0
        for idx in range(len(out_fake)):
            err = out_fake[idx][0].mean() - out_real[idx][0].mean() + self.config.gp_coef*gps[idx]
            separate_errors += (err,)
            error += err

        return error, separate_errors

    def compute_G_error_WGAN_gp(self, d_out):
        separate_errors = ()
        error = 0
        for idx in range(len(d_out)):
            err = -d_out[idx][0].mean()
            separate_errors += (err,)
            error += err

        return error, separate_errors
    

    def update_discriminator(self, G, D):
        D.zero_grad()

        # forward pass
          #for fake data
        g_inp = sample_generator_input(self.config, self.this_batch_size, 
            self.z_v, 
            self.c_fake_one_hot1_v, self.c_fake_one_hot2_v, 
            self.c_fake1, self.c_fake2)
        
        g_out = G(*g_inp)
        d_inp_fake = self.detach(g_out) #makes sure that the backward pass will stop at generator output
        
        d_inp_real = (self.x1_real_v,) 
        if self.config.coupled:
            d_inp_real += (self.x2_real_v,)

        d_out_fake = D(*d_inp_fake)
        d_out_real = D(*d_inp_real)
        
        if self.config.algorithm == 'wgan_gp':
            error, separate_errors = self.compute_D_error_WGAN_gp(d_inp_real, d_inp_fake, d_out_real, d_out_fake, D)
            error.backward()
            
            self.d_opt.step()
            self.error_storage.store_errors('discriminator', separate_errors) #TODO

        elif self.config.algorithm == 'default':
            # perform backward pass and update
            error_real, separate_errors_real = self.compute_error_GAN(d_out_real, self.y_real_v, self.c_real1_v, self.c_real2_v)
            error_real.backward()

            error_fake, separate_errors_fake = self.compute_error_GAN(d_out_fake, self.y_fake_v, self.c_fake1_v, self.c_fake2_v)
            error_fake.backward()
                        
            self.d_opt.step()
            self.error_storage.store_errors('discriminator', separate_errors_fake, separate_errors_real)
        else :
            raise RuntimeError('Algorithm not implemented: ' + self.config.algorithm)


    def update_generator(self, G, D):
        # forward pass
        G.zero_grad()
        g_inp = sample_generator_input(self.config, self.this_batch_size, 
            self.z_v, 
            self.c_fake_one_hot1_v, self.c_fake_one_hot2_v, 
            self.c_fake1, self.c_fake2)
        g_out = G(*g_inp)
        d_out = D(*g_out)

        # perform backward pass and update
        if self.config.algorithm == 'wgan_gp':
            error, separate_errors = self.compute_G_error_WGAN_gp(d_out)
        elif self.config.algorithm == 'default':
            error, separate_errors = self.compute_error_GAN(d_out, self.y_real_v, self.c_fake1_v, self.c_fake2_v)
        else :
            raise RuntimeError('Algorithm not implemented: ' + self.config.algorithm)

        sum(error).backward()
        self.g_opt.step()
        self.error_storage.store_errors('generator', separate_errors)
        
