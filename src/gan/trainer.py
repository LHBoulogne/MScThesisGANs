import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

from gan.aux.aux import to_one_hot
from gan.aux.sample import sample_generator_input

class GANTrainer():
    def __init__(self, config, G, D):
        self.config = config
        self.init_error_dicts()

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


    def init_error_dicts(self):
        self.error_dicts = []
        self.error_dicts += [{}]
        if self.config.coupled:
            self.error_dicts += [{}]
        for error_dict in self.error_dicts:
            self.init_error_dict(error_dict)

    def init_error_dict(self, d):
        d['G'] = {}
        d['D'] = {}
        d['D']['real'] = {}
        d['D']['fake'] = {}

        d['G']['source'] = []
        d['D']['real']['source'] = []
        d['D']['fake']['source'] = []

        if self.config.auxclas:
            d['G']['classification'] = []
            d['D']['real']['classification'] = []
            d['D']['fake']['classification'] = []

    def save_error(self):
        filename = os.path.join(self.config.savefolder, 'error.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(self.error_dicts, f)

    def load_error(self):
        filename = os.path.join(self.config.loadfolder, 'error.pkl')
        with open(filename, 'rb') as f:
            self.self.error_dicts = pickle.load(f)

    def store_errors_model(self, error_list, errors):
        keys = ['source', 'classification']
        for it, error in enumerate(errors):
            key = keys[it]
            error_list[key] += [error.data.numpy()]

    def store_errors_GAN(self, model, error_fake, error_real, d):
        if model == 'generator':
            error_list_fake = d['G']
        if model == 'discriminator':
            error_list_fake = d['D']['fake']
            error_list_real = d['D']['real']

        self.store_errors_model(error_list_fake, error_fake)

        if not error_real is None:
            self.store_errors_model(error_list_real, error_real)


    def store_errors(self, model, error_fake, error_real=(None,None)):
        for it in range(len(error_fake)):
            self.store_errors_GAN(model, error_fake[it], error_real[it], self.error_dicts[it])


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
            x1_data, x2_data, c1_data, c2_data = data #read out data tuple
            if self.config.auxclas:
                self.c_real1.resize_as_(c1_data).copy_(c1_data)
                self.c_real2.resize_as_(c2_data).copy_(c2_data)
            self.x1_real.resize_as_(x1_data).copy_(x1_data)
            self.x2_real.resize_as_(x2_data).copy_(x2_data)
            
        else :
            x1_data, c1_data = data #read out data tuple
            if self.config.auxclas:
                #set c_real Variable to contain the class conditional vector as input
                self.c_real1.resize_as_(c1_data).copy_(c1_data)
            self.x1_real.resize_as_(x1_data).copy_(x1_data)

        # resize Variables to correct capacity
        self.this_batch_size = x1_data.size(0)
        self.resize_vars()

        # fill real/fake label Variables
        self.y_real.fill_(self.prob_data_is_real)
        self.y_fake.fill_(0.0)


    def detach(self, tup):
        output = ()
        for t in tup:
            output += (t.detach(),)
        return output

    def compute_error_single_GAN(self, d_out, y_v, c_v):
        if self.config.auxclas: #for conditional input
            (verdict, class_probs) = d_out
        else :
            verdict = d_out

        source_error = self.s_criterion(verdict.view(-1), y_v)
        if self.config.auxclas: # add loss for class_criterion() prediction
            classification_error = self.c_criterion(class_probs, c_v) * self.config.c_error_weight
            return (source_error, classification_error)

        return (source_error,)

    # returns a tuple containing:
    #   1 The total error 
    #   2 A tuple that contains all separate error. This tuple contains:
    #    * Per gan:
    #     *  A tuple with the error on the source and 
    #        for an auxclas gan the classification error
    def compute_error(self, d_out, y_v, c1_v, c2_v):
        if self.config.coupled:
            separate_a = self.compute_error_single_GAN(d_out[0], y_v, c1_v)
            separate_b = self.compute_error_single_GAN(d_out[1], y_v, c2_v)
            total = sum(separate_a) + sum(separate_b)
            return (total, (separate_a, separate_b))
        separate = self.compute_error_single_GAN(d_out, y_v, c1_v)
        total = sum(separate)
        return (total, (separate,))

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
        d_out_fake = D(*d_inp_fake)
          
          #for real data 
        d_inp_real = (self.x1_real_v,) 
        if self.config.coupled:
            d_inp_real += (self.x2_real_v,)
        d_out_real = D(*d_inp_real)
        

        # perform backward pass and update
        error_D_real, separate_errors_real = self.compute_error(d_out_real, self.y_real_v, self.c_real1_v, self.c_real2_v)
        error_D_real.backward()

        error_D_fake, separate_errors_fake = self.compute_error(d_out_fake, self.y_fake_v, self.c_fake1_v, self.c_fake2_v)
        error_D_fake.backward()
        
        self.d_opt.step()
        self.store_errors('discriminator', separate_errors_fake, separate_errors_real)


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
        error_G, separate_errors = self.compute_error(d_out, self.y_real_v, self.c_fake1_v, self.c_fake2_v)
        sum(error_G).backward()
        self.store_errors('generator', separate_errors)
        self.g_opt.step()
        