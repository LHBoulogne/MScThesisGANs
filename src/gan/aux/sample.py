import numpy as np
import torch
from torch.autograd import Variable

from gan.aux.aux import to_one_hot

import utils

#Samples from z_distribution and puts result in z
def sample_z(z_distribution, z):
    if z_distribution == 'normal':
        z.normal_(mean=0, std=1)
    elif z_distribution == 'uniform' :
        z.uniform_(-1, 1)
    else :
        raise RuntimeError('z_distribution argument has unknown value: ' + z_distribution)

def sample_c(config, dataset):
    if config.dataname == "MNIST":
        return sample_multi_c(config)
    if config.dataname == "CelebA":
        return dataset.get_random_labelbatch(config.mini_batch_size)

def sample_multi_c(config):
    c1 = sample_multi_c_helper(config.mini_batch_size, config.labels1)
    if config.coupled: 
        c2 = sample_multi_c_helper(config.mini_batch_size, config.labels2)
        return (c1, c2)
    return c1

def sample_multi_c_helper(batch_size, labels) :
    idcs = np.random.randint(len(labels), size=(batch_size,))
    rands = np.array([labels[i] for i in idcs])
    rands = torch.from_numpy(rands)
    return rands

def sample_generator_input(config, this_batch_size, z, c_fake1=None, c_fake2=None):
    sample_z(config.z_distribution, z.data)
    g_inp = (z,)
    if config.auxclas: #for conditional input
        if config.dataname == "MNIST" or config.labeltype == 'onehot':
            c_fake_one_hot1 = Variable(utils.cuda(to_one_hot(config.categories, c_fake1.data)))
            g_inp += (c_fake_one_hot1,) 
            
            if config.coupled:
                c_fake_one_hot2 = Variable(utils.cuda(to_one_hot(config.categories, c_fake2.data)))
                g_inp += (c_fake_one_hot2,)
        else :
            c_fake1_inp = Variable(utils.cuda(c_fake1.data.float()))
            g_inp += (c_fake1_inp,)
            if config.coupled:
                c_fake2_inp = Variable(utils.cuda(c_fake2.data.float()))
                g_inp += (c_fake2_inp,)
    return g_inp