import numpy as np
import torch

from gan.aux.aux import to_one_hot

#Samples from z_distribution and puts result in z
def sample_z(z_distribution, z):
    if z_distribution == 'normal':
        z.normal_(mean=0, std=1)
    elif z_distribution == 'uniform' :
        z.uniform_(-1, 1)
    else :
        raise RuntimeError('z_distribution argument has unknown value: ' + z_distribution)


def sample_c(batch_size, labels) :
    idcs = np.random.randint(len(labels), size=(batch_size,))
    rands = np.array([labels[i] for i in idcs])
    rands = torch.from_numpy(rands)
    return rands


# also puts class vector for error computation in c_fake1 and c_fake2 if specified
def sample_generator_input(config, this_batch_size, z_v, c_fake_one_hot1_v, c_fake_one_hot2_v, c_fake1=None, c_fake2=None):
    sample_z(config.z_distribution, z_v.data)
    g_inp = (z_v,)
    if config.auxclas: #for conditional input
        c_fake_tmp1 = sample_c(this_batch_size, config.labels1)
        if not c_fake1 is None:
            c_fake1.copy_(c_fake_tmp1)
        c_fake_one_hot1_v.data.copy_(to_one_hot(config.categories, c_fake_tmp1))
        g_inp += (c_fake_one_hot1_v,) 
        
        if config.coupled:
            c_fake_tmp2 = sample_c(this_batch_size, config.labels2)
            if not c_fake2 is None:
                c_fake2.copy_(c_fake_tmp2)
            c_fake_one_hot2_v.data.copy_(to_one_hot(config.categories, c_fake_tmp2))
            g_inp += (c_fake_one_hot2_v,)

    return g_inp