################
##### Data #####
################
import numpy as np
from keras.datasets import mnist

def get_data(type_of_data, n=3, size=(10000, 10000)):
    if type_of_data == 'mnist':
        return MNIST_data(size)
    elif type_of_data == 'lisajou' or type_of_data == 'gaussians on circle' :
        (x_train, y_train) = toy_data(type_of_data, n, size[0])
        (x_test,  y_test)  = toy_data(type_of_data, n, size[1])
        return x_train, y_train, x_test, y_test
    return 


def MNIST_data(size) :
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if size[0] > x_train.shape[0]:
        print('\033[93mWARNING: requested ' + str(size[0]) + ' train samples for MNIST data, but only '+ str(x_train.shape[0]) +' available.')
    if size[1] > x_test.shape[0]:
        print('\033[93mWARNING: requested ' + str(size[1]) + ' test samples for MNIST data, but only '+ str(x_test.shape[0]) +' available.')

    y_train = to_one_hot(y_train[:size[0]])
    y_test  = to_one_hot(y_test[:size[1]])

    x_train = x_train.reshape(x_train.shape + (1,))/127.5 -1 
    x_test = x_test.reshape(x_test.shape + (1,))/127.5 - 1

    return x_train[:size[0]], y_train, x_test[:size[1]], y_test

def toy_data(toytype, n=5, size=10000) :
    data = np.array([]).reshape((0,2))
    labels = np.array([], dtype=np.int)
    
    if toytype == 'lisajou':
        rands = np.random.rand(size) * 16*np.pi
        (a,b) = (np.sin(2*rands), np.cos(1*rands))
        data = np.vstack((a,b)).T
        labels = None
    
    elif toytype == 'gaussians on circle':
        circle_fraction = 2 * np.pi/n
        std = circle_fraction/100
        cov = np.identity(2) * std
        for it in range(n):
            a = it*circle_fraction #how far along on the circle the distribution should be
            mean = (np.sin(a), np.cos(a))
            data_part = np.random.multivariate_normal(mean, cov, (int(size/n,)))
            data = np.append(data, data_part, axis=0)
            label_part = np.full(int(size/n), it)   
            labels = np.append(labels, label_part) 
    
    labels = to_one_hot(labels)
    return data, labels

def to_one_hot(int_array):
    maximum = max(int_array)
    width = maximum+1
    output = np.zeros((len(int_array), width))
    output[range(len(int_array)), int_array] = 1
    return output

def sample_data(batch_size, x_train) :
    rands = np.random.randint(len(x_train), size=(batch_size,))
    return (x_train[rands], rands)

def sample_noise(batch_size) :
    rands = np.random.uniform(-1, 1, (batch_size, self.z_len))
    return rands
