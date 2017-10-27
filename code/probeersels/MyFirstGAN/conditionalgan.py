import numpy as np 
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers.core import Dense, Activation
from keras.layers import Input, merge, Conv2DTranspose, ZeroPadding2D, BatchNormalization, PReLU, Reshape, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras import optimizers
#from termcolor import colored

################
##### Data #####
################
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

    x_train = x_train.reshape(x_train.shape + (1,))/255.0
    x_test = x_test.reshape(x_test.shape + (1,))/255.0

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

def sample_noise(batch_size, dim) :
    #mean = np.zeros(dim)
    #cov = np.identity(dim)
    #rands = np.random.multivariate_normal(mean, cov, batch_size)
    rands = np.random.uniform(-1, 1, (batch_size, 100))
    return rands


####################
##### Building #####
####################

# outputs (input for Model(), concatenation of regular and conditional variable input)
# Conditional variable input can be specified by c_inp
def conditional_layer(conditionalvars, inp, c_inp=None):
    if conditionalvars > 0:
        if c_inp == None:
            c_inp = Input(shape=(conditionalvars,))
        hidden = merge([inp, c_inp], mode= 'concat')
        inp = [inp, c_inp]
    else: 
        hidden = inp

    return (inp, hidden)

# MNIST architecture partially from https://github.com/mingyuliutw/CoGAN_PyTorch/blob/master/src/net_gan_mnist.py
# Due to limited padding options, the architecture has been changed.
def generator(z_len, type_of_data, gopt, conditionalvars=0) :
    def layers(hidden, type_of_data):
        if type_of_data == 'mnist_x':
            hidden = Reshape((1,1) + hidden._keras_shape[1:])(hidden)

            km = 16 #kernel multiple

            hidden = Conv2DTranspose(16*km, 4, strides=1, padding='valid', kernel_initializer='glorot_uniform')(hidden)
            hidden = BatchNormalization()(hidden)
            hidden = PReLU()(hidden)
            print(hidden._keras_shape[1:])

            hidden = Conv2DTranspose(4*km, 3, strides=2, padding='same', kernel_initializer='glorot_uniform')(hidden)
            hidden = BatchNormalization()(hidden)
            hidden = PReLU()(hidden)
            print(hidden._keras_shape[1:])

            hidden = Conv2DTranspose(2*km, 3, strides=2, padding='same', kernel_initializer='glorot_uniform')(hidden)
            hidden = BatchNormalization()(hidden)
            hidden = PReLU()(hidden)
            print(hidden._keras_shape[1:])

            hidden = Conv2DTranspose(2*km, 3, strides=2, padding='same', kernel_initializer='glorot_uniform')(hidden)
            hidden = BatchNormalization()(hidden)
            hidden = PReLU()(hidden)
            print(hidden._keras_shape[1:])

            hidden = Conv2DTranspose(1*km, 5, strides=1, padding='valid', kernel_initializer='glorot_uniform')(hidden)
            hidden = BatchNormalization()(hidden)
            hidden = PReLU()(hidden)
            print(hidden._keras_shape[1:])

            hidden = Conv2DTranspose(1, 6, strides=1)(hidden)
            return Activation('sigmoid')(hidden)


        elif type_of_data == 'mnist':
            hidden = Reshape((1,1) + hidden._keras_shape[1:])(hidden)

            m = 2
            hidden = Conv2DTranspose(128*m, 4, strides=1, padding='valid', kernel_initializer='glorot_uniform')(hidden)
            hidden = BatchNormalization()(hidden)
            hidden = PReLU()(hidden)

            hidden = UpSampling2D(2)(hidden)
            hidden = Conv2D(64*m, 3, strides=1, padding='same', kernel_initializer='glorot_uniform')(hidden)
            hidden = BatchNormalization()(hidden)
            hidden = PReLU()(hidden)

            hidden = UpSampling2D(2)(hidden)
            hidden = Conv2D(32*m, 3, strides=1, padding='valid', kernel_initializer='glorot_uniform')(hidden)
            hidden = BatchNormalization()(hidden)
            hidden = PReLU()(hidden)

            hidden = UpSampling2D(2)(hidden)
            hidden = Conv2D(16*m, 3, strides=1, padding='same', kernel_initializer='glorot_uniform')(hidden)
            hidden = BatchNormalization()(hidden)
            hidden = PReLU()(hidden)

            hidden = Conv2D(1, 6, strides=1, padding='same', kernel_initializer='glorot_uniform')(hidden)
            return Activation('sigmoid')(hidden)

        else: #toy data
            activation = 'relu'
            hidden = Dense(20, kernel_initializer='glorot_uniform')(hidden)
            hidden = Activation(activation)(hidden)
            hidden = Dense(20, kernel_initializer='glorot_uniform')(hidden)
            hidden = Activation(activation)(hidden)
            out_len = 2
            return Dense(out_len)(hidden)

    inp = Input(shape=(z_len,))
    (inp, hidden) = conditional_layer(conditionalvars, inp)
    
    hidden = layers(hidden, type_of_data)

    G = Model(inp, hidden)
    G.compile(loss='categorical_crossentropy', optimizer=gopt)
    G.summary()

    return G

def discriminator(inpshape, type_of_data, dopt, conditionalvars=0) :
    def feature_extractor(hidden, type_of_data):
        m = 2
        if type_of_data == 'mnist':
            hidden = Conv2D(2*m, 5, strides=1, kernel_initializer='glorot_uniform')(hidden)
            hidden = MaxPooling2D(2)(hidden)
            hidden = Conv2D(5*m, 5, strides=1, kernel_initializer='glorot_uniform')(hidden)
            hidden = MaxPooling2D(2)(hidden)
            hidden = Conv2D(50*m, 4, strides=1, kernel_initializer='glorot_uniform')(hidden)
            hidden = PReLU()(hidden)
            return Flatten()(hidden)

        else: #toy data
            activation = 'relu'
            hidden = Dense(20, kernel_initializer='glorot_uniform')(hidden)
            hidden = Activation(activation)(hidden)
            hidden = Dense(20, kernel_initializer='glorot_uniform')(hidden)
            return Activation(activation)(hidden)

    f_inp = Input(shape=inpshape)
    f_hidden = feature_extractor(f_inp, type_of_data)
    c_inp = Input(shape=(conditionalvars,))
    c_hidden = Dense(100, kernel_initializer='glorot_uniform')(c_inp)

    hidden = merge([f_hidden, c_hidden], mode= 'concat')

    hidden = Dense(2, kernel_initializer='glorot_uniform')(hidden)
    hidden = Activation('softmax')(hidden)

    D = Model([f_inp, c_inp], hidden)
    D.compile(loss='categorical_crossentropy', optimizer=dopt)
    D.summary()

    return D



def buildGAN(z_len, type_of_data, conditionalvars=0):
    dopt = optimizers.Adam(lr=0.0002)
    gopt = optimizers.Adam(lr=0.0002)

    G = generator(z_len, type_of_data, gopt, conditionalvars)
    D_inpshape = G.outputs[0]._keras_shape[1:]
    D = discriminator(D_inpshape, type_of_data, dopt, conditionalvars)
    
    #Freeze weights in discriminator
    for layer in D.layers:
        layer.trainable = False

    GAN_in = Input(shape=(z_len,))
    if conditionalvars > 0:
        GANc_in = Input(shape=(conditionalvars,))
        GAN_hidden = G([GAN_in, GANc_in])
        GAN_hidden = D([GAN_hidden, GANc_in])
        GAN = Model([GAN_in, GANc_in], GAN_hidden)
    else :
        GAN_hidden = G(GAN_in)
        GAN_hidden = D(GAN_hidden)
        GAN = Model(GAN_in, GAN_hidden)

    GAN.compile(loss='categorical_crossentropy', optimizer=gopt)
    GAN.summary()

    return (G, D, GAN)

####################
##### Training #####
####################

def trainGAN(x_train, real, noise, G, D, GAN, z_len, mini_batch_size, k, nr_batches, c_train=None, plot_step=10):
    if not c_train is None:
        categories = c_train.shape[1]
    else:
        categories = 0

    print("Starting training:")

    for it in range(nr_batches):
        if plot_step!=0 and (it%plot_step == 0 or it+1 == nr_batches):
            plot(real, noise, G, it, categories)

        print("\rBatch " + str(it+1) + "/" + str(nr_batches), end='\r')
        
        y_real = np.vstack((np.ones(mini_batch_size), np.zeros(mini_batch_size))).T
        y_fake = 1-y_real
        y = np.append(y_real, y_fake, axis=0)
        
        for it2 in range(k):
            # Get training data for D
            z = sample_noise(mini_batch_size, z_len)
            (x_real, idcs) = sample_data(mini_batch_size, x_train)
            
            if categories: #for conditional input
                c_fake = to_one_hot(np.random.randint(categories, size=(mini_batch_size,)))
                z = [z, c_fake]
            
            x_fake = G.predict(z)
        
            x = np.append(x_real, x_fake, axis=0)
            
            if categories: #for conditional input
                c_real = c_train[idcs]
                x = [x, np.append(c_real, c_fake,axis=0)]

            D.train_on_batch(x, y)

        # Update D
        z = sample_noise(mini_batch_size, z_len)
        if categories: #for conditional input
            c_fake = to_one_hot(np.random.randint(categories, size=(mini_batch_size,)))
            z = [z, c_fake] 
        GAN.train_on_batch(z, y_real)

    print("Batch " + str(it+1) + "/" + str(nr_batches))
    return (G, D, GAN)

####################
##### Plotting #####
####################

def plot(real, noise, G, iteration, categories=0, savefolder='train_images', save=True, plot=False) :
        if not plot and not save:
            pass


        if categories == 0:
            fake = G.predict(noise)
        else :
            fake = G.predict([noise, to_one_hot(np.arange(len(noise))%categories)])
        
        images = len(fake.shape) == 4

        if images:
            xdim = fake.shape[1] 
            ydim = fake.shape[2]

            x = y = int(np.sqrt(fake.shape[0]))
            image = np.zeros((x*xdim, y*ydim))
            for ity in range(x):
                for itx in range(y):
                    xstart = itx*xdim
                    ystart = ity*ydim
                    image[xstart:xstart+xdim,ystart:ystart+ydim] = fake[itx+x*ity,:,:,0]
        
        if save and images:
            np.save(savefolder + '/' + str(iteration), image)

        if plot:
            plt.figure(1)
            plt.gcf().clear()
            if images:
                plt.imshow(image, cmap='gray')
                plt.title('Iteration ' + str(iteration))
                
            else:
                plt.scatter(real[:,0], real[:,1], color='r')
                plt.scatter(fake[:,0], fake[:,1], color='b')
                
                axes = plt.gca()

                minx = np.min(real[:,0])-1
                maxx = np.max(real[:,0])+1
                miny = np.min(real[:,1])-1
                maxy = np.max(real[:,1])+1

                axes.set_xlim([minx, maxx])
                axes.set_ylim([miny, maxy])

            plt.draw()
            plt.pause(0.00001)

################
##### Main #####
################

def main() :
    ### Set params
    mini_batch_size = 128
    k=1
    nr_batches = 25000

    type_of_data = 'mnist' # 'mnist'  'lisajou'  'gaussians on circle'
    categories = 10 #only for gaussians
    data_size = (60000, 10000)

    z_len = 100

    ###
    if type_of_data == 'lisajou' : 
        categories = 0
    elif type_of_data == 'mnist' :
        categories = 10

    x_train, c_train, x_test, c_test = get_data(type_of_data, categories, data_size)

    if categories == 0:
        c_train = None

    (G, D, GAN) = buildGAN(z_len, type_of_data, categories)

    plot_size = 16
    (real, _) = sample_data(plot_size, x_train)
    noise = sample_noise(plot_size, z_len)
    plt.ion()

    (G, D, GAN) = trainGAN(x_train, real, noise, G, D, GAN, z_len, mini_batch_size, k, nr_batches, c_train=c_train, plot_step=10)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
