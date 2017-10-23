import numpy as np 
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers.core import Dense, Activation
from keras.layers import Input
from keras import optimizers





def toy_data(size) :
    rands = np.random.rand(size) * 16*np.pi
    (a,b) = (np.sin(2*rands), np.cos(1*rands))
    return np.vstack((a,b)).T

x_train = toy_data(10000)
x_shape = x_train.shape[1:]

def sample_real_data(batch_size) :
    rands = np.random.randint(len(x_train), size=(batch_size,))
    return x_train[rands]

def sample_noise(batch_size, dim) :
    mean = np.zeros(dim)
    cov = np.identity(dim)
    rands = np.random.multivariate_normal(mean, cov, batch_size)
    return rands

def buildGAN(z_len, out_len):
    gopt = optimizers.Adam(lr=1e-4, decay=1e-6)
    dopt = optimizers.Adam(lr=1e-3, decay=1e-6)

    G_in = Input(shape=(z_len,))
    G_hidden = Dense(20)(G_in)
    G_hidden = Activation('relu')(G_hidden)
    G_hidden = Dense(20)(G_hidden)
    G_hidden = Activation('relu')(G_hidden)
    G_hidden = Dense(out_len)(G_hidden)
    G = Model(G_in, G_hidden)
    G.compile(loss='categorical_crossentropy', optimizer=gopt)
    G.summary()

    D_in = Input(shape=(2,))
    D_hidden = Dense(20)(D_in)
    D_hidden = Activation('relu')(D_hidden)
    D_hidden = Dense(20)(D_hidden)
    D_hidden = Activation('relu')(D_hidden)
    D_hidden = Dense(2)(D_hidden)
    D_hidden = Activation('softmax')(D_hidden)

    D = Model(D_in, D_hidden)
    D.compile(loss='categorical_crossentropy', optimizer=dopt)
    D.summary()

    #Freeze weights in discriminator
    for layer in D.layers:
        layer.trainable = False

    GAN_in = Input(shape=(z_len,))
    GAN_hidden = G(GAN_in)
    GAN_hidden = D(GAN_hidden)
    GAN = Model(GAN_in, GAN_hidden)
    GAN.compile(loss='categorical_crossentropy', optimizer=gopt)
    GAN.summary()

    return (G, D, GAN)

def plot(real, noise, G) :
        plt.figure(1)
        fake = G.predict(noise)
        plt.gcf().clear()
        plt.scatter(real[:,0], real[:,1], color='r')
        plt.scatter(fake[:,0], fake[:,1], color='b')
        
        axes = plt.gca()
        axes.set_xlim([-2,2])
        axes.set_ylim([-2,2])

        plt.draw()
        plt.pause(0.00001)

def trainGAN(real, noise, G, D, GAN, mini_batch_size, z_len, k, nr_batches, plot_step=300):
    print("Starting training:")
    for it in range(nr_batches):
        if it%plot_step == 0 :
            plot(real, noise, G)

        print("\rBatch " + str(it+1) + "/" + str(nr_batches), end='\r')
        for it2 in range(k):
            # Get training data for D
            z = sample_noise(mini_batch_size, z_len)
            x_real = sample_real_data(mini_batch_size)
            x_fake = G.predict(z)
            x = np.append(x_real, x_fake, axis=0)
            
            y_real = np.vstack((np.ones(mini_batch_size), np.zeros(mini_batch_size))).T
            y_fake = 1-y_real
            y = np.append(y_real, y_fake, axis=0)
            # Update D
            D.train_on_batch(x, y)

        # Update D      
        z = sample_noise(mini_batch_size, z_len)
        GAN.train_on_batch(z, y_real)

    print("Batch " + str(it+1) + "/" + str(nr_batches))
    return (G, D, GAN)

def main() :
    ### Set params
    mini_batch_size = 128
    z_len = 2
    k=1
    nr_batches = 10000
    ###

    (G, D, GAN) = buildGAN(z_len, 2)

    plot_size = 300
    real = sample_real_data(plot_size)
    noise = sample_noise(plot_size, z_len)

    plt.ion()
    (G, D, GAN) = trainGAN(real, noise, G, D, GAN, mini_batch_size, z_len, k, nr_batches)

    plot(real, noise, G)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()