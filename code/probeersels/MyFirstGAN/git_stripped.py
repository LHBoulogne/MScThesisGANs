import os,random
os.environ["KERAS_BACKEND"] = "theano"
#os.environ["THEANO_FLAGS"]  = "device=gpu%d,lib.cnmem=0"%(random.randint(0,3))
import numpy as np
import theano as th
import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers.recurrent import LSTM
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle
#import _pickle as cPickle
import random, sys, keras
from keras.models import Model
#from IPython import display
from keras.utils import np_utils
from tqdm import tqdm

z_len = 10

def sample_real_data(batch_size) :
    rands = np.random.rand(batch_size) * 2*np.pi
    (a,b) = (np.sin(rands), np.cos(rands))
    return np.vstack((a,b)).T

def sample_noise(batch_size, dim) :
    mean = np.zeros(dim)
    cov = np.identity(dim)
    rands = np.random.multivariate_normal(mean, cov, batch_size)
    return rands

img_rows, img_cols = 28, 28



# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = sample_real_data(10000)
X_test = sample_real_data(10000)


print(str(np.min(X_train)) + str(np.max(X_train)))

print('X_train shape:' + str(X_train.shape))
print(str(X_train.shape[0]) + 'train samples')
print(str(X_test.shape[0]) + 'test samples')

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

shp = X_train.shape[1:]
opt = Adam(lr=1e-4)
dopt = Adam(lr=1e-3)

# Build Generative model ...
g_input = Input(shape=[z_len])
H = Dense(50)(g_input)
H = Activation('relu')(H)
H = Dense(50)(H)
H = Activation('relu')(H)
g_V = Dense(2)(H)
generator = Model(g_input,g_V)
generator.compile(loss='binary_crossentropy', optimizer=opt)
generator.summary()


# Build Discriminative model ...
d_input = Input(shape=shp)
H = Dense(50)(d_input)
H = Activation('relu')(H)
H = Dense(50)(H)
H = Activation('relu')(H)
d_V = Dense(2,activation='softmax')(H)
discriminator = Model(d_input,d_V)
discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)
discriminator.summary()

# Freeze weights in the discriminator for stacked training
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
make_trainable(discriminator, False)

# Build stacked GAN model
gan_input = Input(shape=[z_len])
H = generator(gan_input)
gan_V = discriminator(H)
GAN = Model(gan_input, gan_V)
GAN.compile(loss='categorical_crossentropy', optimizer=opt)
GAN.summary()

def plot_res(losses, n_ex=1000):
    plt.ion()

    plt.gcf().clear()
    noise = sample_noise(n_ex,z_len)
    generated_images = generator.predict(noise)

    idx = np.random.randint(0,X_train.shape[0],n_ex)
    real_images = X_train[idx,:]
    plt.figure(1)

    plt.subplot(211)
    plt.plot(losses["d"], label='discriminitive loss')
    plt.plot(losses["g"], label='generative loss')
    plt.legend()

    plt.subplot(212)
    plt.scatter(real_images[:,0], real_images[:,1], color='r')
    plt.scatter(generated_images[:,0], generated_images[:,1], color='b')
    plt.tight_layout()

    plt.draw()
    plt.pause(0.00001)


ntrain = 10000
trainidx = random.sample(range(0,X_train.shape[0]), ntrain)
XT = X_train[trainidx,:]

# Pre-train the discriminator network ...
noise_gen =sample_noise(XT.shape[0],z_len)
print("Generating images...")
generated_images = generator.predict(noise_gen)
X = np.concatenate((XT, generated_images))
n = XT.shape[0]
y = np.zeros([2*n,2])
y[:n,1] = 1
y[n:,0] = 1

make_trainable(discriminator,True)
print("Training discriminator...")
#discriminator.fit(X,y, nb_epoch=1, batch_size=128)
y_hat = discriminator.predict(X)

# Measure accuracy of pre-trained discriminator network
y_hat_idx = np.argmax(y_hat,axis=1)
y_idx = np.argmax(y,axis=1)
diff = y_idx-y_hat_idx
n_tot = y.shape[0]
n_rig = (diff==0).sum()
acc = n_rig*100.0/n_tot
print("Accuracy: %0.02f pct (%d of %d) right"%(acc, n_rig, n_tot))

# set up loss storage vector
losses = {"d":[], "g":[]}

# Set up our main training loop
def train_for_n(nb_epoch=5000, plt_frq=25,BATCH_SIZE=32):

    for e in tqdm(range(nb_epoch)):  
        
        # Make generative images
        image_batch = X_train[np.random.randint(0,X_train.shape[0],size=BATCH_SIZE),:]    
        noise_gen = sample_noise(BATCH_SIZE,z_len)
        generated_images = generator.predict(noise_gen)
        
        # Train discriminator on generated images
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2*BATCH_SIZE,2])
        y[0:BATCH_SIZE,1] = 1
        y[BATCH_SIZE:,0] = 1
        
        #make_trainable(discriminator,True)
        d_loss  = discriminator.train_on_batch(X,y)
        losses["d"].append(d_loss)
    
        # train Generator-Discriminator stack on input noise to non-generated output class
        noise_tr = sample_noise(BATCH_SIZE,z_len)
        y2 = np.zeros([BATCH_SIZE,2])
        y2[:,1] = 1
        
        #make_trainable(discriminator,False)
        g_loss = GAN.train_on_batch(noise_tr, y2 )
        losses["g"].append(g_loss)
        
        # Updates plots
        if e%plt_frq==plt_frq-1:
            plot_res(losses=losses)
        
print("Starting real training...")
# Train for 6000 epochs at original learning rates
train_for_n(nb_epoch=6000, plt_frq=500,BATCH_SIZE=32)

# Train for 2000 epochs at reduced learning rates
opt.lr.set_value(1e-5)
dopt.lr.set_value(1e-4)
train_for_n(nb_epoch=2000, plt_frq=500,BATCH_SIZE=32)

# Train for 2000 epochs at reduced learning rates
opt.lr.set_value(1e-6)
dopt.lr.set_value(1e-5)
train_for_n(nb_epoch=2000, plt_frq=500,BATCH_SIZE=32)

# Plot some generated images from our GAN
plot_res(losses=losses)
plt.ioff()
plt.show()