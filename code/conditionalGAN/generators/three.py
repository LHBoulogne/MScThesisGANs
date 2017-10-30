from keras.models import Model, Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Input, merge, Conv2DTranspose, ZeroPadding2D, BatchNormalization, PReLU, LeakyReLU, Reshape, Conv2D, MaxPooling2D, UpSampling2D, Flatten


def layers(hidden):
    hidden = Reshape((1,1) + hidden._keras_shape[1:])(hidden)
    hidden = Conv2DTranspose(128, 4, strides=1, padding='valid', kernel_initializer='glorot_uniform')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('relu')(hidden)
    
    hidden = Reshape((7, 7, 128), input_shape=(128*7*7,))(hidden)
    hidden = UpSampling2D(size=(2, 2))(hidden)
    hidden = Conv2D(64, (5, 5), padding='same')(hidden)
    hidden = Activation('relu')(hidden)
    
    hidden = UpSampling2D(size=(2, 2))(hidden)
    hidden = Conv2D(1, (5, 5), padding='same')(hidden)
    
    return Activation('tanh')(hidden)