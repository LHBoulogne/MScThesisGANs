from keras.models import Model, Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Input, merge, Conv2DTranspose, ZeroPadding2D, BatchNormalization, PReLU, LeakyReLU, Reshape, Conv2D, MaxPooling2D, UpSampling2D, Flatten


def layers(hidden):
    m = 4
    hidden = Reshape((1,1) + hidden._keras_shape[1:])(hidden)
    hidden = Conv2DTranspose(128*m, 4, strides=1, padding='valid', kernel_initializer='glorot_uniform')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('relu')(hidden)

    hidden = UpSampling2D(2)(hidden)
    hidden = Conv2D(64*m, 3, strides=1, padding='same', kernel_initializer='glorot_uniform')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('relu')(hidden)

    hidden = UpSampling2D(2)(hidden)
    hidden = Conv2D(32*m, 3, strides=1, padding='valid', kernel_initializer='glorot_uniform')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('relu')(hidden)

    hidden = UpSampling2D(2)(hidden)
    hidden = Conv2D(16*m, 3, strides=1, padding='same', kernel_initializer='glorot_uniform')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('relu')(hidden)

    hidden = Conv2D(1, 5, strides=1, padding='same', kernel_initializer='glorot_uniform')(hidden)
    hidden = BatchNormalization()(hidden)
    return Activation('tanh')(hidden)