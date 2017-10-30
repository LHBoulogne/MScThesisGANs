from keras.models import Model, Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Input, merge, Conv2DTranspose, ZeroPadding2D, BatchNormalization, PReLU, LeakyReLU, Reshape, Conv2D, MaxPooling2D, UpSampling2D, Flatten


def layers(hidden):
    hidden = Dense(128*7*7)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('relu')(hidden)
    
    hidden = Reshape((7, 7, 128), input_shape=(128*7*7,))(hidden)
    hidden = UpSampling2D(size=(2, 2))(hidden)
    hidden = Conv2D(64, (5, 5), padding='same')(hidden)
    hidden = Activation('relu')(hidden)
    
    hidden = UpSampling2D(size=(2, 2))(hidden)
    hidden = Conv2D(1, (5, 5), padding='same')(hidden)
    
    return Activation('tanh')(hidden)