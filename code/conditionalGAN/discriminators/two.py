from keras.layers.core import Dense, Activation
from keras.layers import Input, merge, Conv2DTranspose, ZeroPadding2D, BatchNormalization, PReLU, LeakyReLU, Reshape, Conv2D, MaxPooling2D, UpSampling2D, Flatten

def feature_extractor(hidden):
    hidden = Conv2D(64, (5, 5), padding='same')(hidden)
    hidden = Activation('tanh')(hidden)
    
    hidden = MaxPooling2D(pool_size=(2, 2))(hidden)
    hidden = Conv2D(128, (5, 5))(hidden)
    hidden = Activation('tanh')(hidden)
    
    hidden = MaxPooling2D(pool_size=(2, 2))(hidden)
    hidden = Flatten()(hidden)

    return Dense(1024)(hidden)