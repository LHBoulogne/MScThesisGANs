from keras.layers.core import Dense, Activation
from keras.layers import Input, merge, Conv2DTranspose, ZeroPadding2D, BatchNormalization, PReLU, LeakyReLU, Reshape, Conv2D, MaxPooling2D, UpSampling2D, Flatten

def feature_extractor(hidden):
	m = 10
	hidden = Conv2D(2*m, 5, strides=2, kernel_initializer='glorot_uniform')(hidden)
	
	hidden = Conv2D(5*m, 5, strides=2, kernel_initializer='glorot_uniform')(hidden)
	
	hidden = Conv2D(50*m, 4, strides=1, kernel_initializer='glorot_uniform')(hidden)
	hidden = MaxPooling2D(2)(hidden)
	hidden = LeakyReLU()(hidden)
	return Flatten()(hidden)