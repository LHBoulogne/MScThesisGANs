import numpy as np 

from keras.models import Model
from keras.layers.core import Dense, Activation
from keras.layers import Input, merge, Conv2DTranspose, ZeroPadding2D, BatchNormalization, PReLU, LeakyReLU, Reshape, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras import optimizers


class GAN() : 
    #opts: (gen opt, disc opt)
    def __init__(self, data_shape, categories=0, z_len=100, generator=None, discriminator=None):
        self.z_len = z_len
        self.data_shape = data_shape
        self.categories = categories
        self.reinitialize(generator, discriminator)

    def reinitialize(self, G, D):
        #Only init G and D if they are not specified
        if G is None:
            gopt = optimizers.Adam(lr=0.0002)
            G = self.generator(gopt)
        else :
            gopt = G.optimizer

        if D is None:
            dopt = optimizers.Adam(lr=0.0002)
            D = self.discriminator(dopt)

        #Freeze weights in discriminator
        for layer in D.layers:
            layer.trainable = False

        GAN_in = Input(shape=(self.z_len,))
        if self.categories > 0:
            GANc_in = Input(shape=(self.categories,))
            GAN_hidden = G([GAN_in, GANc_in])
            GAN_hidden = D([GAN_hidden, GANc_in])
            GAN = Model([GAN_in, GANc_in], GAN_hidden)
        else :
            GAN_hidden = G(GAN_in)
            GAN_hidden = D(GAN_hidden)
            GAN = Model(GAN_in, GAN_hidden)

        GAN.compile(loss='categorical_crossentropy', optimizer=gopt)
        GAN.summary()

        self.G = G
        self.D = D
        self.GAN = GAN

    ####################
    ##### Building #####
    ####################

    # outputs (input for Model(), concatenation of regular and conditional variable input)
    # Conditional variable input can be specified by c_inp

    # MNIST architecture partially from https://github.com/mingyuliutw/CoGAN_PyTorch/blob/master/src/net_gan_mnist.py
    # Due to limited padding options, the architecture has been changed.
    def generator(self, gopt) :
        def conditional_layer(inp, c_inp=None):
            if self.categories > 0:
                if c_inp == None:
                    c_inp = Input(shape=(self.categories,))
                hidden = merge([inp, c_inp], mode= 'concat')
                inp = [inp, c_inp]
            else: 
                hidden = inp

            return (inp, hidden)
        
        def layers(hidden):
            if self.data_shape == (28,28,1): #MNIST
                hidden = Reshape((1,1) + hidden._keras_shape[1:])(hidden)
                m = 4
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

                hidden = Conv2D(1, 6, strides=1, padding='same', kernel_initializer='glorot_uniform')(hidden)
                return Activation('tanh')(hidden)

            elif self.data_shape == (2):
                activation = 'relu'
                hidden = Dense(20, kernel_initializer='glorot_uniform')(hidden)
                hidden = Activation(activation)(hidden)
                hidden = Dense(20, kernel_initializer='glorot_uniform')(hidden)
                hidden = Activation(activation)(hidden)
                out_len = 2
                return Dense(out_len)(hidden)
            else: 
                raise RuntimeError("No default generator for data shape: " + str(data_shape))

        inp = Input(shape=(self.z_len,))
        (inp, hidden) = conditional_layer(inp)
        
        hidden = layers(hidden)

        G = Model(inp, hidden)
        G.compile(loss='categorical_crossentropy', optimizer=gopt)
        G.summary()

        return G

    def discriminator(self, dopt) :
        def feature_extractor(hidden):
            m = 2
            if self.data_shape == (28,28,1): #MNIST
                hidden = Conv2D(2*m, 5, strides=2, kernel_initializer='glorot_uniform')(hidden)
                hidden = LeakyReLU()(hidden)
                hidden = Conv2D(5*m, 5, strides=2, kernel_initializer='glorot_uniform')(hidden)
                hidden = LeakyReLU()(hidden)
                hidden = Conv2D(50*m, 4, strides=1, kernel_initializer='glorot_uniform')(hidden)
                hidden = LeakyReLU()(hidden)
                return Flatten()(hidden)

            elif self.data_shape == (2):
                activation = 'relu'
                hidden = Dense(20, kernel_initializer='glorot_uniform')(hidden)
                hidden = Activation(activation)(hidden)
                hidden = Dense(20, kernel_initializer='glorot_uniform')(hidden)
                return Activation(activation)(hidden)
            else: 
                raise RuntimeError("No default discriminator for data shape: " + str(data_shape))

        f_inp = Input(shape=self.data_shape)
        f_hidden = feature_extractor(f_inp)

        if self.categories > 0:
            c_inp = Input(shape=(self.categories,))
            c_hidden = Dense(100, kernel_initializer='glorot_uniform')(c_inp)
            hidden = merge([f_hidden, c_hidden], mode= 'concat')
            inp = [f_inp, c_inp]
        else: 
            hidden = f_hidden
            inp = f_inp

        hidden = Dense(2, kernel_initializer='glorot_uniform')(hidden)
        hidden = Activation('softmax')(hidden)

        D = Model(inp, hidden)
        D.compile(loss='categorical_crossentropy', optimizer=dopt)
        D.summary()

        return D


    ####################
    ##### Training #####
    ####################
    def sample_noise(self, batch_size) :
        rands = np.random.uniform(-1, 1, (batch_size, self.z_len))
        return rands

    def sample_data(self, batch_size, x_train) :
        rands = np.random.randint(len(x_train), size=(batch_size,))
        return (x_train[rands], rands)


    def to_one_hot(self, int_array):
        maximum = max(int_array)
        width = maximum+1
        output = np.zeros((len(int_array), width))
        output[range(len(int_array)), int_array] = 1
        return output

    def train(self, x_train, c_train=None, mini_batch_size=128, k=1, nr_batches=25000, vis_step=10, vis_dim=10):
        if self.categories != 0 and c_train is None:
            raise RuntimeError("c_train not specified for conditional GAN")

        if self.categories != 0 and c_train.shape[1] != self.categories:
            raise RuntimeError("c_train shape "+str(c_train.shape)+" does not agree with number of conditional variables ("+str(self.categories)+")")

        if vis_step != 0:
            visualization_noise = self.sample_noise(vis_dim*vis_dim)

        print("Starting training:")
        for batch in range(nr_batches):
            if vis_step!=0 and (batch%vis_step == 0 or it+1 == nr_batches):
                self.save_images(visualization_noise, vis_dim, batch)

            print("\rBatch " + str(batch+1) + "/" + str(nr_batches), end='\r')
            
            y_real = np.vstack((np.ones(mini_batch_size), np.zeros(mini_batch_size))).T
            y_fake = 1-y_real
            y = np.append(y_real, y_fake, axis=0)
            
            for it in range(k):
                # Get training data for D
                z = self.sample_noise(mini_batch_size)
                (x_real, idcs) = self.sample_data(mini_batch_size, x_train)
                
                if self.categories: #for conditional input
                    c_fake = self.to_one_hot(np.random.randint(self.categories, size=(mini_batch_size,)))
                    z = [z, c_fake]
                
                x_fake = self.G.predict(z)
            
                x = np.append(x_real, x_fake, axis=0)
                
                if self.categories: #for conditional input
                    c_real = c_train[idcs]
                    x = [x, np.append(c_real, c_fake,axis=0)]

                self.D.train_on_batch(x, y)

            # Update D
            z = self.sample_noise(mini_batch_size)
            if self.categories: #for conditional input
                c_fake = self.to_one_hot(np.random.randint(self.categories, size=(mini_batch_size,)))
                z = [z, c_fake] 
            self.GAN.train_on_batch(z, y_real)

        print("Batch " + str(batch+1) + "/" + str(nr_batches))

    ####################
    ##### Plotting #####
    ####################

    def save_images(self, noise, dim, iteration, savefolder='train_images') :
        if self.categories == 0:
            fake = self.G.predict(noise)
        else :
            fake = self.G.predict([noise, to_one_hot(np.arange(len(noise))%self.categories)])
        
        x = fake.shape[1] 
        y = fake.shape[2]

        image = np.zeros((dim*x, dim*y))
        
        for ity in range(dim):
            for itx in range(dim):
                xstart = itx*x
                ystart = ity*y
                image[xstart:xstart+x,ystart:ystart+y] = fake[itx+dim*ity,:,:,0]
        np.save(savefolder + '/' + str(iteration), image)

################
##### Main #####
################

def main() :
    import data
    ### Set params
    type_of_data = 'mnist' # 'mnist'  'lisajou'  'gaussians on circle'
    categories = 10 #only needed for gaussians
    data_size = (60000, 10000)

    if type_of_data == 'lisajou' : 
        categories = 0
    elif type_of_data == 'mnist' :
        categories = 10

    x_train, c_train, x_test, c_test = data.get_data(type_of_data, categories, data_size)

    categories = 0
    if categories == 0:
        c_train = None    

    gan = GAN(x_train.shape[1:], categories)
    gan.train(x_train, c_train)

if __name__ == "__main__":
    main()
