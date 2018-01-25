import sys

from data.celeba_coupled import *
from data.mnistedge import *
from data.celeba import *

from vis.visualizer import *
from vis import errorplot

from gan.trainer import *
from gan.aux.aux import rescale
import os

class GAN():
    def __init__(self, config):
        self.config = config
        self.init_generator()
        self.init_discriminator()
        if not os.path.exists(config.savefolder):
            os.mkdir(config.savefolder)

    def init_generator(self):
        module_name = "gan.model.generator." + self.config.generator
        __import__(module_name, fromlist=["*"])
        mod = sys.modules[module_name]
        self.G = mod.Generator(self.config)

    def init_discriminator(self):
        module_name = "gan.model.discriminator." + self.config.discriminator
        __import__(module_name, fromlist=["*"])
        mod = sys.modules[module_name]
        self.D = mod.Discriminator(self.config)
    
    def save(self):
        torch.save(self.G.state_dict(), os.path.join(self.config.savefolder, 'generator.h5'))
        torch.save(self.D.state_dict(), os.path.join(self.config.savefolder, 'discriminator.h5'))

    def load(self) : #TODO: Make functionality that you can directly load a GAN because the config is also saved
        gstate = torch.load(os.path.join(self.config.savefolder, 'generator.h5'))
        dstate = torch.load(os.path.join(self.config.savefolder, 'discriminator.h5'))
        self.G.load_state_dict(gstate)
        self.D.load_state_dict(dstate)
    
    def get_dataset(self):
        if self.config.coupled: 
            if self.config.dataname == "MNIST":
                dataset = MNIST_edge(self.config, transform=transforms.Compose([transforms.Scale((self.config.imsize,self.config.imsize)),
                                               transforms.ToTensor(),
                                               transforms.Lambda(rescale)]))
            elif self.config.dataname == "CelebA":
                dataset = CelebA_dataset_coupled(colabelname=self.config.colabelname, 
                      root='../data/celeba/', 
                      transform=transforms.Compose([transforms.CenterCrop(160),
                                                    transforms.Scale((self.config.imsize,self.config.imsize)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])) 
                      # TODO: batches=self.config.batches)
        else :
            if self.config.dataname == "MNIST":
                dataset = dset.MNIST(root='../data/mnist', download=True, 
                      transform=transforms.Compose([transforms.Scale((self.config.imsize,self.config.imsize)),
                                                    transforms.ToTensor(),
                                                    transforms.Lambda(rescale)]))
            elif self.config.dataname == "CelebA":
                dataset = dset.ImageFolder(root='../data/celeba/', 
                      transform=transforms.Compose([transforms.CenterCrop(160),
                                                    transforms.Scale((self.config.imsize,self.config.imsize)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]))
        return dataset

    def make_snapshot(self, epoch, batch, trainer, imgsaver):
        trainer.save_error()
        if self.config.visualize_training:
            imgsaver.save_training_imgs(epoch, batch, self.G)
            errorplot.save_error_plots(trainer.error_dicts, self.config)
        self.save()


    def train(self):
        dataset = self.get_dataset()
        dataloader = torch.utils.data.DataLoader(dataset, 
            batch_size=self.config.mini_batch_size, shuffle=True, num_workers=3)

        imgsaver = Visualizer(dataset, self.config)
        trainer = GANTrainer(self.config, self.G, self.D)

        epoch = 0
        steps_without_G_update = 0

        while epoch < self.config.epochs:
            print("Epoch: "+str(epoch+1)+ "/" + str(self.config.epochs) + ' '*10)
            for batch, data in enumerate(dataloader) :
                print("\rBatch " + str(batch), end='\r')

                trainer.next_step(data)
                trainer.update_discriminator(self.G, self.D)

                #Allow for multiple updates of D with respect to G
                steps_without_G_update += 1
                if steps_without_G_update == self.config.k: 
                    steps_without_G_update = 0

                    trainer.update_generator(self.G, self.D)

                if batch%self.config.snap_step == 0:
                    self.make_snapshot(epoch, batch, trainer, imgsaver)
            self.make_snapshot(epoch, batch, trainer, imgsaver)
            epoch += 1


    def test(self):
        self.load()
        dataset = self.get_dataset()
        dataloader = torch.utils.data.DataLoader(dataset, 
            batch_size=self.config.mini_batch_size, shuffle=True, num_workers=3)

        imgsaver = Visualizer(dataset, self.config)
        imgsaver.save_test_imgs(self.G)
