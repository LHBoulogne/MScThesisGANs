import sys
import os

from data.coupled import *
from data.mnistedge import *
from data.celeba import *

from vis.visualizer import *
from vis import errorplot

from gan.trainer import *
from gan.aux.aux import rescale
import utils

from gan.aux.sample import sample_c

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
        utils.cuda(self.G)

    def init_discriminator(self):
        module_name = "gan.model.discriminator." + self.config.discriminator
        __import__(module_name, fromlist=["*"])
        mod = sys.modules[module_name]
        self.D = mod.Discriminator(self.config)
        utils.cuda(self.D)
    
    def save(self):
        torch.save(self.G.state_dict(), os.path.join(self.config.savefolder, 'generator.h5'))
        torch.save(self.D.state_dict(), os.path.join(self.config.savefolder, 'discriminator.h5'))

    def load(self) : #TODO: Make functionality that you can directly load a GAN because the config is also saved
        gstate = torch.load(os.path.join(self.config.loadfolder, 'generator.h5'), map_location=lambda storage, loc: storage)
        dstate = torch.load(os.path.join(self.config.loadfolder, 'discriminator.h5'), map_location=lambda storage, loc: storage)
        self.G.load_state_dict(gstate)
        self.D.load_state_dict(dstate)
    
    def get_celeba_dataset(self, pos_labels, neg_labels, domain_label=None, domain_val=None):
        return CelebA_dataset(root='../data/celeba/', 
              labelnames=self.config.labelnames, pos_labels=pos_labels, neg_labels=neg_labels, 
              domain_label=domain_label, domain_val=domain_val,
              transform=transforms.Compose([transforms.CenterCrop(160),
                                            transforms.Scale((self.config.imsize,self.config.imsize)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]))

    def get_dataset(self):
        if self.config.coupled: 
            if self.config.dataname == "MNIST":
                dataset = MNIST_edge(self.config, transform=transforms.Compose([transforms.Scale((self.config.imsize,self.config.imsize)),
                                               transforms.ToTensor(),
                                               transforms.Lambda(rescale)]))
            elif self.config.dataname == "CelebA":
                dataset1 = self.get_celeba_dataset(self.config.labels1, self.config.labels1_neg, self.config.domainlabel, 1)
                dataset2 = self.get_celeba_dataset(self.config.labels2, self.config.labels2_neg, self.config.domainlabel, 0)
                dataset = CoupledDataset(self.config, dataset1, dataset2)

        else :
            if self.config.dataname == "MNIST":
                dataset = dset.MNIST(root='../data/mnist', download=True, 
                      transform=transforms.Compose([transforms.Scale((self.config.imsize,self.config.imsize)),
                                                    transforms.ToTensor(),
                                                    transforms.Lambda(rescale)]))
            elif self.config.dataname == "CelebA":
                dataset = self.get_celeba_dataset(self.config.labels1, self.config.labels1_neg)

        return dataset

    def make_snapshot(self, epoch, batch, trainer, imgsaver):
        trainer.save_error()
        if self.config.visualize_training:
            imgsaver.save_training_imgs(epoch, batch, self.G)
            errorplot.save_error_plots(trainer.get_error_storage(), self.config)
        self.save()


    def train(self):
        dataset = self.get_dataset()
        dataloader = torch.utils.data.DataLoader(dataset, 
            batch_size=self.config.mini_batch_size, shuffle=True, num_workers=3)

        imgsaver = Visualizer(self.config)
        trainer = GANTrainer(self.config, self.G, self.D)

        epoch = 0
        steps_without_G_update = 0
        c_fake = None
        while epoch < self.config.epochs:
            print("Epoch: "+str(epoch+1)+ "/" + str(self.config.epochs) + ' '*10)

            for batch, data in enumerate(dataloader) :
                if batch%self.config.snap_step == 0:
                    self.make_snapshot(epoch, batch, trainer, imgsaver)
                
                print("\rBatch " + str(batch))
                
                if self.config.auxclas :
                    c_fake = sample_c(self.config, dataset)
                trainer.next_step(data, c_fake) #! Using the SAME c_fake for generator and discriminator update!

                if trainer.update_discriminator(self.G, self.D):
                    steps_without_G_update += 1
                #Allow for multiple updates of D with respect to G
                
                if steps_without_G_update >= self.config.k: 
                    steps_without_G_update = 0
                    for it in range(self.config.G_updates):
                        g_updated = False
                        while not g_updated:
                            g_updated = trainer.update_generator(self.G, self.D)

            self.make_snapshot(epoch, batch+1, trainer, imgsaver)
            epoch += 1


    def test(self):
        self.load()
        dataset = self.get_dataset()
        dataloader = torch.utils.data.DataLoader(dataset, 
            batch_size=self.config.mini_batch_size, shuffle=True, num_workers=3)

        imgsaver = Visualizer(self.config)
        imgsaver.save_test_imgs(self.G)
