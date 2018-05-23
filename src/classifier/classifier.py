import sys
import os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import pickle

from data.mnist import *
from data.subset import *

from gan.aux.aux import rescale
import utils

class Classifier():
    def __init__(self, config):
        self.config = config
        if not os.path.exists('../c_savedata'):
            os.mkdir('../c_savedata')
        if not os.path.exists(config.savefolder):
            os.mkdir(config.savefolder)
        
    def get_dataset(self, train):
        return MNIST([0,1,2,3,4,5,6,7,8,9], transform=transforms.Compose([
                                        transforms.Scale((self.config.imsize, self.config.imsize)),
                                        transforms.ToTensor(),
                                        transforms.Lambda(rescale)]),
                     root='../data/mnist/',
                     train=train)

    def get_model(self):
        module_name = "classifier.model." + self.config.classifier
        __import__(module_name, fromlist=["*"])
        mod = sys.modules[module_name]
        self.model = mod.Classifier(self.config)
        utils.cuda(self.model)

    def save_errors(self, errors, errortype):
        filename = os.path.join(self.config.savefolder, errortype + '_errors.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(errors, f)

    def save_idcs(self, idcs):
        filename = os.path.join(self.config.savefolder, 'train_idcs.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(idcs, f)

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.config.savefolder, 'classfier_' + str(epoch) + '.h5'))
        pass

    def save_error_plot(self, train, val):
        x = range(len(train))
        plt.plot(x, train, label='training data')
        plt.plot(x, val, label='validation data')
        plt.xlabel('epoch')
        plt.ylabel('mean loss')
        lgd = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                         ncol=2, mode="expand", borderaxespad=0.)
        plt.savefig(os.path.join(self.config.savefolder, 'loss.png'),
                    bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()


    def test(self):
        test_set = self.get_dataset(train=False)
        test_loader = torch.utils.data.DataLoader(test_set, 
            batch_size=self.config.mini_batch_size, shuffle=True, num_workers=self.config.dloadworkers)

    def train(self):
        self.get_model()
        opt = optim.Adam(self.model.parameters())
        
        trainval_set = self.get_dataset(train=True)
        train_idcs = set(np.random.choice(len(trainval_set), len(trainval_set)-self.config.val_set_len, replace=False))
        all_idcs = set(range(len(trainval_set)))
        val_idcs = all_idcs - train_idcs

        train_set = Subset(trainval_set, list(train_idcs))
        val_set = Subset(trainval_set, list(val_idcs))

        train_loader = torch.utils.data.DataLoader(train_set, 
            batch_size=self.config.mini_batch_size, shuffle=True, num_workers=self.config.dloadworkers)
        val_loader = torch.utils.data.DataLoader(val_set, 
            batch_size=self.config.mini_batch_size, shuffle=True, num_workers=self.config.dloadworkers)

        criterion = nn.CrossEntropyLoss() #Includes the softmax function

        train_errors = {}
        train_error = []
        val_error = []
        for epoch in range(self.config.epochs):
            print("\rEpoch " + str(epoch))

            #Training                
            self.model.train()
            train_errors[epoch] = []
            for batch, (data, labels) in enumerate(train_loader):
                self.model.zero_grad()
                data = utils.cuda(Variable(data))
                labels = utils.cuda(Variable(labels))

                if self.config.mini_batch_size != data.size(0):
                    continue
                print("\rBatch " + str(batch))

                output = self.model(data)
                error = 0
                for it in range(len(output)):
                    error += criterion(output[it].view(self.config.mini_batch_size,-1), labels[:,it])
                train_errors[epoch] += [error.data.cpu().numpy()]
                error.backward()
                opt.step()
            train_error += [np.mean(train_errors[epoch])]

            self.save_errors(train_errors, 'individual_train')
            self.save_errors(train_error, 'train')
            self.save_model(epoch)

            #Validation
            self.model.eval()
            val_error += [0]
            for batch, (data, labels) in enumerate(val_loader):
                data = utils.cuda(Variable(data))
                labels = utils.cuda(Variable(labels))

                if self.config.mini_batch_size != data.size(0):
                    continue
                print("\rBatch " + str(batch))

                output = self.model(data)
                error = 0
                for it in range(len(output)):
                    error += criterion(output[it].view(self.config.mini_batch_size,-1), labels[:,it])
                val_error[epoch] += error.data.cpu().numpy()

            print(batch)
            val_error[epoch] /= batch
            self.save_errors(val_error, 'val')
            self.save_error_plot(train_error, val_error)

            
            

