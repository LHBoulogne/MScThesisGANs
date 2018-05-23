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

    def load_model(self) : 
        self.get_model()
        state = torch.load(os.path.join(self.config.loadfolder, 'classfier_' + str(self.config.load_epoch) + '.h5'), map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state)
        
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

        count = {}
        correct = {}
        for cat_idx, classes in enumerate(self.config.categories):
            count[cat_idx] = np.zeros(classes)
            correct[cat_idx] = np.zeros(classes)

        print("Testing classifier", ':')
        for batch, data in enumerate(test_loader):
            if batch%classes == 0:
                print('\r', '%.2f'%(100 * batch / len(test_loader)),'%', end='\r')

            # get disc input
            x, c = utils.cuda(data) #read out data tuple
            out = self.model(Variable(x))

            # get predictions
            for cat_idx, classes in enumerate(self.config.categories):
                cond = c[:,cat_idx]
                prd = out[cat_idx].data
                _, predicted = torch.max(prd, 1)

                count[cat_idx] += [(cond == it).sum() for it in range(classes)]
                for it in range(classes):
                    idcs = (cond == it).nonzero().squeeze()
                    if len(idcs>0) :
                        correct[cat_idx][it] += (cond[idcs] == predicted[idcs]).sum() 

        print("Accuracy:")
        for cat_idx, classes in enumerate(self.config.categories):
            print("Class nr" ,cat_idx, 'with', classes, 'categories:')
            print('count:\t', count[cat_idx])
            acc = [(100*correct[cat_idx][it]/count[cat_idx][it]) for it in range(classes)]
            print('acc:\t', ['%.2f' % x for x in acc])
            tot_count = count[cat_idx].sum()
            tot_acc = sum([count[cat_idx][it]*acc[it] for it in range(classes)])/tot_count
            print('Total:')
            print('count:\t', tot_count)
            print('acc:\t', '%.2f' % tot_acc, '\n')

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

            
            

