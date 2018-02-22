from torch.utils.data import Dataset
import numpy as np

#Samples randomly from dataset1 and dataset2.
class CoupledDataset(Dataset) :
    def __init__(self, config, dataset1, dataset2) :
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.length = config.batches * config.mini_batch_size

    def __len__(self):
        return self.length

    def get_random_labelbatch(self, batchsize):
        c1 = self.dataset1.get_random_labelbatch(batchsize)
        c2 = self.dataset2.get_random_labelbatch(batchsize)
        return c1, c2

    def __getitem__(self, idx):
        np.random.seed()
        idx1 = np.random.randint(len(self.dataset1))
        idx2 = np.random.randint(len(self.dataset2))
        
        im1, lab1 = self.dataset1[idx1]
        im2, lab2 = self.dataset2[idx2]
        
        return im1, im2, lab1, lab2

if __name__ == "__main__":

    from celeba import *

    def get_celeba_dataset(pos_labels, neg_labels, domain_label=None, domain_val=None):
        return CelebA_dataset(root='../../data/celeba/', 
              labelnames=['Smiling'], pos_labels=pos_labels, neg_labels=neg_labels, 
              domain_label=domain_label, domain_val=domain_val,
              transform=transforms.ToTensor())

    class MiniConfig():
        def __init__(self):
            self.batches = 10
            self.mini_batch_size = 10

    labels1 = ['Smiling']
    labels2 = labels2_neg = labels1_neg = []
    domainlabel = 'Male'
    dataset1 = get_celeba_dataset(labels1, labels1_neg, domainlabel, 0)
    dataset2 = get_celeba_dataset(labels2, labels2_neg, domainlabel, 1)
    dataset = CoupledDataset(MiniConfig(), dataset1, dataset2)


    import matplotlib.pyplot as plt
    for it in range(10):
        im1, im2, lab1, lab2 = dataset[it]
        print('First Domain:')
        print(lab1.numpy())
        im = np.swapaxes(np.swapaxes(im1.numpy(), 0, 2), 0,1)
        plt.imshow(im)
        plt.show()

        print('Second Domain:')
        print(lab2.numpy())
        im = np.swapaxes(np.swapaxes(im2.numpy(), 0, 2), 0,1)
        plt.imshow(im)
        plt.show()
