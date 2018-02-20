from torch.utils.data import Dataset
import numpy as np

#Samples randomly from dataset1 and dataset2.
class CoupledDataset(Dataset) :
    def __init__(self, config, dataset1, dataset2) :
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.length = config.batches

    def __len__(self):
        return self.length

    def get_random_labelbatch(self, batchsize):
        c1 = self.dataset1.get_random_labelbatch(batchsize)
        c2 = self.dataset2.get_random_labelbatch(batchsize)
        return c1, c2

    def __getitem__(self, idx):
        idx1 = np.random.randint(len(self.dataset1))
        idx2 = np.random.randint(len(self.dataset2))
        
        im1, lab1 = self.dataset1[idx1]
        im2, lab2 = self.dataset2[idx2]
        
        return im1, im2, lab1, lab2
