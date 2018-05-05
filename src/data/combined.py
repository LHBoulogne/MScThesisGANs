from torch.utils.data import Dataset
import numpy as np

#Samples randomly from dataset1 and dataset2.
class CombinedDataset(Dataset) :
    def __init__(self, dataset1, dataset2) :
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.length = len(dataset1) + len(dataset2)

    def __len__(self):
        return self.length

    def get_random_labelbatch(self, batchsize):
        n = np.random.randint(self.length)
        if n < len(self.dataset1):
            return self.dataset1.get_random_labelbatch(batchsize)
        return self.dataset2.get_random_labelbatch(batchsize)

    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            return self.dataset1[idx]
        return self.dataset2[idx-len(self.dataset1)]