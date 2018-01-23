from torchvision import datasets as dset
from torchvision import transforms
import torch
import os
import numpy as np
import random

class CelebA_dataset(torch.utils.data.Dataset):
    def __init__(self, labelnames=None, root='../../../data/celeba/', transform=None) :
        self.img_dataset = dset.ImageFolder(root=root, transform=transform)
        self.labelnames = labelnames

        self.length = len(self.img_dataset)

        if self.labelnames:
            with open(os.path.join(root, "list_attr_celeba.txt")) as f:
                self.contents= list(f)

            all_labelnames = self.contents[1].split()
            self.label_idcs = []
            for labelname in labelnames:
                self.label_idcs += [all_labelnames.index(labelname) + 1] # +1 to ignore the file name entry

            self.contents = self.contents[2:]
            for idx in range(len(self.contents)):
                labels = self.contents[idx].split()
                labels = [1 if labels[i] == '1' else 0 for i in self.label_idcs]
                self.contents[idx] = labels



    def get_label_names(self):
        return self.labelnames

    def __len__(self):
        return self.length

    def get_labels(self, idx):
        return torch.FloatTensor(self.contents[idx])

    def get_random_labels(self, batch_size):
        list_of_labels = []
        for it in range(batch_size):
            idx = random.randrange(self.length)
            list_of_labels += [self.contents[idx]]
        return torch.FloatTensor(list_of_labels)

    def __getitem__(self, idx):
        if not self.labelnames:
            return self.img_dataset[idx][0]

        return self.img_dataset[idx][0], self.get_labels(idx)

if __name__ == "__main__":
    dataset = CelebA_dataset(['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive'], "../../data/celeba", transforms.ToTensor())
    for it in range(10):
        print(dataset[it])
