from torchvision import datasets as dset
from torchvision import transforms
import torch
import os
import numpy as np

class CelebA_dataset_coupled(torch.utils.data.Dataset):
    def __init__(self, config, root='../../../data/celeba/', transform=None) :

        self.img_dataset = dset.ImageFolder(root=root, transform=transform)
        
        with open(os.path.join(root, "list_attr_celeba.txt")) as f:
            contents= list(f)

        all_labelnames = contents[1].split()
        colabel_idx = all_labelnames.index(config.colabelname) + 1 # +1 to ignore the file name entry
        classlabel_idx = all_labelnames.index(config.classlabelname) + 1 # +1 to ignore the file name entry

        contents = contents[2:]
        self.a_idcs = []
        self.b_idcs = []
        self.a_labels = []
        self.b_labels = []
        for idx, line in enumerate(contents):
            colabel = int(line.split()[colabel_idx])
            classlabel = int(line.split()[classlabel_idx])
            classlabel = (classlabel + 1) // 2 # scale from -1, 1 to 0, 1
            if colabel > 0:
                if classlabel in config.labels1:
                    self.a_idcs += [idx]
                    self.a_labels += [classlabel]
            else:
                if classlabel in config.labels2:
                    self.b_idcs += [idx] 
                    self.b_labels += [classlabel]

        self.length = min(len(self.a_idcs), len(self.b_idcs))

    def get_label_name(self):
        return self.colabelname

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        a_idx = self.a_idcs[idx]
        a = self.img_dataset[a_idx][0]
        a_label = self.a_labels[idx]
        
        b_idx = self.b_idcs[idx]
        b = self.img_dataset[b_idx][0]
        b_label = self.b_labels[idx]
        
        return a, b, a_label, b_label

if __name__ == "__main__":
    dataset = CelebA_dataset_coupled('Male', "../../data/celeba", transforms.ToTensor())
    for it in range(10):
        dataset[it]
