from torchvision import datasets as dset
from torchvision import transforms
import torch
import os
import numpy as np

class CelebA_dataset_coupled(torch.utils.data.Dataset):
    def __init__(self, colabelname='Male', root='../../../data/celeba/', transform=None) :
        self.img_dataset = dset.ImageFolder(root=root, transform=transform)
        self.colabelname = colabelname

        with open(os.path.join(root, "list_attr_celeba.txt")) as f:
            contents= list(f)

        all_labelnames = contents[1].split()
        colabel_idx = all_labelnames.index(colabelname) + 1 # +1 to ignore the file name entry

        contents = contents[2:]
        self.a_idcs = []
        self.b_idcs = []
        for idx, line in enumerate(contents):
            label = int(line.split()[colabel_idx])
            if label > 0:
                self.a_idcs += [idx]
            else:
                self.b_idcs += [idx]

        self.length = min(len(self.a_idcs), len(self.b_idcs))

    def get_label_name(self):
        return self.colabelname

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        a_idx = self.a_idcs[idx]
        b_idx = self.b_idcs[idx]
        a = self.img_dataset[a_idx][0]
        b = self.img_dataset[b_idx][0]
        return a, b, torch.FloatTensor([0])

if __name__ == "__main__":
    dataset = CelebA_dataset_coupled('Male', "../../data/celeba", transforms.ToTensor())
    for it in range(10):
        dataset[it]
