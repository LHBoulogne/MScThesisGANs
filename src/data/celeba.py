from torchvision import datasets as dset
from torchvision import transforms
import torch
import os
import numpy as np
import random

class CelebA_dataset(torch.utils.data.Dataset):
    def __init__(self, labelnames=["Male"], pos_labels=[], neg_labels=[], domain_label=None, domain_val=None, root='../../../data/celeba/', transform=None) :
        self.img_dataset = dset.ImageFolder(root=root, transform=transform)
        self.labelnames = labelnames

        with open(os.path.join(root, "list_attr_celeba.txt")) as f:
            contents= list(f)

        all_labelnames = contents[1].split()
        
        if not domain_label is None:
            domain_val = '1' if domain_val == 1 else '-1'
            domain_idx = self.names_to_idcs(all_labelnames, [domain_label])[0]
        label_idcs = self.names_to_idcs(all_labelnames, labelnames)
        pos_idcs   = self.names_to_idcs(all_labelnames, pos_labels)
        neg_idcs   = self.names_to_idcs(all_labelnames, neg_labels)

        contents = contents[2:]
        self.labels = {}
        self.valid_idcs = []
        for idx in range(len(contents)):
            y = contents[idx].split()

            # check if labeled correctly
            if domain_label is None or y[domain_idx] == domain_val:
                if all([y[i] == '1' for i in pos_idcs]) and all([y[i] != '1' for i in neg_idcs]):    
                    y = [1 if y[i] == '1' else 0 for i in label_idcs]
                    self.valid_idcs += [idx]
                    self.labels[idx] = y
        
    def names_to_idcs(self, all_labelnames, labels):
        idcs = []
        for labelname in labels:
            idcs += [all_labelnames.index(labelname) + 1] # +1 to ignore the file name entry
        return idcs

    def __len__(self):
        return len(self.valid_idcs)

    def get_random_labelbatch(self, batch_size):
        np.random.seed()
        ys = []
        for it in range(batch_size):
            idx = np.random.randint(len(self))
            idx2 = self.valid_idcs[idx]
            ys += [self.get_y(idx2).unsqueeze(0)]
        return torch.cat(ys, 0)

    def get_y(self, idx):
        return torch.FloatTensor(self.labels[idx])

    def __getitem__(self, idx):
        idx2 = self.valid_idcs[idx]
        return self.img_dataset[idx2][0], self.get_y(idx2)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = CelebA_dataset(labelnames=['Male', 'Smiling', 'Arched_Eyebrows'],  
                             pos_labels=['Smiling'],
                             neg_labels=['Male'],
                             root="../../data/celeba", 
                             transform=transforms.ToTensor())
    for it in range(10):
        print(dataset[it][1].numpy())
        im = np.swapaxes(np.swapaxes(dataset[it][0].numpy(), 0, 2), 0,1)
        plt.imshow(im)
        plt.show()
