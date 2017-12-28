import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import os   

import pickle

import numpy as np
from PIL import Image
import random

class MNIST_edge(Dataset) :
    def __init__(self, transform=None, root='../../../data/mnist', labels_original=[0,1,2,3,4,5,6,7,8,9], labels_edge=[0,1,2,3,4,5,6,7,8,9]) :
        self.dataset = datasets.MNIST(root=root, download=True)
        self.transform = transform
        label_file_name = os.path.join(root, 'labels.pkl')
        
        if os.path.exists(label_file_name):
            with open(label_file_name, "rb" ) as f:
                self.label_dict = pickle.load(f)
        else:
            self.create_label_dict(label_file_name)

        self.original_idcs = []
        self.edge_idcs = []
        for key in labels_original:
            self.original_idcs += self.label_dict[key]

        for key in labels_edge:
            self.edge_idcs += self.label_dict[key]

    def create_label_dict(self, filename) :
        print("Creating label dictionary...")
        self.label_dict = {}
        for it in range(10):
            self.label_dict[it] = []
        print("\r" + str(0) + '/' + str(self.dataset.__len__()), end='\r')
        for idx in range(self.dataset.__len__()):
            _, label = self.dataset.__getitem__(idx)
            print("\r" + str(idx+1) + '/' + str(self.dataset.__len__()), end='\r')
            self.label_dict[label] += [idx]
            
        with open(filename, "wb" ) as f:
            pickle.dump(self.label_dict, f)
        print("Label dictionary saved.")

    #Basically infinite
    def __len__(self):
        return 2*(10**6)

    #idx is not used. Random combinations of data points are returned 
    def __getitem__(self, idx):
        idx_original = random.randint(0, len(self.original_idcs))
        idx_edge     = random.randint(0, len(self.edge_idcs))
        
        original, original_label = self.dataset.__getitem__(self.original_idcs[idx_original])
        im      , edge_label     = self.dataset.__getitem__(self.edge_idcs[idx_edge])
        im_np = np.array(im)
        edge_np = cv2.Canny(im_np,0,0)
        edge = Image.fromarray(edge_np)
        
        if self.transform:
            edge     = self.transform(edge)
            original = self.transform(original)

        return original, edge, original_label, edge_label #TODO: double the label array?

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = MNIST_edge(transform=transforms.Compose([transforms.Scale((64,64)),
                                                transforms.ToTensor(),
                                                transforms.Lambda(rescale)]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=3)

    for batch, (item, edge, label) in enumerate(dataloader) :
        item = item.numpy()[0,0]
        edge = edge.numpy()[0,0]

        plt.subplot(2,1,1)
        plt.imshow(item, cmap="gray")
        plt.subplot(2,1,2)
        plt.imshow(edge, cmap="gray")
        plt.show()
        continue