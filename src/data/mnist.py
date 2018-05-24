import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import os   
import pickle

import numpy as np
from PIL import Image

class MNIST(Dataset) :
    def __init__(self, labels, img_type='original', transform=None, root='../../../data/mnist', train=True, domain_val=None):
        if not len(set(labels)) == len(labels):
            raise RuntimeError("labels must not contain duplicates")
        self.domain_val = domain_val
        self.labels = labels
        self.dataset = datasets.MNIST(root=root, download=True, train=train)
        self.transform = transform
        self.img_type = img_type
        
        label_file_name = os.path.join(root, 'labels_'+str(train)+'.pkl')
        
        if os.path.exists(label_file_name):
            with open(label_file_name, "rb" ) as f:
                self.label_dict = pickle.load(f)
        else:
            self.create_label_dict(label_file_name)

        self.length = sum([len(self.label_dict[key]) for key in self.labels])

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

    def __len__(self):
        return self.length

    def get_random_labelbatch(self, batch_size):
        np.random.seed()
        ys = []
        for it in range(batch_size):
            label = self.random_label()
            label = [label.unsqueeze(0)]
            ys += label
        return torch.cat(ys, 0)

    def random_label(self):
        idx = np.random.randint(len(self))
        idx2 = self.get_index(idx)
        _, label = self.dataset[idx2]
        return self.complete_label(label)

    def complete_label(self, label):
        if self.domain_val is None:
            return torch.LongTensor([label])
        return torch.LongTensor([label, self.domain_val])

    def get_index(self, idx):
        for key in self.labels:
            listlen = len(self.label_dict[key])
            if idx < listlen:
                return self.label_dict[key][idx]
            idx -= listlen

    def __getitem__(self, idx):
        idx2 = self.get_index(idx)
        img, label = self.dataset[idx2]
        if self.img_type != 'original':
            img_np = np.array(img)
            if self.img_type == 'edge':
                edge_np = cv2.Canny(img_np,0,0)
            if self.img_type == 'diledge':
                dilation = cv2.dilate(img_np, np.ones((3, 3), np.uint8), iterations=1)
                edge_np = dilation - img
            img = Image.fromarray(edge_np)
        
        if self.transform:
            img = self.transform(img)
        
        return img, self.complete_label(label)
