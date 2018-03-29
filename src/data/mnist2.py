import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import os   

import numpy as np
from PIL import Image

class MNIST(Dataset) :
    def __init__(self, labels, img_type='original', transform=None, root='../../../data/mnist') :
        if not len(set(labels)) == len(labels):
            raise RuntimeError("labels must not contain duplicates")
        
        self.dataset = datasets.MNIST(root=root, download=True)
        self.transform = transform
        self.img_type = img_type
        label_file_name = os.path.join(root, 'labels.pkl')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.img_type == 'edge':
            img_np = np.array(img)
            edge_np = cv2.Canny(img_np,0,0)
            img = Image.fromarray(edge_np)
        
        if self.transform:
            img = self.transform(img)
            
        return img, label


