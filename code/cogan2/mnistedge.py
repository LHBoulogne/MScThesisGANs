import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

import numpy as np
from PIL import Image

class MNIST_edge(Dataset) :
    def __init__(self, transform=None) :
        self.dataset = datasets.MNIST(root='../../../data/mnist', download=True)
        self.transform = transform

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        im, label = self.dataset.__getitem__(idx)
        im_np = np.array(im)
        edge_np = cv2.Canny(im_np,0,0)
        edge = Image.fromarray(edge_np)
        
        if self.transform:
            edge = self.transform(edge)
            im = self.transform(im)

        return im, edge, label #TODO: double the label array?

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