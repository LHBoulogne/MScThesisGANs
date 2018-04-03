#From https://github.com/mingyuliutw/CoGAN/tree/master/cogan_pytorch/src

from PIL import Image
import cv2
import os
import numpy as np
import _pickle as pickle
import gzip
import torch.utils.data as data
import torch
import urllib


class USPS(data.Dataset):
    # Num of Train = 7438, Num ot Test 1860
    def __init__(self, labels, transform=None, root='../data/usps', train=True):
        self.filename = 'usps_28x28.pkl'
        self.train = train
        self.root = root
        self.transform = transform
        self.test_set_size = 0
        self.download()
        self.labels = labels
        self.img_data, self.img_labels = self.load_samples()
        self.img_data *= 255.0
        self.img_data = self.img_data.transpose((0, 2, 3, 1))  # convert to HWC

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
        print("\r" + str(0) + '/' + str(len(self.img_data)), end='\r')
        for idx in range(len(self.img_data)):
            _, label = self.img_data[idx, ::], self.img_labels[idx]
            print("\r" + str(idx+1) + '/' + str(len(self.img_data)), end='\r')
            self.label_dict[label] += [idx]
            
        with open(filename, "wb" ) as f:
            pickle.dump(self.label_dict, f)
        print("Label dictionary saved.")

    def get_index(self, idx):
        for key in self.labels:
            listlen = len(self.label_dict[key])
            if idx < listlen:
                return self.label_dict[key][idx]
            idx -= listlen

    def __getitem__(self, idx):
        idx2 = self.get_index(idx)
        img, label = self.img_data[idx2, ::], self.img_labels[idx2]
        img = np.array(img).reshape((28,28)).astype(np.uint8)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([int(label)])
        return img, label

    def __len__(self):
        return self.length

    def download(self):
        filename = os.path.join(self.root, self.filename)
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        if os.path.isfile(filename):
            return
        print("Download %s to %s" % (self.url, filename))
        urllib.urlretrieve(self.url, filename)
        print("[DONE]")
        return

    def load_samples(self):
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, 'rb')
        data_set = pickle.load(f, encoding='latin1')
        f.close()
        if self.train:
            images = data_set[0][0]
            labels = data_set[0][1]
        else:
            images = data_set[1][0]
            labels = data_set[1][1]
            self.test_set_size = labels.shape[0]
        return images, labels

