
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

def show(image):
    if image.shape[2] == 3:
        print(np.min(image))
        print(np.max(image))
        plt.imshow(image)
    else:
        plt.imshow(image[:,:,0], cmap='gray')


step = int(sys.argv[1])

wd = os.path.join('./', sys.argv[2])
while True:
    files = os.listdir(wd)

    imgfiles = []
    for file in files:
        if '.npy' in file:
            imgfiles.append(file)

    imgfiles.sort(key=lambda f: int(''.join(list(filter(str.isdigit, f)))))

    for idx, file in enumerate(imgfiles):
        if idx % step == 0 or idx+1 == len(imgfiles):
            image = np.load(os.path.join(wd, file))
            plt.figure(1)
            plt.gcf().clear()
            
            show(image)
            plt.title('batch: ' + ''.join(list(filter(str.isdigit, file))))
            plt.draw()
            plt.pause(0.0001)

    show(image)
    plt.draw()
    plt.pause(5)
