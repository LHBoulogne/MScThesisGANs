
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

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

            plt.imshow(image, cmap='gray')
            plt.title('batch: ' + ''.join(list(filter(str.isdigit, file))))
            plt.draw()
            plt.pause(0.00001)

    plt.imshow(image, cmap='gray')
    plt.draw()
    plt.pause(5)
