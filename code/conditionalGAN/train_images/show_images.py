
import os
import numpy as np
import matplotlib.pyplot as plt

files = os.listdir('./')

imgfiles = []
for file in files:
    if '.npy' in file:
        imgfiles.append(file)

imgfiles.sort(key=lambda f: int(filter(str.isdigit, f)))

while True:
    for file in imgfiles:
        image = np.load(file)
        plt.figure(1)
        plt.gcf().clear()

        plt.imshow(image, cmap='gray')
        plt.title('epoch: ' + filter(str.isdigit, file))
        plt.draw()
        plt.pause(0.00001)

    plt.imshow(image, cmap='gray')
    plt.show()
