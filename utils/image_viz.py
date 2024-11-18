import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# =========================== #
# Image Visualization Methods #
# =========================== #

def ShowImage(image, label, figsize=(5, 5), ax=None):
    # ==================================================
    # Input(s)
    # - image[np.array]     : NumPy array containing values of range [0, 1]
    # - label[string]       : Title of the plot
    # ==================================================
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.imshow(image, cmap='gray')
    ax.set_title(f'{label}')
    #ax.set_axis_off()


def ShowImages(images, labels, figsize=(5, 5), ncols=6):
    # ==================================================
    # Input(s)
    # - images[np.array]    : List of numpy arrays
    # - label[string]       : List of titles for each figure
    # ==================================================
    try: nrows = 1 + (images.shape[0] - 1) // ncols
    except: nrows = 1 + (len(images) - 1) // ncols
    
    figsize_full = (ncols * figsize[0], nrows * figsize[1])
    _, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize_full)

    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            image, label = images[idx], labels[idx]
            try: ShowImage(image, label, figsize=figsize, ax=axs[i][j])
            except: ShowImage(image, label, figsize=figsize, ax=axs[j])

    #plt.savefig('SampleImage.svg', format='svg')
    plt.show()