import torch
from torchvision import transforms
import numpy as np
import glob

def mnist():
    # exchange with the corrupted mnist dataset
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])

    train = []
    for train_npz in glob.iglob(r'../../../data/corruptmnist/train_*.npz'):
        data = np.load(train_npz, mmap_mode='r')
        np_imgs,np_labels = data.f.images, data.f.labels
        imgs = transform(torch.from_numpy(np_imgs)).float()
        for ix in range(len(np_labels)):
            train.append(( imgs[ix], np_labels[ix]))

    test = []
    data = np.load('../../../data/corruptmnist/test.npz', mmap_mode='r')
    np_imgs,np_labels = data.f.images, data.f.labels
    imgs = transform(torch.from_numpy(np_imgs)).float()
    for ix in range(len(np_imgs)):
        test.append((imgs[ix], np_labels[ix]))

    return train, test

