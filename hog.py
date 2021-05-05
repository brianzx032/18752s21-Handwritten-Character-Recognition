import string
from os.path import join
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

import util
import visual_words
from opts import get_opts
from skimage.feature import hog

opts = get_opts()
visual_words.set_opts(opts)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def main():
    dataset = torchvision.datasets.ImageFolder(
        "./data/classified/", transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    features = None
    for x, y in dataloader:
        img = np.moveaxis(x.numpy(), 1, -1).reshape(64, 64, 3)
        hog_feat = None
        for scale in range (3):
            fd, hog_image = hog(img, orientations=8, pixels_per_cell=(2**(scale+1), 2**(scale+1)),
                            cells_per_block=(1, 1), visualize=True, multichannel=True)
            try:
                hog_feat = np.dstack((hog_feat,hog_image))
            except:
                hog_feat = hog_image
        try:
            features = np.vstack((features,hog_feat.reshape(1,-1)))
        except:
            features = hog_feat.reshape(1,-1)
    
    np.savez_compressed(join(opts.feat_dir, 'hog.npz'),
                        features=features,
                        labels=dataset.targets,
                        )


if __name__ == "__main__":
    main()
