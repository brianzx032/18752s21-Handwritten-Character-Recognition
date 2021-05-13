import string
from os.path import join
from time import time

import mahotas
import matplotlib.pyplot as plt
import numpy as np
import skimage.color
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

import util
import visual_words
from opts import get_opts

opts = get_opts()
visual_words.set_opts(opts)

def main():
    dataset = torchvision.datasets.ImageFolder(
        "./data/classified/", transform=util.transform(64))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    features = None
    for x, _ in dataloader:
        image = np.moveaxis(x.numpy().squeeze(), 0, 2)
        image = skimage.color.rgb2gray(image)
        zm_feat = mahotas.features.zernike_moments(image, 21, degree=8)
        try:
            features = np.vstack((features, zm_feat.reshape(1, -1)))
        except:
            features = zm_feat.reshape(1, -1)

    np.savez_compressed(join(opts.feat_dir, 'zernike.npz'),
                        features=features,
                        labels=dataset.targets,
                        )


if __name__ == "__main__":
    main()
