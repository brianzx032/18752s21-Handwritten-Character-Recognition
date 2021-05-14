from os.path import join

import mahotas
import matplotlib.pyplot as plt
import numpy as np
import skimage.color
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

import util
from opts import get_opts

opts = get_opts()

def extract_zernike():
    dataset = torchvision.datasets.ImageFolder(
        opts.data_dir, transform=util.transform(64))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print("Extracting zernike features...")
    features = None
    for x, _ in dataloader:
        image = np.moveaxis(x.numpy().squeeze(), 0, 2)
        image = skimage.color.rgb2gray(image)
        zm_feat = mahotas.features.zernike_moments(image, 21, degree=8)
        try:
            features = np.vstack((features, zm_feat.reshape(1, -1)))
        except:
            features = zm_feat.reshape(1, -1)

    print("Extracted:",features.shape)
    np.savez_compressed(join(opts.feat_dir, 'zernike.npz'),
                        features=features,
                        labels=dataset.targets,
                        )


if __name__ == "__main__":
    extract_zernike()
