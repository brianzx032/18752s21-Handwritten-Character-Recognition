import multiprocessing
from os.path import join

import numpy as np
import torchvision
from skimage.feature import hog
from torch.utils.data import DataLoader

import util
from opts import get_opts

opts = get_opts()


def get_one_hog(img):
    img = np.moveaxis(np.array(img), 0, -1).reshape(64, 64, 3)
    hog_feat = None
    for scale in range(opts.stack):
        fd, hog_image = hog(img, orientations=8, pixels_per_cell=(2**(scale+1), 2**(scale+1)),
                            cells_per_block=(1, 1), visualize=True, multichannel=True)
        try:
            hog_feat = np.dstack((hog_feat, hog_image))
        except:
            hog_feat = hog_image
    return hog_feat.reshape(1, -1).squeeze()


def extract_hog_stack():
    dataset = torchvision.datasets.ImageFolder(
        opts.data_dir, transform=util.transform(64))
    dataloader = DataLoader(dataset, batch_size=500, shuffle=False)
    features = None
    print("Extracting HOG stacked...")
    for x, _ in dataloader:
        with multiprocessing.Pool(util.get_num_CPU()) as p:
            hog_feat = np.array(p.map(get_one_hog, x.tolist()))
        try:
            features = np.vstack((features, hog_feat))
        except:
            features = hog_feat
        print("Extracted:", features.shape)

    np.savez_compressed(join(opts.feat_dir, 'hog_stack.npz'),
                        features=features,
                        labels=dataset.targets,
                        )


if __name__ == "__main__":
    extract_hog_stack()
