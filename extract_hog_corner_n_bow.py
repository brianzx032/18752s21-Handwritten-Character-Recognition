import string
from os.path import join
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, random_split

import util
import helper
from opts import get_opts

g_opts = get_opts()
helper.set_opts(g_opts)


def extract_hog_corner(loader, n_cpu):
    global g_opts
    print("Extracting HOG corner features...")
    np.savez_compressed(join(g_opts.feat_dir, 'hog_corner.npz'),
                        features=None, labels=None,)
    for x, y in loader:
        img = np.moveaxis(x.numpy(), 1, -1).reshape(y.size(0), 32, 32, 3)
        helper.compute_response(g_opts, img, y, n_worker=n_cpu)


def extract(cmd, opts):
    global g_opts
    g_opts = opts
    helper.set_opts(opts)
    dataset = torchvision.datasets.ImageFolder(
        "./data/classified/", transform=util.transform(32))
    test_num = len(dataset)//5
    train_num = len(dataset)-test_num
    trainset, testset = random_split(
        dataset, [train_num, test_num])
    trainloader = DataLoader(
        trainset, batch_size=g_opts.batch_size, shuffle=True)
    testloader = DataLoader(
        testset, batch_size=g_opts.batch_size, shuffle=False)
    n_cpu = util.get_num_CPU()
    if "hog_corner" in cmd:
        extract_hog_corner(trainloader, n_cpu)
    if "bow_feat" in cmd:
        print("Extracting BOW features...")
        helper.compute_dictionary()
        helper.build_recognition_system(g_opts, n_cpu)


if __name__ == "__main__":
    extract(["hog_corner"], g_opts)
    extract(["bow_feat"], g_opts)
