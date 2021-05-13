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
import visual_recog
from opts import get_opts

g_opts = get_opts()
visual_words.set_opts(g_opts)


def extract_features(loader, n_cpu):
    global g_opts
    print("extract_features")
    np.savez_compressed(join(g_opts.feat_dir, 'hog_corner_feat.npz'),
                        features=None, labels=None,)
    for x, y in loader:
        img = np.moveaxis(x.numpy(), 1, -1).reshape(y.size(0), 32, 32, 3)
        visual_words.compute_response(g_opts, img, y, n_worker=n_cpu)
    feat = np.load(join(g_opts.feat_dir, 'hog_corner_feat.npz'))["features"]
    print(feat.shape)


def main(cmd, opts):
    global g_opts
    g_opts = opts
    visual_words.set_opts(opts)
    dataset = torchvision.datasets.ImageFolder(
        "./data/classified/", transform=util.transform(32))
    test_num = len(dataset)//5
    train_num = len(dataset)-test_num
    trainset, testset = torch.utils.data.random_split(
        dataset, [train_num, test_num])
    trainloader = DataLoader(
        trainset, batch_size=g_opts.batch_size, shuffle=True)
    testloader = DataLoader(
        testset, batch_size=g_opts.batch_size, shuffle=False)
    n_cpu = util.get_num_CPU()
    if "extract" in cmd:
        extract_features(trainloader, n_cpu)
    if "build" in cmd:
        visual_words.compute_dictionary()
        visual_recog.build_recognition_system(g_opts, n_cpu)


if __name__ == "__main__":
    main(["extract"], g_opts)
    main(["build"], g_opts)