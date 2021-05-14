'''
Reference: 16720 Computer Vision S21 HW1
'''
import multiprocessing
import string
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from extract_autoencoder import extract_encoded
from extract_hog_corner_n_bow import extract
from extract_hog_stack import extract_hog_stack
from extract_zernike import extract_zernike


def get_num_CPU():
    '''
    Counts the number of CPUs available in the machine.
    '''
    return multiprocessing.cpu_count()


def transform(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_image_data(size,opts):
    dataset = torchvision.datasets.ImageFolder(
        opts.data_dir, transform=transform(size))
    loader = DataLoader(dataset, len(dataset), shuffle=False)
    features, labels = None, None
    for x, y in loader:
        features, labels = x.reshape(len(dataset), -1), y
    return features.numpy(), labels.numpy()


def load_features(opts):
    # hog stacked`
    if opts.feature == "hs":
        if opts.re_extract:
            extract_hog_stack()
        npz_data = np.load(join(opts.feat_dir, 'hog_stack.npz'))  # LR 92.53%

    # hog corner
    if opts.feature == "hc":
        if opts.re_extract:
            extract(["hog_corner"], opts)
        npz_data = np.load(join(opts.feat_dir, 'hog_corner.npz'))  # LR 73.92%

    # bag of visual words
    if opts.feature == "bow":
        if opts.re_extract:
            extract(["bow_feat"], opts)
        npz_data = np.load(
            join(opts.feat_dir, 'bow_trained_system.npz'))  # LDA 59.11%

    # zernike
    if opts.feature == "z":
        if opts.re_extract:
            extract_zernike()
        npz_data = np.load(join(opts.feat_dir, 'zernike.npz'))  # QDA 62.48%

    # autoencoder
    if opts.feature == "ae":
        if opts.re_extract:
            extract_encoded()
        npz_data = np.load(
            join(opts.feat_dir, 'autoencoder.npz'))  # LDA 63.84%

    try:
        X, y = npz_data["features"], npz_data["labels"]
        if opts.feature == "hc":
            X = X.reshape((-1, opts.pattern_size**2*opts.alpha))
    except:
        # resized images
        if opts.feature == "orig":
            X, y = get_image_data(64,opts)  # LR 85.64%`
    X -= np.sum(X, axis=0)/X.shape[0]
    return X, y


def display_features(features, s, num, opts):
    '''
    Visualizes the feature images.
    '''

    plt.figure(1)
    features = features.reshape((-1, s[0], s[1], s[2]))

    for i in range(num):
        for j in range(s[2]):
            plt.subplot(num, s[2], i*s[2] + j+1)
            feat_img = features[i, :, :, j]
            feat_img_min = feat_img.min(axis=(0, 1), keepdims=True)
            feat_img_max = feat_img.max(axis=(0, 1), keepdims=True)
            feat_img = (feat_img - feat_img_min) / \
                (feat_img_max - feat_img_min)
            plt.imshow(feat_img)
            plt.axis("off")

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95,
                        bottom=0.05, wspace=0.05, hspace=0.05)
    plt.show()


def visualize_wordmap(original_image, wordmap, out_path=None):
    '''
    Visualizes the wordmap corresponding to an image.
    '''
    fig = plt.figure(2, figsize=(12.8, 4.8))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(original_image)
    plt.axis("off")
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(wordmap)
    plt.axis("off")
    plt.show()
    if out_path:
        plt.savefig(out_path, pad_inches=0)


def visualize_confusion_matrix(confusion_matrix):
    plt.imshow(confusion_matrix, interpolation='nearest')
    plt.grid(True)
    plt.xticks(np.arange(36), ''.join(
        [str(_) for _ in range(10)])+string.ascii_uppercase[:26])
    plt.yticks(np.arange(36), ''.join(
        [str(_) for _ in range(10)])+string.ascii_uppercase[:26])
    plt.show()
