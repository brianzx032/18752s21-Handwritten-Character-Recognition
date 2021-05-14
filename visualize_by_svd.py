import string
from os.path import join

import mahotas
import matplotlib.pyplot as plt
import numpy as np
import skimage.color
import torchvision
from cycler import cycler
from skimage.feature import hog
from torch.utils.data import DataLoader

import util
from opts import get_opts

opts = get_opts()

features, labels = util.load_features(opts)

U, Sig, V = np.linalg.svd(features)
bases = V[:3, :]
coord = np.linalg.solve(bases@bases.T, bases@features.T)

rand_idx = np.random.randint(0, features.shape[0], 5)
if opts.feature == "hs":
    util.display_features(features[rand_idx], (64, 64, 3), 5, opts)
    util.display_features(V[:5, :], (64, 64, 3),  5, opts)
if opts.feature == "hc":
    util.display_features(
        features[rand_idx], (opts.pattern_size, opts.pattern_size, opts.alpha), 5, opts)
    util.display_features(
        V[:5, :], (opts.pattern_size, opts.pattern_size, opts.alpha),  5, opts)
if opts.feature == "bow":
    feat_num = features.shape[1]
    plt.hist(range(feat_num), range(feat_num+1), weights=features[0, :])
    plt.hist(range(feat_num), range(feat_num+1), weights=V[0, :])
if opts.feature == "ae":
    util.display_features(features[rand_idx], (4, 4, 32), 5, opts)
    util.display_features(V[:5, :], (4, 4, 32),  5, opts)
if opts.feature == "orig":
    util.display_features(np.moveaxis(features[rand_idx].reshape(
        (5, 3, 64, 64)), 1, -1), (64, 64, 3), 5, opts)
    util.display_features(np.moveaxis(V[:5, :].reshape(
        (5, 3, 64, 64)), 1, -1), (64, 64, 3),  5, opts)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
colorcycle = cycler(color=['blue', 'orange', 'green', 'red',
                    'magenta', 'cyan', 'indigo', 'crimson', 'gray', 'pink'])
plt.gca().set_prop_cycle(colorcycle)
bias = 20*5
for c in range(5, 15):
    idx = (labels == c)
    xs = coord[0, idx]
    ys = coord[1, idx]
    zs = coord[2, idx]
    ax.scatter(xs, ys, zs, s=10)
plt.legend([str(_) for _ in range(5, 10)]+[string.ascii_uppercase[_]
           for _ in range(5)])
plt.tight_layout()
ax.set_xlabel('dim 1')
ax.set_ylabel('dim 2')
ax.set_zlabel('dim 3')
plt.title(opts.feature)
plt.show()
