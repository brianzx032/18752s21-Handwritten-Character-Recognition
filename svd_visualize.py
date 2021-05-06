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

npz_data = np.load(join(opts.feat_dir, 'hog_stack.npz')) 
# npz_data = np.load(join(opts.feat_dir, 'hog_corner_feat.npz'))
# npz_data = np.load(join(opts.feat_dir, 'zernike.npz'))
# npz_data = np.load(join(opts.feat_dir, 'autoencoder.npz'))
# npz_data = np.load(join(opts.feat_dir, 'bow_trained_system.npz')) 

features,labels = npz_data["features"], npz_data["labels"]
try:
    # for hog_corner_feat.npz
    features = features.reshape((-1, opts.pattern_size**2*opts.alpha))
except:
    pass


U, Sig, V = np.linalg.svd(features)
bases = V[:3, :]
coord = np.linalg.solve(bases@bases.T, bases@features.T)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
colorcycle = cycler(color=['blue', 'orange', 'green', 'red',
                    'magenta', 'cyan', 'indigo', 'crimson', 'gray', 'pink'])
plt.gca().set_prop_cycle(colorcycle)
bias = 20*5
for c in range(5,15):
    idx = (labels==c)
    xs = coord[0, idx]
    ys = coord[1, idx]
    zs = coord[2, idx]
    ax.scatter(xs, ys, zs, s=10)
plt.legend([str(_) for _ in range(5,10)]+[string.ascii_uppercase[_]
           for _ in range(5)])
plt.tight_layout()
ax.set_xlabel('dim 1')
ax.set_ylabel('dim 2')
ax.set_zlabel('dim 3')
# plt.title("zernike_moments")
# plt.title("HOG")
plt.show()
