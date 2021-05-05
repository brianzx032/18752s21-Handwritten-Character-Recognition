from cycler import cycler
import string
import mahotas
import matplotlib.pyplot as plt
import numpy as np
import skimage.color
import torchvision
from torch.utils.data import DataLoader
from skimage.feature import hog

from bag_of_words import opts, transform

dataset = torchvision.datasets.ImageFolder(
    "./data/classified/", transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

features = None
for x, _ in loader:
    image = np.moveaxis(x.numpy().squeeze(), 0, 2)
    image = skimage.color.rgb2gray(image)
    # zernike moments
    # feat = mahotas.features.zernike_moments(image, 21, degree=8)

    # HOG
    fd, hog_feat = hog(image, orientations=8, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), visualize=True, multichannel=False)
    feat = hog_feat.reshape(1,-1)
    try:
        features = np.vstack((features, feat))
    except:
        features = feat


U, Sig, V = np.linalg.svd(features)
bases = V[:3, :]
coord = np.linalg.solve(bases@bases.T, bases@features.T)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
colorcycle = cycler(color=['blue', 'orange', 'green', 'red',
                    'magenta', 'cyan', 'indigo', 'crimson', 'gray', 'pink'])
plt.gca().set_prop_cycle(colorcycle)
bias = 20*5
for c in range(10):
    xs = coord[0, c*20+bias:c*20+20+bias]
    ys = coord[1, c*20+bias:c*20+20+bias]
    zs = coord[2, c*20+bias:c*20+20+bias]
    ax.scatter(xs, ys, zs, s=10)
# plt.legend([str(_) for _ in range(10)]+[string.ascii_uppercase[_] for _ in range(26)])
plt.legend([str(_) for _ in range(5,10)]+[string.ascii_uppercase[_]
           for _ in range(5)])
plt.tight_layout()
ax.set_xlabel('dim 1')
ax.set_ylabel('dim 2')
ax.set_zlabel('dim 3')
# plt.title("zernike_moments")
plt.title("HOG")
plt.show()
