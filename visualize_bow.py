from os.path import join

import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import util
import helper
from opts import get_opts

opts = get_opts()

dataset = torchvision.datasets.ImageFolder(
    opts.data_dir, transform=util.transform(32))
loader = DataLoader(dataset, batch_size=1, shuffle=True)

n_cpu = util.get_num_CPU()

hog = np.load(join(opts.feat_dir, "hog_stack.npz"))["features"]

for x, y in loader:
    img = np.moveaxis(x.numpy(), 1, -1).reshape(32, 32, 3)

    # view wordmap
    dictionary = np.load(join(opts.feat_dir, 'bow_dictionary.npy'))
    wordmap = helper.get_visual_words(opts, img, dictionary)
    util.visualize_wordmap(img, wordmap)

    break
