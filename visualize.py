import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from os.path import join
import util
import visual_words
from opts import get_opts
from bag_of_words import transform,opts

dataset = torchvision.datasets.ImageFolder(
    "./data/classified/", transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

n_cpu = util.get_num_CPU()

hog = np.load(join(opts.feat_dir, "hog.npz"))["features"]
util.display_hog_images(hog)

for x, y in loader:
    img = np.moveaxis(x.numpy(), 1, -1).reshape(32, 32, 3)
    
    # view wordmap
    dictionary = np.load(join(opts.feat_dir, 'bow_dictionary.npy'))
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    util.visualize_wordmap(img, wordmap)

    # view filer responses
    filter_responses = visual_words.extract_filter_responses(opts, img)
    util.display_filter_responses(opts, filter_responses)

    break

