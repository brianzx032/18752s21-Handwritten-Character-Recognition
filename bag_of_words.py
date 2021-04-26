from os.path import join
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

import util
import visual_words
from opts import get_opts

opts = get_opts()
visual_words.set_opts(opts)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
batch_size = 64
dataset = torchvision.datasets.ImageFolder(
    "./data/classified/", transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

n_cpu = util.get_num_CPU()
for x, y in loader:
    img = np.moveaxis(x.numpy(),1,-1).reshape(batch_size,224,224,3)
    # filter_responses = visual_words.extract_filter_responses(opts, img)
    # util.display_filter_responses(opts, filter_responses)
    visual_words.compute_response(opts,img,y, n_worker=n_cpu)
visual_words.compute_dictionary()
    
