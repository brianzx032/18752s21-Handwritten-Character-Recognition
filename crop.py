import os
import warnings

import cv2
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import skimage
import skimage.color
import skimage.filters
import skimage.io
import skimage.measure
import skimage.morphology
import skimage.restoration
import skimage.segmentation


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
# takes a color image
# returns a list of bounding boxes and black_and_white image


def findLetters(image):
    bboxes = []
    bw = None
    sigma_est = skimage.restoration.estimate_sigma(image, multichannel=False)
    im2 = skimage.restoration.denoise_wavelet(
        image, 3*sigma_est, multichannel=True)
    grey = skimage.color.rgb2gray(image)
    thresh = skimage.filters.threshold_otsu(grey)
    binary = (grey < thresh)  # .astype(np.float)
    sem = skimage.morphology.square(7)
    open2 = skimage.morphology.closing(binary, sem)
    opened = skimage.segmentation.clear_border(open2)
    labels = skimage.measure.label(opened)

    for region in skimage.measure.regionprops(labels):
        if region.area >= 100:
            bboxes.append(region.bbox)
    bw = 1.0-binary
    bw = skimage.morphology.erosion(bw, skimage.morphology.square(3))
    return bboxes, bw


for img in os.listdir('./data/raw_images'):
    im1 = skimage.img_as_float(
        skimage.io.imread(os.path.join('./data/raw_images', img)))
    bboxes, bw = findLetters(im1)

    # plt.imshow(bw)
    # for bbox in bboxes:
    #     minr, minc, maxr, maxc = bbox
    #     rect = matplotlib.patches.Rectangle((minc, minr),
    #                                         maxc - minc,
    #                                         maxr - minr,
    #                                         fill=False,
    #                                         edgecolor='red',
    #                                         linewidth=2)
    #     plt.gca().add_patch(rect)
    # plt.show()

    # crop the bounding boxes
    for idx, bbox in enumerate(bboxes):
        minr, minc, maxr, maxc = bbox
        h, w = maxr-minr, maxc-minc
        crop = bw[minr:maxr + 1, minc:maxc+1]
        p = (np.maximum(h, w)-np.array([h, w]))//2+np.maximum(h, w)//5

        # pad
        val = np.amax(crop)
        crop = np.pad(crop, ((p[0], p[0]), (p[1], p[1])), 'constant',
                      constant_values=((val, val), (val, val)))

        # blur
        crop = skimage.filters.gaussian(crop, sigma=2)

        # crop = skimage.transform.resize(crop, (32, 32))
        crop = (crop*255).astype(np.uint8)
        skimage.io.imsave("./data/unclassified/"+str(idx)+"_"+img, crop)
        print(img, ":", idx)
        # cv2.imshow("crop", crop)
        # cv2.waitKey()
