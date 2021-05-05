'''
Reference: 16720 Computer Vision S21 HW1
'''

import os
import multiprocessing
from os.path import join, isfile

import numpy as np
import scipy.ndimage
import skimage.color
from PIL import Image
from sklearn.cluster import KMeans
import skimage.feature
import cv2

g_opts = None


def set_opts(opts):
    global g_opts
    g_opts = opts


def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.
    '''

    if len(img.shape) == 3:
        img = skimage.color.rgb2gray(img)
    filter_responses = None
    result_img = skimage.feature.corner_fast(img, n=opts.hog_n, threshold=opts.hog_thres)
    locs = skimage.feature.corner_peaks(result_img, min_distance=1)
    H,W = result_img.shape[0],result_img.shape[1]
    # sort
    ind = np.lexsort((locs[:,1],locs[:,0]))
    locs = locs[ind]
    m = opts.pattern_size//2
    n = opts.pattern_size - m
    for i in range(opts.alpha):
        try:
            n = np.min([H - locs[i,0],W - locs[i,1],n])
            m = opts.pattern_size - n
            patch = img[locs[i][0]-m:locs[i][0]+n, locs[i][1]-m:locs[i][1]+n]
            fd, hog_pattern = skimage.feature.hog(patch, orientations=8,
                                                pixels_per_cell=(2, 2),
                                                cells_per_block=(1, 1), visualize=True, multichannel=False)
        except:
            hog_pattern = np.zeros((opts.pattern_size, opts.pattern_size))

        try:
            filter_responses = np.dstack((filter_responses, hog_pattern))
        except:
            filter_responses = hog_pattern
    # cv2.imshow("p",filter_responses[...,0])
    # cv2.waitKey()
    return filter_responses


def compute_response_one_image(img):
    '''
    Extracts a random subset of filter responses of an image and save it to
    disk. This is a worker function called by compute_dictionary.
    '''

    global g_opts
    img = np.array(img)
    sing_img_resp = extract_filter_responses(g_opts, img)

    [H, W, fr_num] = sing_img_resp.shape
    sampled_resp = sing_img_resp.reshape((H*W, fr_num))

    # print("compute_response_one_image ({})".format(os.getpid()))
    print('.', end='')
    return sampled_resp


def compute_response(opts, x, y, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    filter_responses = None
    train_files = x.tolist()

    global g_opts
    g_opts = opts

    with multiprocessing.Pool(n_worker) as p:
        result = np.array(p.map(compute_response_one_image, train_files))

    try:
        feat_trained = np.load(join(feat_dir, "bow_feature.npz"))
        feat = feat_trained["features"]
        label = feat_trained["labels"]
        feat = np.vstack((feat, result))
        label = np.hstack((label, y))
    except:
        feat = result
        label = y
    np.savez_compressed(join(feat_dir, 'bow_feature.npz'),
                        features=feat,
                        labels=label,
                        )


def compute_dictionary():
    global g_opts
    feat_dir = g_opts.feat_dir
    out_dir = g_opts.out_dir
    feat = np.load(join(feat_dir, "bow_feature.npz"))["features"]
    feat = feat.reshape((feat.shape[0]*feat.shape[1], feat.shape[2]))
    print("Start clustering...")
    kmeans = KMeans(n_clusters=g_opts.K).fit(feat)
    dictionary = kmeans.cluster_centers_
    print("Kmeans clustering OK")
    np.save(join(feat_dir, 'bow_dictionary.npy'), dictionary)


def get_visual_words_from_resp(opts, filter_responses, dictionary):
    word_map = np.zeros(filter_responses.shape[0:2])

    dist = scipy.spatial.distance.cdist(filter_responses.reshape(
        filter_responses.shape[0]*filter_responses.shape[1],
        filter_responses.shape[2]),
        dictionary, metric='euclidean')
    word_map = np.argmin(dist, axis=1).reshape(
        filter_responses.shape[0], filter_responses.shape[1])
    return word_map


def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of
    visual words.
    '''

    filter_responses = extract_filter_responses(opts, img)
    return get_visual_words_from_resp(opts, filter_responses, dictionary)
