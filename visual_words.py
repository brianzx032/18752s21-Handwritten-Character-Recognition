import os
import multiprocessing
from os.path import join, isfile

import numpy as np
import scipy.ndimage
import skimage.color
from PIL import Image
from sklearn.cluster import KMeans
from skimage.feature import hog

g_opts = None


def set_opts(opts):
    global g_opts
    g_opts = opts


def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.
    '''

    filter_scales = opts.filter_scales
    # check input channels
    if len(img.shape) == 2:
        H, W = img.shape
        img = skimage.color.gray2rgb(img)
    else:
        H, W, _ = img.shape

    srcImg = skimage.color.rgb2lab(img)

    filter_responses = np.zeros((H, W, 3*5*len(filter_scales)))

    for scale in range(len(filter_scales)):
        for ch in range(3):  # gaussian
            scipy.ndimage.gaussian_filter(srcImg[:, :, ch],
                                          filter_scales[scale],
                                          output=filter_responses[:, :, 3*5*scale+ch])

        for ch in range(3):  # laplace of gaussian
            scipy.ndimage.gaussian_laplace(srcImg[:, :, ch],
                                           filter_scales[scale],
                                           output=filter_responses[:, :, 3*5*scale+ch+3])

        for ch in range(3):  # hog
            fd, hog_image = hog(srcImg[:, :, ch], orientations=8,
                                pixels_per_cell=(
                                    filter_scales[scale]*2, filter_scales[scale]*2),
                                cells_per_block=(1, 1), visualize=True, multichannel=False)
            filter_responses[:, :, 3*5*scale+ch+6] = hog_image

        for ch in range(3):  # gaussian derivitive in x
            scipy.ndimage.gaussian_filter(srcImg[:, :, ch],
                                          filter_scales[scale], order=(0, 1),
                                          output=filter_responses[:, :, 3*5*scale+ch+9])

        for ch in range(3):  # gaussian derivitive in y
            scipy.ndimage.gaussian_filter(srcImg[:, :, ch],
                                          filter_scales[scale], order=(1, 0),
                                          output=filter_responses[:, :, 3*5*scale+ch+12])

    return filter_responses


def compute_response_one_image(img):
    # def compute_response_one_image(opts,img_file):
    '''
    Extracts a random subset of filter responses of an image and save it to
    disk. This is a worker function called by compute_dictionary.
    '''

    global g_opts
    img = np.array(img)
    sing_img_resp = extract_filter_responses(g_opts, img)

    # random pick alpha pixels
    [H, W, fr_num] = sing_img_resp.shape
    rnd_idx = np.random.choice(H*W, g_opts.alpha)
    sampled_resp = sing_img_resp.reshape((H*W, fr_num))[rnd_idx[:], :]

    # print("compute_response_one_image ({})".format(os.getpid()))
    print('.',end='')
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
