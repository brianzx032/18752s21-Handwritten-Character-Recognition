'''
Reference: 16720 Computer Vision S21 HW1
'''

import multiprocessing
from os.path import join

import cv2
import numpy as np
import skimage.color
import skimage.feature
from sklearn.cluster import KMeans
import scipy.spatial.distance

g_opts = None
g_dictionary = None


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
    result_img = skimage.feature.corner_fast(
        img, n=opts.hog_n, threshold=opts.hog_thres)
    locs = skimage.feature.corner_peaks(result_img, min_distance=1)
    H, W = result_img.shape[0], result_img.shape[1]
    # sort
    ind = np.lexsort((locs[:, 1], locs[:, 0]))
    locs = locs[ind]
    m = opts.pattern_size//2
    n = opts.pattern_size - m
    for i in range(opts.alpha):
        try:
            n = np.min([H - locs[i, 0], W - locs[i, 1], n])
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
        feat_trained = np.load(join(feat_dir, "hog_corner.npz"))
        feat = feat_trained["features"]
        label = feat_trained["labels"]
        feat = np.vstack((feat, result))
        label = np.hstack((label, y))
    except:
        feat = result
        label = y
    print("Extracted:",feat.shape)
    np.savez_compressed(join(feat_dir, 'hog_corner.npz'),
                        features=feat,
                        labels=label,
                        )


def compute_dictionary():
    global g_opts
    feat_dir = g_opts.feat_dir
    out_dir = g_opts.out_dir
    feat = np.load(join(feat_dir, "hog_corner.npz"))["features"]
    feat = feat.reshape((feat.shape[0]*feat.shape[1], feat.shape[2]))
    print("Start clustering...")
    kmeans = KMeans(n_clusters=g_opts.K).fit(feat)
    dictionary = kmeans.cluster_centers_
    print("K-means clustering OK")
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

def get_hist_cnt(cur_l):
    return (4**(cur_l)-1)//3


def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.
    '''

    K = opts.K
    L = opts.L
    [H, W] = wordmap.shape
    hist_all = None

    ''' calculate normed hists of finest layer '''
    cur_l = int(L)
    h_finest = H // 2**cur_l
    w_finest = W // 2**cur_l
    for i in range(2**cur_l):
        for j in range(2**cur_l):
            hist_finest, bins = np.histogram(
                wordmap[i*h_finest:(i+1)*h_finest, j*w_finest:(j+1)*w_finest],
                np.arange(K+1))
            if hist_all is None:
                hist_all = hist_finest.astype(float)
            else:
                hist_all = np.hstack((hist_all, hist_finest.astype(float)))

    ''' calculate hists of other layers by combining hists of next layer '''
    while cur_l:
        cur_l -= 1  # when computing level 0, it also works
        hist_layer = None
        for i in range(2**cur_l):
            for j in range(2**cur_l):
                idx = 2**(cur_l+2)*i+2*j
                hist_cell = hist_all[idx*K:(idx+1)*K] +\
                    hist_all[(idx+1)*K:(idx+2)*K] + \
                    hist_all[(idx+2**(cur_l+1))*K:(idx+2**(cur_l+1)+1)*K] + \
                    hist_all[(idx+2**(cur_l+1)+1)*K:(idx+2**(cur_l+1)+2)*K]
                if hist_layer is None:
                    hist_layer = hist_cell
                else:
                    hist_layer = np.hstack((hist_layer, hist_cell))
        # hists of finer layer in the back
        hist_all = np.hstack((hist_layer, hist_all))

    ''' normalize by hist '''
    for hist in range(get_hist_cnt(L+1)):
        hist_all[hist*K:(hist+1)*K], bins = np.histogram(
            np.arange(K), np.arange(K+1), density=True,
            weights=hist_all[hist*K:(hist+1)*K])

    ''' weight by layer '''
    for cur_l in range(L+1):
        idx = get_hist_cnt(cur_l)
        hist_all[idx*K:(idx+4**cur_l)*K] *= 2.0**(cur_l-L-1)
    hist_all[:K] *= 2  # coarsest layer

    ''' normalize final hist '''
    hist_all, bins = np.histogram(np.arange(get_hist_cnt(
        L+1)*K), np.arange(get_hist_cnt(L+1)*K+1), density=True,
        weights=hist_all)
    return hist_all


def get_image_feature(img):
    ''' called by multiprocessing and call get_image_feature()'''
    global g_opts, g_dictionary
    img = np.array(img).astype(np.float32)/255
    wordmap = get_visual_words(g_opts, img, g_dictionary)
    return get_feature_from_wordmap_SPM(g_opts, wordmap)


def get_feat_from_resp(resp):
    global g_opts
    resp = resp.reshape(g_opts.pattern_size, g_opts.pattern_size, -1)
    wordmap = get_visual_words_from_resp(
        g_opts, resp, g_dictionary)
    return get_feature_from_wordmap_SPM(g_opts, wordmap)


def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from
    all training images.
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    feat_dir = opts.feat_dir
    SPM_layer_num = opts.L

    trained_feat = np.load(join(feat_dir, "hog_corner.npz"))
    train_resps = trained_feat['features']
    train_labels = trained_feat['labels']
    dictionary = np.load(join(feat_dir, 'bow_dictionary.npy'))

    features = None

    global g_opts, g_dictionary
    g_opts = opts
    g_dictionary = dictionary

    with multiprocessing.Pool(n_worker) as p:
        features = np.array(p.map(get_feat_from_resp, train_resps))

    g_opts = None
    g_dictionary = None
    print("Extracted:", features.shape)
    np.savez_compressed(join(feat_dir, 'bow_trained_system.npz'),
                        features=features,
                        labels=train_labels,
                        dictionary=dictionary,
                        SPM_layer_num=SPM_layer_num,
                        )
