'''
Reference: 16720 Computer Vision S21 HW1
'''
import multiprocessing
from os.path import join

import numpy as np

import visual_words

# global variables for multiprocessing
g_opts = None
g_dictionary = None
g_features = None
g_trained_labels = []

def get_hist_cnt(cur_l):
    return (4**(cur_l)-1)//3


def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.
    '''

    K = opts.K
    L = opts.L
    # ----- TODO -----
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
    wordmap = visual_words.get_visual_words(g_opts, img, g_dictionary)
    return get_feature_from_wordmap_SPM(g_opts, wordmap)

def get_feat_from_resp(resp):
    global g_opts
    resp = resp.reshape(g_opts.pattern_size,g_opts.pattern_size,-1)
    wordmap = visual_words.get_visual_words_from_resp(g_opts, resp, g_dictionary)
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

    trained_feat = np.load(join(feat_dir, "hog_corner_feat.npz"))
    train_resps = trained_feat['features']
    train_labels = trained_feat['labels']
    dictionary = np.load(join(feat_dir, 'bow_dictionary.npy'))

    features = None

    global g_opts, g_dictionary
    g_opts = opts
    g_dictionary = dictionary

    print("build_recognition_system")
    with multiprocessing.Pool(n_worker) as p:
        features = np.array(p.map(get_feat_from_resp, train_resps))

    g_opts = None
    g_dictionary = None
    np.savez_compressed(join(feat_dir, 'bow_trained_system.npz'),
                        features=features,
                        labels=train_labels,
                        dictionary=dictionary,
                        SPM_layer_num=SPM_layer_num,
                        )

