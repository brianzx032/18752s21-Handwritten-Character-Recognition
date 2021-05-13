from os.path import join

import numpy as np
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from torch.utils.data import DataLoader

import util
from opts import get_opts

opts = get_opts()

''' extracted features '''
X,y = util.load_features(opts)
# # hog stacked
# if opts.feature == "hs":
#     if opts.re_extract:
#         extract_hog_stack()
#     npz_data = np.load(join(opts.feat_dir, 'hog_stack.npz'))  # LR 92.53%

# # hog corner
# if opts.feature == "hc":
#     if opts.re_extract:
#         extract(["hog_corner"], opts)
#     npz_data = np.load(join(opts.feat_dir, 'hog_corner.npz')) # LR 73.92%

# # bag of visual words
# if opts.feature == "bow":
#     if opts.re_extract:
#         extract(["bow_feat"], opts)
#     npz_data = np.load(join(opts.feat_dir, 'bow_trained_system.npz')) # LDA 59.11%

# # zernike
# if opts.feature == "z":
#     if opts.re_extract:
#         extract_zernike()
#     npz_data = np.load(join(opts.feat_dir, 'zernike.npz')) # QDA 62.48%

# # autoencoder
# if opts.feature == "ae":
#     if opts.re_extract:
#         extract_encoded()
#     npz_data = np.load(join(opts.feat_dir, 'autoencoder.npz')) # LDA 63.84%

# try:
#     X, y = npz_data["features"], npz_data["labels"]
#     if opts.feature == "hc":
#         X = X.reshape((-1, opts.pattern_size**2*opts.alpha))
# except:
#     # resized images  
#     if opts.feature == "orig":
#         X, y =util.get_image_data(64) # LR 85.64%


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

print("Total Data:",X.shape[0])
print("Train Data:",X_train.shape[0])
print("Test Data:",X_test.shape[0])

''' model '''
if opts.classifier == "gnb":
    model = GaussianNB()
if opts.classifier == "lda":
    model = LinearDiscriminantAnalysis()
if opts.classifier == "qda":
    model = QuadraticDiscriminantAnalysis()
if opts.classifier == "lr":
    model = LogisticRegression(max_iter=1000, penalty='l2', C=100)

''' train and test'''
y_pred = model.fit(X_train, y_train).predict(X_test)

''' visualize '''
confusion_matrix = np.zeros((36, 36))
for t, p in zip(y_test, y_pred):
    confusion_matrix[t, p] += 1
accuracy = np.sum(confusion_matrix.diagonal()) / np.sum(confusion_matrix)
print("Test acc: {:.2f}%".format(accuracy*100))
util.visualize_confusion_matrix(confusion_matrix)
