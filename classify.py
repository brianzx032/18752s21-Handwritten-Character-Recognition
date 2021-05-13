from os.path import join

import numpy as np
import torch
import torchvision
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from torch.utils.data import DataLoader, TensorDataset
import util

from opts import get_opts

opts = get_opts()

''' extracted features '''
# npz_data = np.load(join(opts.feat_dir, 'hog_stack.npz'))  # LR 92.53%
# npz_data = np.load(join(opts.feat_dir, 'hog_corner_feat.npz')) # LR 73.92%
# npz_data = np.load(join(opts.feat_dir, 'zernike.npz')) # QDA 62.48%
# npz_data = np.load(join(opts.feat_dir, 'autoencoder.npz')) # LDA 63.84%
# npz_data = np.load(join(opts.feat_dir, 'bow_trained_system.npz')) # LDA 59.11%

X, y = npz_data["features"], npz_data["labels"]
try:
    # for hog_corner_feat.npz
    X = X.reshape((-1, opts.pattern_size**2*opts.alpha))
except:
    pass

''' resized images '''  # LR 85.64%
# X, y =util.get_image_data(64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

print("Total Data:",X.shape[0])
print("Train Data:",X_train.shape[0])
print("Test Data:",X_test.shape[0])

''' model '''
# model = GaussianNB()
# model = LinearDiscriminantAnalysis()
# model = QuadraticDiscriminantAnalysis()
model = LogisticRegression(max_iter=1000, penalty='l2', C=100)

''' train '''
y_pred = model.fit(X_train, y_train).predict(X_test)

''' visualize '''
confusion_matrix = np.zeros((36, 36))
for t, p in zip(y_test, y_pred):
    confusion_matrix[t, p] += 1
accuracy = np.sum(confusion_matrix.diagonal()) / np.sum(confusion_matrix)
print("Test acc: {:.2f}%".format(accuracy*100))
util.visualize_confusion_matrix(confusion_matrix)
