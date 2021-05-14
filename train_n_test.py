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
