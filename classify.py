from os.path import join

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from torch.utils.data import DataLoader, TensorDataset
import util

from opts import get_opts

opts = get_opts()

''' extracted features '''
npz_data = np.load(join(opts.feat_dir, 'hog_stack.npz'))  # LR 92.53%
# npz_data = np.load(join(opts.feat_dir, 'hog_corner_feat.npz'))
# npz_data = np.load(join(opts.feat_dir, 'zernike.npz')) # LR 51.46%
# npz_data = np.load(join(opts.feat_dir, 'autoencoder.npz')) # LDA 63.84%
# npz_data = np.load(join(opts.feat_dir, 'bow_trained_system.npz')) # LDA 43.42%

X, y = npz_data["features"], npz_data["labels"]
try:
    # for hog_corner_feat.npz
    X = X.reshape((-1, opts.pattern_size**2*opts.alpha))
except:
    pass
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

''' resized images '''  # LR 85.64%
# dataset = torchvision.datasets.ImageFolder(
#     "./data/classified/", transform=util.transform(64))
# test_num = len(dataset)//3
# train_num = len(dataset)-test_num
# trainset, testset = torch.utils.data.random_split(
#     dataset, [train_num, test_num])
# trainloader = DataLoader(trainset,train_num,shuffle=True)
# testloader = DataLoader(testset,test_num,shuffle=False)
# X_train, X_test, y_train, y_test = None,None,None,None
# for x,y in trainloader:
#     X_train,y_train=x.reshape(train_num,-1),y
# for x,y in testloader:
#     X_test,y_test=x.reshape(test_num,-1),y
# X = np.vstack((X_train,X_test))

print("Total Data:",X.shape[0])
print("Train Data:",X_train.shape[0])
print("Test Data:",X_test.shape[0])

''' model '''
# model = GaussianNB()
# model = LinearDiscriminantAnalysis()
model = LogisticRegression(max_iter=500, penalty='l2', C=100)

''' train '''
y_pred = model.fit(X_train, y_train).predict(X_test)

''' visualize '''
confusion_matrix = np.zeros((36, 36))
for t, p in zip(y_test, y_pred):
    confusion_matrix[t, p] += 1
accuracy = np.sum(confusion_matrix.diagonal()) / np.sum(confusion_matrix)
print("Test acc: {:.2f}%".format(accuracy*100))
util.visualize_confusion_matrix(confusion_matrix)
