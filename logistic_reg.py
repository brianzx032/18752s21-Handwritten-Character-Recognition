from os.path import join
from time import time_ns

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import skimage.color
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from skimage.transform import resize
from torch.utils.data import DataLoader, TensorDataset
from opts import get_opts
import string
opts = get_opts()


def train_model(model, trainset, validset, batch_size, learning_rate, epoch_num, weight_decay):
    def run_epoch(dataloader, batch_num, no_grad):
        total_loss = 0.0
        total_acc = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            if not no_grad:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item()
            pred = torch.argmax(sm(y_pred), dim=1)
            acc = torch.sum(pred == y)/x.shape[0]
            total_acc += acc/batch_num
        return total_acc, total_loss
    # get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    param_str = model.feat_name + \
        '_b{}lr{:.1}w{:.1}e{}'.format(
            batch_size, learning_rate, weight_decay, epoch_num)
    print(param_str, ':', device)

    # dataloader
    trainloader = DataLoader(trainset,
                             batch_size=batch_size,
                             #  num_workers=2,
                             shuffle=True)
    validloader = DataLoader(validset,
                             batch_size=batch_size,
                             # num_workers=2,
                             shuffle=False)
    # batch num
    train_batch_num = np.ceil(len(trainset)/batch_size)
    valid_batch_num = np.ceil(len(validset)/batch_size)

    # optimizer = optim.SGD(model.parameters(), lr=learning_rate*5, weight_decay=1e-5)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()
    sm = nn.Softmax(dim=1)

    # run on gpu
    criterion.to(device)
    model.to(device)
    sm.to(device)

    list_of_train_acc_per_iter = []
    list_of_valid_acc_per_iter = []
    list_of_train_loss_per_iter = []
    list_of_valid_loss_per_iter = []

    start_time = time_ns()
    for epoch in range(epoch_num):
        train_acc, running_loss = run_epoch(
            trainloader, train_batch_num, False)
        with torch.no_grad():
            valid_acc, valid_loss = run_epoch(
                validloader, valid_batch_num, True)
        list_of_train_acc_per_iter.append(train_acc)
        list_of_valid_acc_per_iter.append(valid_acc)
        list_of_train_loss_per_iter.append(running_loss)
        list_of_valid_loss_per_iter.append(valid_loss)
        if epoch % 20 == 19:
            Avg_time = (time_ns() - start_time)//20
            Avg_second = Avg_time/1e9
            print('[%d] loss: %.3f acc: %.3f valid_loss: %.3f valid_acc: %.3f Avg_time: %d\'%d\'\'%dms' %
                  (epoch + 1, running_loss, train_acc, valid_loss, valid_acc,
                   Avg_second//60, Avg_second % 60, (Avg_time // 1e6) % 1e3))
            start_time = time_ns()

    print('Finished Training')

    PATH = join(opts.out_dir,param_str+'.pth')
    torch.save(model.state_dict(), PATH)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(list_of_train_acc_per_iter, 'b')
    ax1.plot(list_of_valid_acc_per_iter, 'r')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Accuracy')
    ax1.legend(['train', 'valid'])

    ax2.plot(list_of_train_loss_per_iter, 'b')
    ax2.plot(list_of_valid_loss_per_iter, 'r')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.legend(['train', 'valid'])
    plt.savefig(join(opts.out_dir,param_str+'.png'))
    # plt.show()
    return valid_acc


def test_model(model, testset, param):
    batch_size, learning_rate, epoch_num, weight_decay = param
    # get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    param_str = model.feat_name + \
        '_b{}lr{:.1}w{:.1}e{}_test'.format(
            batch_size, learning_rate, weight_decay, epoch_num)
    print(param_str, ':', device)

    # dataloader
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # batch num
    batch_num = np.ceil(len(testset)/batch_size)

    sm = nn.Softmax(dim=1)

    # run on gpu
    model.to(device)
    sm.to(device)

    confusion = np.zeros((36,36))
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            y_pred = torch.argmax(sm(model(x)), dim=1)
            for t,p in zip(y,y_pred):
                confusion[t,p] += 1

    # plt.show()
    return confusion


''' models '''


class LR(nn.Module):
    def __init__(self, feat_name, D_in, D_out):
        super().__init__()
        self.fc1 = nn.Linear(D_in, D_out)
        self.D_in = D_in
        self.feat_name = feat_name

    def forward(self, x):
        x = x.view(-1, self.D_in)
        x = self.fc1(x)
        return x

# npz_data = np.load(join(opts.feat_dir, 'zm.npz'))
# logreg = LR('logistic-regression', 25, 36)

# npz_data = np.load(join(opts.feat_dir, 'hog.npz'))
# logreg = LR('logistic-regression', 64*64*3, 36)

npz_data = np.load(join(opts.feat_dir, 'bow_feature.npz'))
logreg = LR("bow", 32*32*20, 36)

dataset = TensorDataset(torch.from_numpy(npz_data["features"].astype(np.float32)),
                        torch.from_numpy(npz_data["labels"]))
valid_num = len(dataset)//5
test_num = valid_num
train_num = len(dataset)-valid_num-valid_num
trainset, validset, testset = torch.utils.data.random_split(
    dataset, [train_num, valid_num, test_num])

best_param = np.zeros(4)
best_val_acc = 0
for bs in [1500]:
    for lr in np.arange(5e-5, 1e-3, 5e-5):
        for e in [100]:
            for w in np.arange(5e-5, 1e-3, 5e-5):
                valid_acc = train_model(
                    logreg, trainset, validset, bs, lr, e, w)
                if valid_acc > best_val_acc:
                    best_param = np.array([bs, lr, e, w])
print(best_param)
confusion = test_model(logreg,testset,best_param)
accuracy = np.sum(confusion.diagonal()) / np.sum(confusion)
print(accuracy)
plt.imshow(confusion,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),''.join([str(_) for _ in range(10)])+string.ascii_uppercase[:26])
plt.yticks(np.arange(36),''.join([str(_) for _ in range(10)])+string.ascii_uppercase[:26])
plt.show()