from os.path import join
from time import time_ns

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import bag_of_words
import util
from opts import get_opts

opts = get_opts()


def train_model(model, trainset, validset, param):
    batch_size, learning_rate, epoch_num, weight_decay, alpha, pattern_sz, threshold, n = param

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
        '_lr{:.2}w{:.2}a{}ps{}n{}thrs{:.2}'.format(
            learning_rate, weight_decay, alpha, pattern_sz, n, threshold)
    print(param_str, ':', device)

    # dataloader
    trainloader = DataLoader(trainset,
                             batch_size=batch_size,
                             shuffle=True)
    validloader = DataLoader(validset,
                             batch_size=batch_size,
                             shuffle=False)
    # batch num
    train_batch_num = np.ceil(len(trainset)/batch_size)
    valid_batch_num = np.ceil(len(validset)/batch_size)

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

    PATH = join(opts.out_dir, param_str+'.pth')
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
    plt.savefig(join(opts.out_dir, param_str+'.png'))
    # plt.show()
    plt.close()
    return valid_acc


def test_model(model, testset, param):
    batch_size, learning_rate, epoch_num, weight_decay, alpha, pattern_sz, threshold, n = param
    # get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    param_str = model.feat_name + \
        '_lr{:.2}w{:.2}a{}ps{}n{}thrs{:.2}'.format(
            learning_rate, weight_decay, alpha, pattern_sz, n, threshold)
    print(param_str, ':', device)

    # dataloader
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # batch num
    batch_num = np.ceil(len(testset)/batch_size)

    sm = nn.Softmax(dim=1)

    # run on gpu
    model.to(device)
    sm.to(device)

    confusion = np.zeros((36, 36))
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            y_pred = torch.argmax(sm(model(x)), dim=1)
            for t, p in zip(y, y_pred):
                confusion[t, p] += 1

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


def tuning():
    best_param = [0 for i in range(8)]
    best_val_acc = 0
    best_model = None
    testset_for_best = None
    for alpha in range(15, 22, 3):
        opts.alpha = alpha
        for pattern_size in range(11, 13, 1):
            opts.pattern_size = pattern_size
            for threshold in np.arange(0.14, 0.18, 0.02):
                opts.thres = threshold
                for n in range(8, 13, 2):
                    opts.hog_n = n

                    bag_of_words.main(["extract"], opts)

                    npz_data = np.load(
                        join(opts.feat_dir, 'hog_corner_feat.npz'))
                    logreg = LR("hog_corner", opts.pattern_size *
                                opts.pattern_size*opts.alpha, 36)
                    dataset = TensorDataset(torch.from_numpy(npz_data["features"].astype(np.float32)),
                                            torch.from_numpy(npz_data["labels"]))
                    valid_num = len(dataset)//3
                    test_num = valid_num
                    train_num = len(dataset)-valid_num-test_num
                    trainset, validset, testset = torch.utils.data.random_split(
                        dataset, [train_num, valid_num, test_num])

                    for lr in np.arange(1.5e-3, 2.5e-3, 5e-4):
                        for w in np.arange(1.5e-3, 2.5e-3, 5e-4):
                            valid_acc = train_model(logreg, trainset, validset, [
                                                    opts.batch_size, lr, opts.epoch, w, alpha, pattern_size, threshold, n])
                            if valid_acc > best_val_acc:
                                best_param = [opts.batch_size, lr, opts.epoch,
                                              w, alpha, pattern_size, threshold, n]
                                best_val_acc = valid_acc
                                best_model = logreg
                                testset_for_best = testset
                    print("current best:", best_param,
                          "acc:", float(best_val_acc))
    print(best_param)
    confusion = test_model(best_model, testset_for_best, best_param)
    accuracy = np.sum(confusion.diagonal()) / np.sum(confusion)
    print("valid_acc:{:.3f}; test_acc:{:.3f}".format(best_val_acc, accuracy))
    return confusion, best_param


'''hog'''
confusion, best_param = tuning()
util.visualize_confusion_matrix(confusion)
_, _, _, _, opts.alpha, opts.pattern_size, opts.hog_thres, opts.hog_n = best_param
best_param = [opts.batch_size, opts.lr, opts.epoch, opts.weight_decay,
              opts.alpha, opts.pattern_size, opts.hog_thres, opts.hog_n]
best_param[2] = 1000
bag_of_words.main(["extract"], opts)
npz_data = np.load(join(opts.feat_dir, 'hog_corner_feat.npz'))
logreg = LR("best_hog_corner", opts.pattern_size *
            opts.pattern_size*opts.alpha, 36)

dataset = TensorDataset(torch.from_numpy(npz_data["features"].astype(np.float32)),
                        torch.from_numpy(npz_data["labels"]))
test_num = len(dataset)//5
train_num = len(dataset)-test_num
trainset, testset = torch.utils.data.random_split(
    dataset, [train_num, test_num])
train_model(logreg, trainset, testset, best_param)
confusion = test_model(logreg, testset, best_param)
accuracy = np.sum(confusion.diagonal()) / np.sum(confusion)
print("test_acc:{:.3f}".format(accuracy))
util.visualize_confusion_matrix(confusion)
