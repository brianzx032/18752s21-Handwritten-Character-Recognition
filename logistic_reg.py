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
opts = get_opts()


def run_model(model, trainset, testset, batch_size, learning_rate, epoch_num):
    def run_epoch(dataloader, batch_num, no_grad):
        total_loss = 0.0
        total_acc = 0.0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            if not no_grad:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item()
            pred = torch.argmax(sm(y_pred), dim=1)
            acc = torch.sum(pred == y_batch)/x_batch.shape[0]
            total_acc += acc/batch_num
        return total_acc, total_loss
    # get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(model.name, ':', device)

    # dataloader
    trainloader = DataLoader(trainset,
                             batch_size=batch_size,
                             #  num_workers=2,
                             shuffle=True)
    testloader = DataLoader(testset,
                            batch_size=batch_size,
                            # num_workers=2,
                            shuffle=False)
    # batch num
    train_batch_num = np.ceil(len(trainset)/batch_size)
    test_batch_num = np.ceil(len(testset)/batch_size)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           betas=(0.9, 0.999), eps=1e-8)

    criterion = nn.CrossEntropyLoss()
    sm = nn.Softmax(dim=1)

    # run on gpu
    criterion.to(device)
    model.to(device)
    sm.to(device)

    list_of_train_acc_per_iter = []
    list_of_test_acc_per_iter = []
    list_of_train_loss_per_iter = []
    list_of_test_loss_per_iter = []

    start_time = time_ns()
    for epoch in range(epoch_num):
        train_acc, running_loss = run_epoch(
            trainloader, train_batch_num, False)
        with torch.no_grad():
            test_acc, test_loss = run_epoch(testloader, test_batch_num, True)
        list_of_train_acc_per_iter.append(train_acc)
        list_of_test_acc_per_iter.append(test_acc)
        list_of_train_loss_per_iter.append(running_loss)
        list_of_test_loss_per_iter.append(test_loss)
        if epoch % 10 == 9:
            Avg_time = (time_ns() - start_time)//10
            Avg_second = Avg_time/1e9
            print('[%d] loss: %.3f acc: %.3f test_loss: %.3f test_acc: %.3f Avg_time: %d\'%d\'\'%dms' %
                  (epoch + 1, running_loss, train_acc, test_loss, test_acc,
                   Avg_second//60, Avg_second % 60, (Avg_time // 1e6) % 1e3))
            start_time = time_ns()

    print('Finished Training')

    PATH = './'+model.name+'.pth'
    torch.save(model.state_dict(), PATH)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(list_of_train_acc_per_iter, 'b')
    ax1.plot(list_of_test_acc_per_iter, 'r')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Accuracy')
    ax1.legend(['train', 'test'])

    ax2.plot(list_of_train_loss_per_iter, 'b')
    ax2.plot(list_of_test_loss_per_iter, 'r')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.legend(['train', 'test'])
    plt.savefig('./'+model.name+'.png')
    plt.show()


''' models '''
# 6.1.1


class FCNet(nn.Module):
    def __init__(self, name, D_in, H, D_out):
        super().__init__()
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, D_out)
        self.ch = 1
        self.name = name
        self.D_in = D_in

    def forward(self, x):
        x = x.view(-1, self.D_in)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x




class ConvNet64(nn.Module):
    def __init__(self, name, ch, cls_num):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, cls_num)
        self.ch = ch
        self.name = name
        self.patch_size = 64

    def forward(self, x):
        x = x.view(-1, self.ch, self.patch_size, self.patch_size)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# npz_data = np.load(join(opts.feat_dir, 'zm.npz'))
# fc = FCNet('fully-connected', 25, 36, 36)

# npz_data = np.load(join(opts.feat_dir, 'hog.npz'))
# fc = FCNet('fully-connected', 64*64*3, 144, 36)
# cnn = ConvNet64("cnn",3,36)

npz_data = np.load(join(opts.feat_dir, 'bow_feature.npz'))
fc = FCNet('fully-connected', 32*32*45, 32*32, 36)

dataset = TensorDataset(torch.from_numpy(npz_data["features"].astype(np.float32)),
                        torch.from_numpy(npz_data["labels"]))
test_num = len(dataset)//10
train_num = len(dataset)-test_num
trainset, testset = torch.utils.data.random_split(
    dataset, [train_num, test_num])

run_model(fc, trainset, testset, 50, 1e-1, 200)
# run_model(cnn, trainset, testset, 50, 1e-3, 100)
