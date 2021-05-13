from collections import OrderedDict

import torch
import torch.nn as nn
from config import cfg


class Network3(nn.Module):
    def __init__(self):
        super(Network3, self).__init__()
        # TODO: Define the model architecture here
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(8),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2)
                                    )
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(8),
                                    nn.ReLU()
                                    )
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2)
                                    )
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.Dropout2d(p=0.15)
                                    )
        self.layer5 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2)
                                    )
        self.layer6 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2)
                                    )
        self.layer7 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Dropout2d(p=0.2)
                                    )
        # self.layer8 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
        #                             nn.BatchNorm2d(256),
        #                             nn.ReLU(),
        #                             nn.MaxPool2d(kernel_size=2, stride=2)
        #                             )
        # self.layer9 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
        #                             nn.BatchNorm2d(256),
        #                             nn.ReLU(),
        #                             nn.MaxPool2d(kernel_size=2, stride=2)
        #                             )
        # self.layer10 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
        #                              nn.BatchNorm2d(512),
        #                              nn.ReLU(),
        #                              nn.Dropout2d(p=0.3)
        #                              )
        # self.layer11 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
        #                              nn.BatchNorm2d(512),
        #                              nn.ReLU(),
        #                              nn.MaxPool2d(kernel_size=2, stride=2)
        #                              )
        # self.layer12 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
        #                              nn.BatchNorm2d(512),
        #                              nn.ReLU(),
        #                              nn.MaxPool2d(kernel_size=2, stride=2)
        #                              )
        # self.layer13 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
        #                              nn.BatchNorm2d(1024),
        #                              nn.ReLU(),
        #                              nn.Dropout2d(p=0.3)
        #                              )
        # self.layer14 = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
        #                              nn.BatchNorm2d(1024),
        #                              nn.ReLU(),
        #                              nn.MaxPool2d(kernel_size=2, stride=2)
        #                              )
        # self.layer15 = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
        #                              nn.BatchNorm2d(1024),
        #                              nn.ReLU()
        #                              )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(256, 36, bias=False)
        self.init_weights()

    def init_weights(self):
        # NOTE: Not for Vanilla Classsifier
        # TODO: Initalize weights by calling by using the
        # appropriate initialization function
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, x):
        # TODO: Define the forward function of your model
        # print(x.shape)
        x = self.layer1(x)
        r = x
        # print('layer 1')
        # print(x.shape)
        x = self.layer2(x)
        # print('layer 2')
        # print(x.shape)
        # x = r+x
        x = self.layer3(x)
        # print('layer 3')
        # print(x.shape)
        x = self.layer4(x)
        # print('layer 4')
        # print(x.shape)
        r = x
        x = self.layer5(x)
        # print('layer 5')
        # print(x.shape)
        x = self.layer6(x)
        # print('layer 6')
        # print(x.shape)
        # x = r + x
        x = self.layer7(x)
        # print('layer 7')
        # print(x.shape)
        # r = x
        # x = self.layer8(x)
        # # print('layer 8')
        # # print(x.shape)
        # x = self.layer9(x)
        # # print('layer 9')
        # # print(x.shape)
        # r = self.maxpool(r)
        # r = self.maxpool(r)
        # x = x+r
        # x = self.layer10(x)
        # r=x
        # # print('layer 10')
        # # print(x.shape)
        # x = self.layer11(x)
        # # print('layer 11')
        # # print(x.shape)
        # x = self.layer12(x)
        # # print('layer 12')
        # # print(x.shape)
        # r = self.maxpool(r)
        # r = self.maxpool(r)
        # x = x + r
        # x = self.layer13(x)
        # # print('layer 13')
        # # print(x.shape)
        # r = x
        # x = self.layer14(x)
        # # print('layer 14')
        # # print(x.shape)
        # r = self.maxpool(r)
        # x = x + r
        # x = self.layer15(x)
        x = x.reshape(-1, 128*2)
        # print(x.shape)
        x = self.linear(x)
        # print(x.shape)
        return x
        # raise NotImplementedError

    def save(self, ckpt_path):
        # TODO: Save the checkpoint of the model
        torch.save(self.state_dict(), ckpt_path)
        # raise NotImplementedError

    def load(self, ckpt_path):
        # TODO: Load the checkpoint of the model
        torch.load(ckpt_path)
        # raise NotImplementedError


class Network2(nn.Module):
    def __init__(self):
        super(Network2, self).__init__()
        # TODO: Define the model architecture here
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU()
                                    )
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU()
                                    )
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2)
                                    )
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.Dropout2d(p=0.3)
                                    )
        self.layer5 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU()
                                    )
        self.layer6 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU()
                                    )
        self.layer7 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    nn.Dropout2d(p=0.3)
                                    )
        self.layer8 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2)
                                    )
        self.layer9 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2)
                                    )
        self.layer10 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(),
                                     nn.Dropout2d(p=0.3)
                                     )
        self.layer11 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, stride=2)
                                     )
        self.layer12 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, stride=2)
                                     )
        self.layer13 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(1024),
                                     nn.ReLU(),
                                     nn.Dropout2d(p=0.3)
                                     )
        self.layer14 = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(1024),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, stride=2)
                                     )
        self.linear = nn.Linear(1024, 4000, bias=False)
        self.init_weights()

    def init_weights(self):
        # NOTE: Not for Vanilla Classsifier
        # TODO: Initalize weights by calling by using the
        # appropriate initialization function
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('linear'))
        print(self.layer1.parameters())

    def forward(self, x):
        # TODO: Define the forward function of your model
        # print(x.shape)
        x = self.layer1(x)
        r = x
        # print('layer 1')
        # print(x.shape)
        x = self.layer2(x)
        # print('layer 2')
        # print(x.shape)
        x = r+x
        x = self.layer3(x)
        # print('layer 3')
        # print(x.shape)
        x = self.layer4(x)
        # print('layer 4')
        # print(x.shape)
        r = x
        x = self.layer5(x)
        # print('layer 5')
        # print(x.shape)
        x = self.layer6(x)
        # print('layer 6')
        # print(x.shape)
        x = r+x
        x = self.layer7(x)
        # print('layer 7')
        # print(x.shape)
        x = self.layer8(x)
        # print('layer 8')
        # print(x.shape)
        x = self.layer9(x)
        # print('layer 9')
        # print(x.shape)
        x = self.layer10(x)
        # print('layer 10')
        # print(x.shape)
        x = self.layer11(x)
        # print('layer 11')
        # print(x.shape)
        x = self.layer12(x)
        # print('layer 12')
        # print(x.shape)
        x = self.layer13(x)
        # print('layer 13')
        # print(x.shape)
        x = self.layer14(x)
        # print('layer 14')
        # print(x.shape)
        x = x.reshape(-1, 1024)
        x = self.linear(x)
        return x
        # raise NotImplementedError

    def save(self, ckpt_path):
        # TODO: Save the checkpoint of the model
        torch.save(self.state_dict(), ckpt_path)
        # raise NotImplementedError

    def load(self, ckpt_path):
        # TODO: Load the checkpoint of the model
        torch.load(ckpt_path)
        # raise NotImplementedError