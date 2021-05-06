import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.transforms import transforms

from config import cfg

import torchvision


def get_data_loader():
    path = "./data/classified/"

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    data = torchvision.datasets.ImageFolder(root=path, transform=transform)

    print("Total number of data: ", len(data))
    train_num = int(len(data)*0.7)
    test_num = len(data) - train_num

    train_data, test_data = torch.utils.data.random_split(data, [train_num, test_num])

    print("Total train of data: ", len(train_data))
    print("Total test of data: ", len(test_data))

    train_dataset_loader = DataLoader(train_data, batch_size=cfg.get('batch_size'), shuffle=True,
                                    num_workers=cfg.get('num_workers'))
    test_dataset_loader = DataLoader(test_data, batch_size=cfg.get('batch_size'), shuffle=True,
                                    num_workers=cfg.get('num_workers'))

    return train_dataset_loader, test_dataset_loader
