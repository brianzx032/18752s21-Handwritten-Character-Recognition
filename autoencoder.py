from os.path import join
from time import time_ns

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

import visual_words
from opts import get_opts


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # encoder
        self.encoder1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(8),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2)
                                      )
        self.encoder2 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(8),
                                      nn.ReLU()
                                      )
        self.encoder3 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(16),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2)
                                      )
        self.encoder4 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(16),
                                      nn.ReLU(),
                                      nn.Dropout2d(p=0.15)
                                      )
        self.encoder5 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2)
                                      )
        self.encoder6 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2)
                                      )
        self.encoder7 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Dropout2d(p=0.2)
                                      )

        # decoder
        def fc(in_size, out_size):
            return nn.Sequential(nn.Linear(in_size, out_size),
                                 nn.ReLU())
        self.decoder1 = fc(64*2*2, 32*4*4)
        self.decoder2 = fc(32*4*4, 32*4*4)
        self.decoder3 = fc(32*4*4, 16*8*8)
        self.decoder4 = fc(16*8*8, 16*8*8)
        self.decoder5 = fc(16*8*8, 8*16*16)
        self.decoder6 = fc(8*16*16, 8*16*16)
        self.decoder7 = fc(8*16*16, 3*64*64)

    def encoder(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)
        x = self.encoder5(x)
        x = self.encoder6(x)
        x = self.encoder7(x)
        return x

    def decoder(self, x):
        x = x.view(-1, 64*2*2)
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)
        x = self.decoder5(x)
        x = self.decoder6(x)
        x = self.decoder7(x)
        x = x.view(-1, 3, 64, 64)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def save(self, ckpt_path):
        torch.save(self.state_dict(), ckpt_path)

    def load(self, ckpt_path):
        torch.load(ckpt_path)


opts = get_opts()
visual_words.set_opts(opts)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def train_encoder():
    learning_rate = 3e-5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Autoencoder()
    model.to(device)
    loss_fn = torch.nn.MSELoss()
    loss_fn.to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-3)
    dataset = torchvision.datasets.ImageFolder(
        "./data/classified/", transform=transform)
    dataloader = DataLoader(dataset, batch_size=1500, shuffle=True)
    start_time = time_ns()
    for epoch in range(200):
        total_loss = 0
        for x, _ in dataloader:
            x = x.to(device)
            im_out = model(x)
            loss = loss_fn(im_out, x)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        if epoch % 2 == 0:
            print("itr: {:02d} \t loss: {:.2f}".format(epoch, total_loss))
    PATH = join(opts.out_dir, 'autoencoder.pth')
    torch.save(model.state_dict(), PATH)

    features = None
    dataloader = DataLoader(dataset, batch_size=500, shuffle=False)
    for x, _ in dataloader:
        x = x.to(device)
        feat = model.encoder(x).cpu().numpy()
        try:
            features = np.vstack((features, feat.reshape(x.shape[0], -1)))
        except:
            features = feat.reshape(x.shape[0], -1)
    np.savez_compressed(join(opts.feat_dir, 'autoencoder.npz'),
                        features=features,
                        labels=dataset.targets,
                        )


if __name__ == "__main__":
    train_encoder()
