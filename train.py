import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

from data import get_data_loader
from network import Network3
from config import cfg
from sklearn.metrics import roc_auc_score
try:
    from termcolor import cprint
except ImportError:
    cprint = None
#import torchvision.models as models



SEED = 11785
np.random.seed(SEED)
torch.manual_seed(SEED)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU ID
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)

def get_lr(optimizer):
    #TODO: Returns the current Learning Rate being used by
    # the optimizer
    for param_group in optimizer.param_groups:
        return (param_group['lr'])
'''
Use the average meter to keep track of average of the loss or 
the test accuracy! Just call the update function, providing the
quantities being added, and the counts being added
'''
class AvgMeter():
    def __init__(self):
        self.qty = 0
        self.cnt = 0
    
    def update(self, increment, count):
        self.qty += increment
        self.cnt += count
    
    def get_avg(self):
        if self.cnt == 0:
            return 0
        else: 
            return self.qty/self.cnt


def run(net, epoch, loader, optimizer, criterion, logger, scheduler, train=True):
    # TODO: Performs a pass over data in the provided loader
    total_loss = 0.0
    # TODO: Initalize the different Avg Meters for tracking loss and accuracy (if test)
    num_elements = len(loader.dataset)
    num_batches = len(loader)
    batch_size = loader.batch_size
    pred_all = torch.zeros(num_elements)
    label_all = torch.zeros(num_elements)
    print(train, batch_size)

    # TODO: Iterate over the loader and find the loss. Calculate the loss and based on which
    # set is being provided update you model. Also keep track of the accuracy if we are running
    # on the test set.
    if train:
        net.train()
    else:
        net.eval()
    count = 1
    fraction = 5
    for i, (data, label) in enumerate(loader):
        # print(data.shape)
        # print(label)
        if i > (count*num_batches/fraction):
            print("Batch Finished: " + str(i) + "/" + str(num_batches))
            count += 1
        data = data.to(device)
        label = label.to(device)
        #label = label.long()
        #output = net.forward(data.float())
        output = net.forward(data)
        loss = criterion(output, label)
        pred = torch.argmax(output, dim=1)

        start = i * batch_size
        end = start + batch_size
        if i == num_batches - 1:
            end = num_elements
        pred_all[start:end] = pred
        label_all[start:end] = label

        total_loss += loss.item()
        if(train):
            net.zero_grad()
            loss.backward()
            optimizer.step()
    pred_all = np.asarray(pred_all)
    label_all = np.asarray(label_all)
    # if not train:
    #     print(pred_all)
    #     print(label_all)
    accuracy = np.mean(label_all == pred_all)
    # TODO: Log the training/testing loss using tensorboard.
    if(train):
        logger.add_scalar(tag='train_loss', scalar_value=total_loss, global_step=epoch)
    else:
        logger.add_scalar(tag='test_loss', scalar_value=total_loss, global_step=epoch)

    # TODO: return the average loss, and the accuracy (if test set)
    return total_loss, accuracy
    #raise NotImplementedError
        

def train(net, train_loader, test_loader, logger):    
    # TODO: Define the SGD optimizer here. Use hyper-parameters from cfg
    # optimizer = optim.SGD(net.parameters(), lr=cfg.get('lr'), momentum=cfg.get('momentum'), weight_decay=cfg.get('weight_decay'),
    #                       nesterov=cfg.get('nesterov'))
    optimizer = optim.SGD(net.parameters(), momentum=cfg.get('momentum'), weight_decay=cfg.get('weight_decay'), lr=cfg.get('lr'))
    # TODO: Define the criterion (Objective Function) that you will be using
    criterion = nn.CrossEntropyLoss()
    # TODO: Define the ReduceLROnPlateau scheduler for annealing the learning rate
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=cfg.get('patience'), factor=cfg.get('lr_decay'))
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=cfg.get('step_size'), gamma=cfg.get('lr_decay'))
    net.to(device)

    for i in range(cfg['epochs']):
        print("Current Learning Rate: " + str(get_lr(optimizer)))
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        # Run the network on the entire train dataset. Return the average train loss
        # Note that we don't have to calculate the accuracy on the train set.
        loss, _ = run(net, i, train_loader, optimizer, criterion, logger, scheduler)

        # TODO: Get the current learning rate by calling get_lr() and log it to tensorboard
        logger.add_scalar(tag='learning_rate', scalar_value=get_lr(optimizer), global_step=i)

        # Logs the training loss on the screen, while training
        if i % cfg['log_every'] == 0:
            log_text = "Epoch: [%d/%d], Training Loss:%2f" % (i, cfg['epochs'], loss)
            log_print(log_text, color='green', attrs=['bold'])

        # Evaluate our model and add visualizations on tensorboard
        if i % cfg['val_every'] == 0:
            # TODO: HINT - you might need to perform some step before and after running the network
            # on the test set
            # Run the network on the test set, and get the loss and accuracy on the test set 
            loss, acc = run(net, i, test_loader, optimizer, criterion, logger, scheduler, train=False)
            log_text = "Epoch: %d, Test Accuracy:%2f" % (i, acc*100.0)
            log_print(log_text, color='red', attrs=['bold'])

            # TODO: Perform a step on the scheduler, while using the Accuracy on the test set

            scheduler.step()
            # TODO: Use tensorboard to log the Test Accuracy and also to perform visualization of the 
            # 2 weights of the first layer of the network!
            #logger.add_histogram(tag='hidden1_weight', values=net.linear1.weight[:,0].reshape(-1), global_step=i)
            logger.add_scalar(tag='accuracy', scalar_value=acc, global_step=i)
        end_time.record()
        torch.cuda.synchronize()
        time_elapsed = start_time.elapsed_time(end_time)
        print("Time Used: " + str(time_elapsed))

        # print("Model's state_dict:")
        # for param_tensor in net.state_dict():
        #     print(param_tensor, "\t", net.state_dict()[param_tensor].size())
        torch.save(net.state_dict(), (str(i) + 'epoch.ckpt'))
    logger.close()

if __name__ == '__main__':

    # TODO: Create a network object
    net = Network3()
    #net = models.resnet152(pretrained=True)
    #net.fc = nn.Linear(2048, 4000, bias=False)
    # print(net)
    # TODO: Create a tensorboard object for logging
    writer = SummaryWriter()
    # command to check: tensorboard --logdir=runs

    train_loader, test_loader = get_data_loader()

    #Run the training!
    train(net, train_loader, test_loader, writer)
    # torch.save(net.state_dict(), 'result.ckpt')