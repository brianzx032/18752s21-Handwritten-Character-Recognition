import os
import torch
import numpy as np
import torch.nn as nn
from data import get_data_loader
from network import Network3
from config import cfg
try:
    from termcolor import cprint
except ImportError:
    cprint = None
#import torchvision.models as models

#TODO: PLEASE CHANGE THIS TO THE EPOCH NUMBER YOU WANT TO RUN
epoch = 29
gen_csv = True

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU ID
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = get_data_loader('test')
    model = Network3()
    #model = models.resnet152(pretrained=True)
    #model.fc = nn.Linear(2048, 4000, bias=False)
    print(model)
    model.to(device)

    model.load_state_dict(torch.load(str(epoch)+'epoch.ckpt'))
    model.eval()
    pred_list = []
    label_list = []
    num_elements = len(loader.dataset)
    num_batches = len(loader)
    for i, (data,label) in enumerate(loader):
        if i%10==0:
            print("Batch Finished: " + str(i) + "/" + str(num_batches))
        data = data.to(device)
        output = model.forward(data)
        pred = torch.argmax(output, axis=1)
        pred_list.append(pred)
        label_list.append(label)
    pred = torch.cat(pred_list)
    pred = pred.cpu().numpy()
    label = torch.cat(label_list)
    label = label.cpu().numpy()
    print(np.average(pred==label))
    model.fc = Identity()
    v_acc, auc = verification(model, gen_csv=gen_csv)
    print('Verification accuracy: ' + str(v_acc))
    print('Auc Score: ' + str(auc))
    #np.savetxt("prediction.csv", np.dstack((np.arange(len(pred)),pred))[0],"%d,%d",header="ID,Category", comments="")
