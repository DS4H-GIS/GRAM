import argparse
from dataloader import GPSDataset
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import numpy as np
import numpy
import random
from torch.utils.data import Subset
from augmentation import *
import copy 
import glob
import pandas as pd
from PIL import Image
import tifffile
from model import *
from utils import *


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

  
set_seed(0)

parser = argparse.ArgumentParser(description='Deeplabv3 pytorch Training')
parser.add_argument('--train_meta', type=str, help='training metadata', default='train_metadata.csv')
parser.add_argument('--test_meta', type=str, help='test metadata', default='UGA_test_metadata.csv')
parser.add_argument('--epoch', type=int, help='# of epoch', default=1)
args = parser.parse_args()


class DomainDiscriminator(nn.Module):
    def __init__(self, n_outputs = 12):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 32 * 32, 250)
        self.fc2 = nn.Linear(250, n_outputs)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        x = self.pool(F.relu(self.conv1(inputs)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 32 * 32 * 32)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def get_train_augmentation(size, seg_fill):
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
    ])


def get_normalize():
    return Compose([
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])



def get_val_augmentation(size):
    return Compose([
        Resize(size),
    ])




traintransform = get_train_augmentation([256, 256], 255)
valtransform = get_val_augmentation([256, 256])
normalize = get_normalize()


trainset = GPSDataset(metadata=args.train_meta, transform=traintransform, normalize=normalize)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=True, num_workers=2)


model = DomainDiscriminator()


model = torch.nn.DataParallel(model).cuda()

optimizer = torch.optim.SGD(model.module.parameters(), lr = 0.01, momentum=0.99)  
ce_criterion = nn.CrossEntropyLoss()



for epoch in range(args.epoch):
    model.train()

    for batch_idx, (images, targets, country_idx) in enumerate(trainloader):
        images, country_idx = images.cuda(), country_idx.cuda().detach()
        
        d_output = model(images)

        loss = ce_criterion(d_output, country_idx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        
        if batch_idx % 100 == 0:
            print(f"[Epoch {epoch} | Batch {batch_idx}] Loss: {loss.item():.4f} |")

    # # === epoch 끝날 때마다 저장 === #
    checkpoint_path = os.path.join("./checkpoint", f"DC_epoch_{epoch}_v1.pth")
    torch.save({
        'state_dict': model.state_dict(),
    }, checkpoint_path)
