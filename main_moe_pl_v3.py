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
import itertools
from tqdm import tqdm




def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)


class Metrics:
    def __init__(self, num_classes, ignore_label):
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.hist = torch.zeros(num_classes, num_classes)

    def update(self, pred, target):
        pred = pred.argmax(dim=1)
        keep = target != self.ignore_label
        self.hist += torch.bincount(target[keep] * self.num_classes + pred[keep], minlength=self.num_classes**2).view(self.num_classes, self.num_classes)

    def compute_iou(self):
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        miou = ious[~ious.isnan()].mean().item()
        return ious.cpu().numpy().tolist(), miou

    def compute_f1(self):
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        mf1 = f1[~f1.isnan()].mean().item()
        return f1.cpu().numpy().tolist(), mf1
    
    def compute_precision(self):
        precision = self.hist.diag() / self.hist.sum(0)
        mp = precision[~precision.isnan()].mean().item()
        return precision.cpu().numpy().tolist(), mp
        
    def compute_recall(self):
        recall = self.hist.diag() / self.hist.sum(1)
        mrecall = recall[~recall.isnan()].mean().item()
        return recall.cpu().numpy().tolist(), mrecall
    
    def compute_pixel_acc(self):
        acc = self.hist.diag() / self.hist.sum(1)
        macc = acc[~acc.isnan()].mean().item()
        return acc.cpu().numpy().tolist(), macc


def get_high_miou_topk_by_pivot_domain(model, dataloader, pivot_domain=0, domain_num=12, num_classes=2, top_ratio=0.5):
    model.eval()
    selected = []
    global_idx = 0

    with torch.no_grad():
        for images, targets, _ in dataloader:
            images, targets = images.cuda(), targets.cuda()
            batch_size = images.size(0)

            # pivot 도메인 예측
            pivot_idx_tensor = torch.full((batch_size,), pivot_domain, dtype=torch.long).cuda()
            pivot_output, _, _ = model(images, pivot_idx_tensor)
            pivot_preds = pivot_output.argmax(dim=1)

            # 나머지 도메인 예측
            domain_preds_list = []
            for d_idx in range(domain_num):
                if d_idx == pivot_domain:
                    continue
                domain_tensor = torch.full((batch_size,), d_idx, dtype=torch.long).cuda()
                output, _, _ = model(images, domain_tensor)
                preds = output.argmax(dim=1)
                domain_preds_list.append(preds)

            # mIoU 계산 (pivot vs each other domain)
            for i in range(batch_size):
                pivot_pred = pivot_preds[i]
                ious = []
                for preds in domain_preds_list:
                    other_pred = preds[i]
                    inter = ((pivot_pred == 1) & (other_pred == 1)).sum().item()
                    union = ((pivot_pred == 1) | (other_pred == 1)).sum().item()
                    iou = inter / union if union > 0 else 0.0
                    ious.append(iou)

                avg_miou = np.mean(ious) if ious else 0.0
                selected.append((avg_miou, global_idx))
                global_idx += 1

    # 상위 top_ratio% 인덱스 추출
    selected.sort(key=lambda x: x[0], reverse=True)
    top_k = int(len(selected) * top_ratio)
    top_indices = [s[1] for s in selected[:top_k]]
    return top_indices
    
metric = Metrics(2, 255)
    
set_seed(0)

parser = argparse.ArgumentParser(description='Deeplabv3 pytorch Training')
parser.add_argument('--test_meta', type=str, help='test metadata', default='UGA_test_metadata.csv')
parser.add_argument('--epoch', type=int, help='# of epoch', default=10)
parser.add_argument('--threshold', type=float, help='threshold', default=0.5)

args = parser.parse_args()


def get_train_augmentation(size, seg_fill):
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(degrees=10, p=0.3, seg_fill=seg_fill),
        RandomResizedCrop(size, scale=(0.5, 2.0), seg_fill=seg_fill),
    ])


def get_normalize():
    return Compose([
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])



def get_val_augmentation(size):
    return Compose([
        Resize(size),
    ])

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
    def forward(self, pred, target):
        CE = F.cross_entropy(pred, target, reduction='none', ignore_index=255)
        pt = torch.exp(-CE)
        loss = ((1 - pt) ** 2) * CE # gamma
        alpha = torch.Tensor([0.5, 0.5]) # alpha(bigger for 1(pos), MNG only)
        alpha = (target==0) * alpha[0] + (target==1) * alpha[1]
        return torch.mean(alpha * loss)


valtransform = get_val_augmentation([256, 256])
normalize = get_normalize()


testset = GPSDataset(metadata=args.test_meta, transform=valtransform, normalize=normalize)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)



model = mit_b5_MOE(patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1,  expert_num=12, select_mode='new_topk', hidden_dims = [2, 4, 10, 16], num_k = 2, domain_num=12)



model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(torch.load("./checkpoint/MOE_epoch_2_v2.pth")["state_dict"])

criterion = FocalLoss().cuda()
optimizer = torch.optim.SGD(model.module.parameters(), lr = 1e-4, momentum=0.99)  


top_indices = get_high_miou_topk_by_pivot_domain(model, testloader, pivot_domain=0,  domain_num=12, num_classes=2, top_ratio=args.threshold)
test_subset = Subset(testset, top_indices)
testloader_filtered = torch.utils.data.DataLoader(test_subset, batch_size=16, shuffle=True, num_workers=2)

print('Start Training')

for epoch in range(0, 10):
    model.train()
    for batch_idx, (images, targets, country_idx) in tqdm(enumerate(testloader_filtered)):
        images, targets, country_idx = images.cuda(), targets.cuda().detach(), country_idx.cuda().detach()
        
        output, d_output, MI_loss = model(images, country_idx)
        targets_pl = torch.argmax(output, dim=1)
    
    
        optimizer.zero_grad()
        loss =  criterion(output, targets_pl)
        loss.backward()
        optimizer.step() 

    model.eval()
    
    metrics = Metrics(2, 255)
    
    
    for batch_idx, (images, targets, country_idx) in tqdm(enumerate(testloader)):
        images, targets, country_idx = images.cuda(), targets.cuda().detach(), country_idx.cuda().detach()
        
        output, _, _ = model(images, country_idx)
        
    
        metrics.update(output.cpu(), targets.cpu())
    
        del output
    
    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    precision, mprecision = metrics.compute_precision()
    recall, mrecall = metrics.compute_recall()
    
    print(f"ious : [{ious[0]},{ious[1]}]")
    print(f"f1 : [{f1[0]},{f1[1]}]")
    print(f"acc : [{acc[0]},{acc[1]}]")
    print(f"precision : [{precision[0]},{precision[1]}]")
    print(f"recall : [{recall[0]},{recall[1]}]")