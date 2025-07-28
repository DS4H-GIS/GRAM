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

def get_all_checkpoint_preds(checkpoints, dataloader):
    all_preds = []

    print("Running inference for all checkpoints...")

    for i, model in enumerate(checkpoints):
        preds = []
        model.eval()
        with torch.no_grad():
            for images, _, country_idx in tqdm(dataloader):
                images, country_idx = images.cuda(), country_idx.cuda().detach()
                output, d_output, MI_loss  = model(images, country_idx)
                pred = output.argmax(dim=1).cpu()
                preds.extend(pred)
        all_preds.append(preds)

    return all_preds

def get_high_miou_topk_by_checkpoint(checkpoints, dataloader, top_ratio=0.5):
    all_preds = get_all_checkpoint_preds(checkpoints, dataloader)
    num_samples = len(all_preds[0])
    num_ckpt = len(all_preds)
    print('Finished inference')
    sample_mious = []

    for idx in range(num_samples):
        preds_per_ckpt = [all_preds[k][idx] for k in range(num_ckpt)]

        ious = []
        for a, b in itertools.combinations(preds_per_ckpt, 2):
            inter = ((a == 1) & (b == 1)).sum().item()
            union = ((a == 1) | (b == 1)).sum().item()
            iou = inter / union if union > 0 else 0.0
            ious.append(iou)

        avg_miou = np.mean(ious) if ious else 0.0
        sample_mious.append((avg_miou, idx))

    # Top-k selection by mIoU
    sample_mious.sort(key=lambda x: x[0], reverse=True)
    top_k = int(len(sample_mious) * top_ratio)
    top_indices = [idx for _, idx in sample_mious[:top_k]]

    return top_indices
            
metric = Metrics(2, 255)
    
set_seed(0)

parser = argparse.ArgumentParser(description='Deeplabv3 pytorch Training')
parser.add_argument('--test_meta', type=str, help='test metadata', default='./metadata/UGA_test_metadata.csv')
parser.add_argument('--epoch', type=int, help='# of epoch', default=10)
parser.add_argument('--threshold', type=float, help='threshold', default=0.5)

args = parser.parse_args()


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
class SoftFocalLoss(nn.Module):
    def __init__(self, alpha0=0.5, alpha1=0.5, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: (N, 1, H, W) — raw output (before sigmoid)
        targets: (N, 1, H, W) — soft labels in [0, 1]
        """
        probs = torch.sigmoid(logits)                     # model output ∈ [0,1]
        ce_loss = F.binary_cross_entropy(probs, targets, reduction='none')  # element-wise

        # alpha = alpha0 + y * (alpha1 - alpha0)
        alpha = self.alpha0 + targets * (self.alpha1 - self.alpha0)

        # modulating factor: |y - p|^gamma
        modulating_factor = (targets - probs).abs().pow(self.gamma)

        loss = alpha * modulating_factor * ce_loss  # element-wise focal loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # shape: (N, 1, H, W)


valtransform = get_val_augmentation([256, 256])
normalize = get_normalize()


testset = GPSDataset(metadata=args.test_meta, transform=valtransform, normalize=normalize)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)



model = mit_b5_MOE(patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1,  expert_num=12, select_mode='new_topk', hidden_dims = [2, 4, 10, 16], num_k = 2, domain_num=12)



model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(torch.load("./checkpoint/MOE_epoch_2_v2.pth")["state_dict"])

criterion = SoftFocalLoss().cuda()
optimizer = torch.optim.SGD(model.module.parameters(), lr = 1e-4, momentum=0.99)  

checkpoints = []

for i in range(3):
    model_temp = mit_b5_MOE(patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1,  expert_num=12, select_mode='new_topk', hidden_dims = [2, 4, 10, 16], num_k = 2, domain_num=12)
    model_temp = torch.nn.DataParallel(model_temp).cuda()
    model_temp.load_state_dict(torch.load(f'./checkpoint/MOE_epoch_{i}_v2.pth')['state_dict'])
    checkpoints.append(model_temp)

top_indices = get_high_miou_topk_by_checkpoint(checkpoints, testloader, top_ratio=args.threshold)
test_subset = Subset(testset, top_indices)
testloader_filtered = torch.utils.data.DataLoader(test_subset, batch_size=16, shuffle=True, num_workers=2)



for epoch in range(0, 10):
    model.train()
    for batch_idx, (images, targets, country_idx) in enumerate(testloader_filtered):
        images, targets, country_idx = images.cuda(), targets.cuda().detach(), country_idx.cuda().detach()
        
        output, d_output, MI_loss = model(images, country_idx)
        targets_pl = torch.argmax(output, dim=1)
    
        prob = F.softmax(output, dim=1).detach()
        
        loss = criterion(output, prob)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

    model.eval()
    
    metrics = Metrics(2, 255)
    
    
    for batch_idx, (images, targets, country_idx) in enumerate(testloader):
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