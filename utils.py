import torch
import numpy 


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