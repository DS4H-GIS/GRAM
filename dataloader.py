import os
import glob
import torch
import numpy as np
import pandas as pd
from skimage import io, transform
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as TF
import random
from PIL import Image
from torchvision import io
import copy

def cutout(img, mask, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3, value_min=0, value_max=255, pixel_level=True):
    if random.random() < p:

        img_h, img_w, img_c = img.shape

        while True:
            size = np.random.uniform(size_min, size_max) * img_h * img_w
            ratio = np.random.uniform(ratio_1, ratio_2)
            erase_w = int(np.sqrt(size / ratio))
            erase_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, img_w)
            y = np.random.randint(0, img_h)

            if x + erase_w <= img_w and y + erase_h <= img_h:
                break

        if pixel_level:
            value = np.random.uniform(value_min, value_max, (erase_h, erase_w, img_c))
        else:
            value = np.random.uniform(value_min, value_max)
            
        img[y:y + erase_h, x:x + erase_w] = torch.from_numpy(value)
        mask[y:y + erase_h, x:x + erase_w] = 255

    return img, mask


class GPSDataset(Dataset):
    def __init__(self, metadata,  transform=None,  normalize=None):
        self.metadata = pd.read_csv(metadata).values
        self.transform = transform
        self.normalize = normalize
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_path = self.metadata[idx][0]
        label_path = self.metadata[idx][1]
        country_idx = self.metadata[idx][2]

        image = io.read_image(image_path)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
            
        try:
            land_value = Image.open(label_path)
            land_value = torch.tensor(np.array(land_value)).unsqueeze(0)
            land_value = torch.clamp(land_value, max=1)

        except:
            land_value = torch.zeros(1,256,256)
            
        
        
        if self.transform:
            image, land_value = self.transform(image, land_value)
            
            
        image, land_value = self.normalize(image, land_value)        
        land_value = land_value.squeeze(0).long()

        return image, land_value, country_idx



    
class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, images):
        normalized = np.stack([F.normalize(x, self.mean, self.std, self.inplace) for x in images]) 
        return normalized
        
class ToTensor(object):
    def __call__(self, images):
        images = images.transpose((0, 3, 1, 2))
        return torch.from_numpy(images).float() 