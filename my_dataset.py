import os
import cv2
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score
from PIL import Image

def random_horizontal_flip(img):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def random_vertical_flip(img):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img

class My_Dataset(Dataset):
    def __init__(self, num, file, choice='train', transform=None, target_transform=None, dataAug=False):
        self.num = num
        self.choice = choice
        self.transform = transform
        self.target_transform = target_transform
        self.filelist = np.load(file)[:num]
        self.dataAug = dataAug


    def __getitem__(self, idx):
        fname, target = self.filelist[idx]
        target=int(target)
        img = cv2.imread(fname, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).convert('RGB')

        if self.dataAug:
            img = random_horizontal_flip(img)
            img = random_vertical_flip(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)


        return img, target

    def __len__(self):
        return self.num



class My_Dataset_1(Dataset):
    def __init__(self, data_list, choice='train', transform=None, target_transform=None, dataAug=False):
        self.num = len(data_list)
        self.choice = choice
        self.transform = transform
        self.target_transform = target_transform
        self.filelist = data_list
        self.dataAug = dataAug


    def __getitem__(self, idx):
        fname, target = self.filelist[idx]
        target = int(target)
        img = cv2.imread(fname, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).convert('RGB')

        if self.dataAug:
            img = random_horizontal_flip(img)
            img = random_vertical_flip(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return self.num
