import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

def get_datasets(mode, cfg):
    '''
    mode: "TRAIN", "TEST"
    '''
    if cfg["DATASETS"]["NAME"] == "FASIONMNIST":
        if mode == "TRAIN":
            return torchvision.datasets.FashionMNIST(root='FashionMNIST', train=True, download=False, transform=transforms.ToTensor())
        elif mode == "TEST":
            return torchvision.datasets.FashionMNIST(root='FashionMNIST', train=False, download=False, transform=transforms.ToTensor())
        else:
            raise Exception("no such mode")
    elif cfg["DATASETS"]["NAME"] == "CIFAR10":
        if mode == "TRAIN":
            return torchvision.datasets.CIFAR10(root='CIFAR10', train=True, download=True, transform=transforms.ToTensor())
        elif mode == "TEST":
            return torchvision.datasets.CIFAR10(root='CIFAR10', train=False, download=True, transform=transforms.ToTensor())
        else:
            raise Exception("no such mode")
    else:
        raise Exception("no such dataset")

