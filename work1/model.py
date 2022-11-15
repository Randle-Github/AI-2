import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

def get_model(cfg):
    if cfg["MODEL"]["NAME"] == "MLP":
        return MLP(cfg["MODEL"]["INPUT_DIM"], cfg["MODEL"]["NUM_CLASSES"], inner_dim=cfg["MODEL"]["INNER_DIM"], layer_num=cfg["MODEL"]["LAYER_NUM"])
    elif cfg["MODEL"]["NAME"] == "LINEAR":
        return LinearNet(cfg[""])
    else:
        raise Exception("no such model")

class MLP(nn.Module):
    def __init__(self, num_inputs, num_outputs, inner_dim=128, layer_num=3):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(num_inputs, inner_dim),
            nn.BatchNorm1d(inner_dim),
            nn.ReLU(), 
            nn.Linear(inner_dim, inner_dim),
            nn.BatchNorm1d(inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, num_outputs),
            nn.BatchNorm1d(num_outputs))
    def forward(self, x): # x shape: (batch, 1, 28, 28)
        y = self.mlp(x.view(x.shape[0], -1))
        y = nn.Softmax(dim=1)(y)
        return y

class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
    def forward(self, x): # x shape: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))
        y = nn.Softmax(dim=1)(y)
        return y