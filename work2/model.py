import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

def get_model(cfg):
    if cfg["MODEL"]["NAME"] == "MLP":
        return MLP(cfg["MODEL"]["INPUT_DIM"], cfg["MODEL"]["NUM_CLASSES"], inner_dim=cfg["MODEL"]["INNER_DIM"], layer_num=cfg["MODEL"]["LAYER_NUM"])
    elif cfg["MODEL"]["NAME"] == "LINEAR":
        return LinearNet(cfg[""])
    elif cfg["MODEL"]["NAME"] == "LENET":
        return LeNet(cfg)
    elif cfg["MODEL"]["NAME"] == "RESNET18":
        return ResNet18()
    elif cfg["MODEL"]["NAME"] == "RESNET50":
        return ResNet50()
    elif cfg["MODEL"]["NAME"] == "RESNET101":
        return ResNet101()
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

class LeNet(nn.Module):
    def __init__(self, cfg):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 6, 5),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(6, 16, 5),
        nn.MaxPool2d(2, 2))
        self.linear = nn.Sequential(
        nn.Linear(16 * 5 * 5, 120),
        nn.BatchNorm1d(120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.BatchNorm1d(84),
        nn.ReLU(),
        nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])