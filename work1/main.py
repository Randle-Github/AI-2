from cProfile import label
import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import yaml

from model import get_model
from datasets import get_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_config(site):
    with open(site, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    return cfg

loss_rec = []
train_acc = []
test_acc = []

def train_epoch(epoch, model, train_loader, optimizer, cfg):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    aver_loss = 0.
    aver_acc = 0.
    for data, label in train_loader:
        preds = model(data.to(device))
        loss = loss_fn(preds, label.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        aver_loss += float(loss) / len(data)
        aver_acc += (torch.sum(torch.argmax(preds, dim=1) == label.to(device))) / len(data)
    print("epoch:", epoch, "loss", aver_loss/len(train_loader), "acc", aver_acc/len(train_loader))
    loss_rec.append(aver_loss/len(train_loader))
    train_acc.append(aver_acc/len(train_loader))

def test_epoch(epoch, model, test_loader, cfg):
    model.eval()
    aver_acc = 0.
    for data, label in test_loader:
        preds = model(data.to(device))
        aver_acc += (torch.sum(torch.argmax(preds, dim=1) == label.to(device))) / len(data)
    print("epoch:", epoch, "acc", aver_acc/len(test_loader))
    test_acc.append(aver_acc/len(test_loader))


def get_optimizer(model, cfg):
    if cfg["OPTIMIZER"] == "SGD":
        return torch.optim.SGD(model.parameters(), lr=cfg["LR"])
    elif cfg["OPTIMIZER"] == "ADAM":
        return torch.optim.Adam(model.parameters(), lr=cfg["LR"])
    else:
        raise Exception("no such optimizer mode")

def main(cfg):
    model = get_model(cfg)
    model = model.to(device)
    train_dataset = get_datasets("TRAIN", cfg)
    test_dataset = get_datasets("TEST", cfg)
    train_loader =  torch.utils.data.DataLoader(train_dataset, batch_size=cfg["BS"], shuffle=True, num_workers=cfg["NUM_WORKERS"])
    test_loader =  torch.utils.data.DataLoader(test_dataset, batch_size=cfg["BS"], shuffle=False, num_workers=cfg["NUM_WORKERS"])
    optimizer = get_optimizer(model, cfg)
    if cfg["TRAIN"]["ENABLE"]:
        for epoch in range(cfg["TRAIN"]["EPOCHES"]):
            train_epoch(epoch, model, train_loader, optimizer, cfg)
            if cfg["TRAIN"]["VAL"]:
                test_epoch(epoch, model, test_loader, cfg)
    print(max(test_acc))

    if cfg["VISUAL"]:
        plt.plot(np.arange(len(loss_rec)), loss_rec, label="train loss")
        plt.legend()
        plt.show()

    

if __name__ == "__main__":
    cfg = get_config("config.yaml")
    main(cfg)