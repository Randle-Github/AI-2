import torch.utils.data as data
import numpy as np
import torch
import torch.fft as fft
import copy
import os
import cv2

def FFT(x):
    onlyphase = copy.deepcopy(x)
    x_min = torch.min(x)
    x_max = torch.max(x)
    x = (x-x_min)/(x_max-x_min)
    x_fft = fft.fft2(x)
    x_p = torch.angle(x_fft)
    x_am = torch.abs(x_fft)
    constant = x_am.mean()
    temp = constant*torch.exp(1j*x_p)
    x_onlyphase = torch.abs(fft.ifft2(temp))
    phase_max=torch.max(x_onlyphase)
    phase_min=torch.min(x_onlyphase)
    x_onlyphase=(x_onlyphase-phase_min)/(phase_max-phase_min)
    onlyphase = x_onlyphase
    return onlyphase

class HDF5Dataset(data.Dataset):
    def __init__(self, file_path, fft=False):
        super(HDF5Dataset, self).__init__()
        self.classes = {"dog": 0, "elephant": 1, "giraffe": 2, "guitar": 3, "horse": 4, "house":5, "person": 6}
        self.fft = fft
        self.data, self.target = self.get_list(file_path)
    
    def get_list(self, file_path):
        data = []
        target = []
        for name, label in self.classes.items():
            root = os.path.join(file_path, name)
            for i in os.listdir(root):
                data.append(os.path.join(root, i))
                target.append(label)
        return data, target

    def __getitem__(self, index):
        img = torch.from_numpy(cv2.resize(cv2.imread(self.data[index]), (224, 224)) / 255).type(torch.FloatTensor)
        label = self.target[index]
        if self.fft:
            img = FFT(img)
        return img, label, index
    
    def __len__(self):
        return len(self.target)