from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import os

import cv2
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        #按空格划分句子得到word，通过调用self.addWord()函数将word加入到字典映射中
        words = sentence.split(" ")
        for word in words:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            # 构建两个字典：self.word2index和self.index2word，
            # 完成【word到对应数字】和【数字到word】的映射，并记录每个word出现的次数self.word2count
            self.n_words += 1
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 0
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = 30

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

# input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

class Flicker8k(): # flicker8k类
    def __init__(self, cfg) -> None:
        self.res = cfg["RESNET"]["ENABLE"] # 此处是否为resnet的预处理部分
        self.img, self.cap = self.get_site() # 这里获得img和caption的地址以及描述
        self.lang = Lang("caption")
        self.get_sentence(self.cap)
        if not self.res:
            self.features = torch.load("data/features.pth") # 正式训练需要直接使用features来进行操作

    def get_site(self):
        f = open("/opt/data/private/liuyangcen/homework/work3/data/captions.txt")
        img = []
        cap = []
        content = f.readlines()
        for line in content[1:]:
            line = line.replace('"', '')
            line = line.rstrip("\n")
            line = line.lower()
            i, c = line.split(".jpg,")
            img.append(i)
            cap.append(c)
        return img, cap

    def get_sentence(self, cap):
        for sentence in cap:
            self.lang.addSentence(sentence)

    def __getitem__(self, idx):
        if self.res: # res模式需要输入原始的图像
            img = cv2.imread("/opt/data/private/liuyangcen/homework/work3/data/Images/"+self.img[idx]+".jpg")
            img = transforms.functional.to_tensor(cv2.resize(img, (224,224)))
            return img, None
        else: # 非res模式需要输入得到的特征
            img = self.features[idx]
            return img, self.cap[idx]

    def __len__(self):
        return len(self.img)

if __name__ == "__main__":
    t = Flicker8k()
