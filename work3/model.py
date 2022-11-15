from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

def get_model(cfg):
    if cfg["MODEL"]["NAME"] == "RNN":
        return EncoderRNN(cfg), DecoderRNN(cfg)
    else:
        raise Exception("no such model")

class EncoderRNN(nn.Module):
    def __init__(self, cfg):
        super(EncoderRNN, self).__init__()
        self.eng = cfg["RNN"]["ENG"]
        self.fra = cfg["RNN"]["FRA"]
        self.input_size = cfg["RNN"]["INPUT_SIZE"]
        self.hidden_size = cfg["RNN"]["HIDDEN_SIZE"]
        self.num_layers = cfg["RNN"]["NUM_LAYERS"]
        self.dropout = cfg["RNN"]["DROPOUT"]
        self.bidirectional = cfg["RNN"]["BIDIRECTIONAL"]

        self.embedding = nn.Embedding(self.fra, self.input_size)
        self.rnn = nn.RNN(self.input_size, self.hidden_size, num_layers=self.num_layers, dropout=self.dropout, bidirectional=self.bidirectional)

    def forward(self, input, hidden):
        input = self.embedding(input).view(1,1,-1)
        output, hn = self.rnn(input, hidden)
        return output, hn

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class DecoderRNN(nn.Module):
    def __init__(self, cfg):
        super(DecoderRNN, self).__init__()
        self.eng = cfg["RNN"]["ENG"]
        self.fra = cfg["RNN"]["FRA"]
        self.input_size = cfg["RNN"]["INPUT_SIZE"]
        self.hidden_size = cfg["RNN"]["HIDDEN_SIZE"]
        self.num_layers = cfg["RNN"]["NUM_LAYERS"]
        self.dropout = cfg["RNN"]["DROPOUT"]
        self.bidirectional = cfg["RNN"]["BIDIRECTIONAL"]
        self.output_size = cfg["RNN"]["OUTPUT_SIZE"]
        # self.embedding = nn.Embedding(self.eng, self.input_size)
        self.rnn = nn.RNN(self.hidden_size, self.hidden_size, num_layers=self.num_layers, dropout=self.dropout, bidirectional=self.bidirectional)                    # 调用nn.RNN()函数初始化网络  TODO: 对比nn.RNN 和nn.GRU对最终模型效果的影响
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # 使用self.embedding将输入转化为向量表达,对于decoder，可以对该输出使用激活函数（如：F.relu）
        # tips:使用self.embedding对input进行编码后，你可能需要使用view函数将输出向量的维度进行重组(view(1,1,-1))
        input = input.view(1,1,-1) # => batchSize × seq_len × feaSize
        # 调用self.rnn计算当前时间步的hidden state和output
        # tips:使用第t-1步的output作为第t步的input
        input = F.relu(input)
        output,hn = self.rnn(input, hidden) # output: batchSize × seq_len × feaSize; hn: numLayers*d × batchSize × hiddenSize
        output = self.softmax(self.out(output[0]))
        #relu激活，softmax计算输出
        return output, hn

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class EncoderGRU(nn.Module):
    #初始化
    def __init__(self, featureSize, hiddenSize, embedding, numLayers=1, dropout=0.1, bidirectional=True):
        super(EncoderRNN, self).__init__()
        self.embedding = embedding
        #核心API，建立双向GRU
        self.gru = nn.GRU(featureSize, hiddenSize, num_layers=numLayers, dropout=(0 if numLayers==1 else dropout), bidirectional=bidirectional, batch_first=True)
        #超参
        self.featureSize = featureSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.bidirectional = bidirectional

    #前向计算，训练和测试必须的部分
    def forward(self, input, lengths, hidden):
        # input: batchSize × seq_len; hidden: numLayers*d × batchSize × hiddenSize
        #给定输入
        input = self.embedding(input) # => batchSize × seq_len × feaSize
        #加入 paddle 方便计算
        packed = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)
        output, hn = self.gru(packed, hidden) # output: batchSize × seq_len × hiddenSize*d; hn: numLayers*d × batchSize × hiddenSize 
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        if self.bidirectional:
            output = output[:,:,:self.hiddenSize] + output[:,:,self.hiddenSize:]
        return output, hn
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

#定义Decoder
class DecoderGRU(nn.Module):
    #初始化
    def __init__(self, featureSize, hiddenSize, outputSize, embedding, numLayers=1, dropout=0.1):
        super(DecoderRNN, self).__init__()
        self.embedding = embedding
        #核心API
        self.gru = nn.GRU(featureSize, hiddenSize, num_layers=numLayers, batch_first=True)
        self.out = nn.Linear(featureSize, outputSize)
    #定义前向计算
    def forward(self, input, hidden):
        # input: batchSize × seq_len; hidden: numLayers*d × batchSize × hiddenSize
        input = self.embedding(input) # => batchSize × seq_len × feaSize
        #relu激活，softmax计算输出
        input = F.relu(input)
        output,hn = self.gru(input, hidden) # output: batchSize × seq_len × feaSize; hn: numLayers*d × batchSize × hiddenSize
        output = F.log_softmax(self.out(output), dim=2) # output: batchSize × seq_len × outputSize
        return output, hn, torch.zeros([input.size(0), 1, input.size(1)])
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)