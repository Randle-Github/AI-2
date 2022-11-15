from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import os

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np

import matplotlib.pylab as plt
import torch.distributed as dist
import torch.multiprocessing as mp

from dataloader import *
from model import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import yaml

def get_config(site):
    with open(site, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    return cfg

def main():
    pass
    mp.multiprocessing(train, cfg)

loss_his = []
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, 
          max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

    loss = 0.
    for ei in range(input_length):
        # 完成训练部分代码
        #######Your Code#######
        # 调用EncoderRNN类完成整个编码计算流程
        encoder_outputs[ei], encoder_hidden = encoder(input_tensor[ei].to(device), encoder_hidden)
        #######End#######
    
    # decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    for di in range(target_length):
        
        # 完成训练部分代码
        #######Your Code#######
        # 调用DecoderRNN类完成整个解码计算流程
        decoder_outputs, decoder_hidden = decoder(encoder_outputs[ei], decoder_hidden)
        loss += criterion(decoder_outputs, target_tensor[di])
        # decoder_input = target_tensor[di]
    
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(encoder, decoder, epoches, print_every=100, plot_every=100, learning_rate=0.01):
    global loss_his
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(epoches)] # 每次只能pick一个样本
    criterion = nn.NLLLoss()

    for iter in range(1, epoches + 1):
        
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = train(input_tensor, target_tensor, encoder,
        decoder, encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print(iter, "/", epoches, ": ",  print_loss_avg)
            print_loss_total = 0
            loss_his.append(float(loss))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_outputs[ei], encoder_hidden = encoder(input_tensor[ei].to(device), encoder_hidden)

        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(max_length):
            decoder_outputs, decoder_hidden = decoder(encoder_outputs[ei], decoder_hidden)
            topv, topi = decoder_outputs.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

        return decoded_words

if __name__ == "__main__":
    cfg = get_config("config.yaml")
    encoder1, decoder1 = get_model(cfg)
    encoder1.to(device)
    decoder1.to(device)

    if cfg["TRAIN"]["ENABLE"]:
        trainIters(encoder1, decoder1, 10000, print_every=1000, learning_rate=0.01)
        trainIters(encoder1, decoder1, 10000, print_every=1000, learning_rate=0.001)
        trainIters(encoder1, decoder1, 10000, print_every=1000, learning_rate=0.0001)
        torch.save(encoder1.state_dict(), "encoder1.pth")
        torch.save(decoder1.state_dict(), "decoder1.pth")
        if cfg["TRAIN"]["PLOT"]:
            plt.title("Training Loss Curve")
            plt.plot(np.arange(len(loss_his)) * 100, loss_his, color='green')
            plt.show()
    else: # directly load state_dict
        state1 = torch.load("encoder1.pth")
        state2 = torch.load("decoder1.pth")
        encoder1.load_state_dict(state1)
        decoder1.load_state_dict(state2)

    if cfg["TEST"]["ENABLE"]:
        pair = random.choice(pairs)
        test_sentence = pair[0]
        translated_sentence = evaluate(encoder1, decoder1, test_sentence)
        print("original sentence: {}\ntarget_sentence: {}\ntranslated_sentence: ".format(pair[0],pair[1]), end="")
        for word in translated_sentence:
            if word == "<EOS>":
                break
            print(word, end = " ")