from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import math
import torch.optim as optimizer

import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import models


def get_model(cfg):
    if cfg["MODEL"]["NAME"] == "RNN":
        return EncoderRNN(cfg), DecoderRNN(cfg)
    if cfg["MODEL"]["NAME"] == "GRU":
        return EncoderGRU(cfg), DecoderGRU(cfg)
    if cfg["MODEL"]["NAME"] == "RESNET50":
        return ResNet50()
    if cfg["MODEL"]["NAME"] == "TRANSFORMER":
        return Transformer(cfg) # 这里需要输出两个模型
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
    def __init__(self, cfg):
        super(EncoderGRU, self).__init__()
        self.eng = cfg["RNN"]["ENG"]
        self.fra = cfg["RNN"]["FRA"]
        self.input_size = cfg["RNN"]["INPUT_SIZE"]
        self.hidden_size = cfg["RNN"]["HIDDEN_SIZE"]
        self.num_layers = cfg["RNN"]["NUM_LAYERS"]
        self.dropout = cfg["RNN"]["DROPOUT"]
        self.bidirectional = cfg["RNN"]["BIDIRECTIONAL"]

        self.embedding = nn.Embedding(self.fra, self.input_size)
        self.rnn = nn.GRU(self.input_size, self.hidden_size, num_layers=self.num_layers, dropout=self.dropout, bidirectional=self.bidirectional)

    def forward(self, input, hidden):
        input = self.embedding(input).view(1,1,-1)
        output, hn = self.rnn(input, hidden)
        return output, hn

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

#定义Decoder
class DecoderGRU(nn.Module):
    def __init__(self, cfg):
        super(DecoderGRU, self).__init__()
        self.eng = cfg["RNN"]["ENG"]
        self.fra = cfg["RNN"]["FRA"]
        self.input_size = cfg["RNN"]["INPUT_SIZE"]
        self.hidden_size = cfg["RNN"]["HIDDEN_SIZE"]
        self.num_layers = cfg["RNN"]["NUM_LAYERS"]
        self.dropout = cfg["RNN"]["DROPOUT"]
        self.bidirectional = cfg["RNN"]["BIDIRECTIONAL"]
        self.output_size = cfg["RNN"]["OUTPUT_SIZE"]
        # self.embedding = nn.Embedding(self.eng, self.input_size)
        self.rnn = nn.GRU(self.hidden_size, self.hidden_size, num_layers=self.num_layers, dropout=self.dropout, bidirectional=self.bidirectional)                    # 调用nn.RNN()函数初始化网络  TODO: 对比nn.RNN 和nn.GRU对最终模型效果的影响
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

def ResNet50():
    return models.resnet50(pretrained=True)

class AutoTransformer(nn.Module):
    def __init__(self, time_len, input_size, pos_dim, class_num, hidden_dim):
        super(Transformer, self).__init__()
        # 定义编码器，词典大小为10，要把token编码成128维的向量
        # self.pos_embedding = self.position_encoding_init(n_position=120, emb_dim=pos_dim)
        self.pos_embedding = nn.Embedding(time_len, pos_dim)
        self.time_len = time_len
        self.transformer = nn.Transformer(d_model=input_size + pos_dim, nhead=8, batch_first=True, dim_feedforward=600, dropout=0.3) # batch_first一定不要忘记
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim*(input_size+pos_dim), 120),
            nn.Linear(120, class_num))
        self.hidden_dim = hidden_dim
        self.time_len = time_len
    
    def position_encoding_init(self, n_position, emb_dim):
        ''' Init the sinusoid position encoding table '''

        # keep dim 0 for padding token position encoding zero vector
        position_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
            if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # apply sin on 0th,2nd,4th...emb_dim
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # apply cos on 1st,3rd,5th...emb_dim
        return torch.from_numpy(position_enc).type(torch.FloatTensor)

    def forward(self, x): # (120,90)
        # x.shape: (batch_size, 120, 90)
        # pos_enc = self.pos_embedding.cuda() # (120,30)
        pos_enc = self.pos_embedding(torch.tensor(list(range(self.time_len))).cuda())
        input = torch.concat([pos_enc.unsqueeze(0).repeat(x.size(0),1,1), x], dim=2) # (120,120)
        hidden_init = torch.zeros(input.size(0), 1, input.size(2)).cuda()
        output = torch.zeros(input.size(0), self.hidden_dim, input.size(2)).cuda()
        for i in range(self.hidden_dim):
            hidden_init = self.transformer(input, hidden_init)
            output[:,i,:] = hidden_init[:,0,:]
        output = self.linear(output.flatten(1).detach())

        return output


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal_(self.linear.weight)
        init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        # q: [b_size x n_heads x len_q x d_k]
        # k: [b_size x n_heads x len_k x d_k]
        # v: [b_size x n_heads x len_v x d_v] note: (len_k == len_v)

        # attn: [b_size x n_heads x len_q x len_k]
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor
        if attn_mask is not None:
            assert attn_mask.size() == scores.size()
            scores.masked_fill_(attn_mask, -1e9)
        attn = self.dropout(self.softmax(scores))

        # outputs: [b_size x n_heads x len_q x d_v]
        context = torch.matmul(attn, v)

        return context, attn


class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid))
        self.beta = nn.Parameter(torch.zeros(d_hid))
        self.eps = eps

    def forward(self, z):
        mean = z.mean(dim=-1, keepdim=True,)
        std = z.std(dim=-1, keepdim=True,)
        ln_out = (z - mean) / (std + self.eps)
        ln_out = self.gamma * ln_out + self.beta

        return ln_out


class PosEncoding(nn.Module):
    def __init__(self, max_seq_len, d_word_vec):
        super(PosEncoding, self).__init__()
        pos_enc = np.array(
            [[pos / np.power(10000, 2.0 * (j // 2) / d_word_vec) for j in range(d_word_vec)]
            for pos in range(max_seq_len)])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
        pad_row = np.zeros([1, d_word_vec])
        pos_enc = np.concatenate([pad_row, pos_enc]).astype(np.float32)

        # additional single row for PAD idx
        self.pos_enc = nn.Embedding(max_seq_len + 1, d_word_vec)
        # fix positional encoding: exclude weight from grad computation
        self.pos_enc.weight = nn.Parameter(torch.from_numpy(pos_enc), requires_grad=False)

    def forward(self, input_len):
        max_len = torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        input_pos = tensor([list(range(1, len+1)) + [0]*(max_len-len) for len in input_len])

        return self.pos_enc(input_pos)

def proj_prob_simplex(inputs):
    # project updated weights onto a probability simplex
    # see https://arxiv.org/pdf/1101.6081.pdf
    sorted_inputs, sorted_idx = torch.sort(inputs.view(-1), descending=True)
    dim = len(sorted_inputs)
    for i in reversed(range(dim)):
        t = (sorted_inputs[:i+1].sum() - 1) / (i+1)
        if sorted_inputs[i] > t:
            break
    return torch.clamp(inputs-t, min=0.0)


def get_attn_pad_mask(seq_q, seq_k):
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    b_size, len_q = seq_q.size()
    b_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # b_size x 1 x len_k
    return pad_attn_mask.expand(b_size, len_q, len_k)  # b_size x len_q x len_k


def get_attn_subsequent_mask(seq):
    assert seq.dim() == 2
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()

    return subsequent_mask

class EncoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)

    def forward(self, enc_inputs, self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs,
                                               enc_inputs, attn_mask=self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, attn


class WeightedEncoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_branches, dropout=0.1):
        super(WeightedEncoderLayer, self).__init__()
        self.enc_self_attn = MultiBranchAttention(d_k, d_v, d_model, d_ff, n_branches, dropout)

    def forward(self, enc_inputs, self_attn_mask):
        return self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, attn_mask=self_attn_mask)


class DecoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.dec_enc_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)

    def forward(self, dec_inputs, enc_outputs, self_attn_mask, enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,
                                                        dec_inputs, attn_mask=self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs,
                                                      enc_outputs, attn_mask=enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)

        return dec_outputs, dec_self_attn, dec_enc_attn

class _MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(_MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads

        self.w_q = Linear([d_model, d_k * n_heads])
        self.w_k = Linear([d_model, d_k * n_heads])
        self.w_v = Linear([d_model, d_v * n_heads])

        self.attention = ScaledDotProductAttention(d_k, dropout)

    def forward(self, q, k, v, attn_mask):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_k x d_model]
        b_size = q.size(0)

        # q_s: [b_size x n_heads x len_q x d_k]
        # k_s: [b_size x n_heads x len_k x d_k]
        # v_s: [b_size x n_heads x len_k x d_v]
        q_s = self.w_q(q).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.w_k(k).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.w_v(v).view(b_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        if attn_mask:  # attn_mask: [b_size x len_q x len_k]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # context: [b_size x n_heads x len_q x d_v], attn: [b_size x n_heads x len_q x len_k]
        context, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask)
        # context: [b_size x len_q x n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_v)

        # return the context and attention weights
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.multihead_attn = _MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.proj = Linear(n_heads * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, q, k, v, attn_mask):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_v x d_model] note (len_k == len_v)
        residual = q
        # context: a tensor of shape [b_size x len_q x n_heads * d_v]
        context, attn = self.multihead_attn(q, k, v, attn_mask=attn_mask)

        # project back to the residual size, outputs: [b_size x len_q x d_model]
        output = self.dropout(self.proj(context))
        return self.layer_norm(residual + output), attn


class MultiBranchAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_branches, dropout):
        super(MultiBranchAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_branches = n_branches

        self.multihead_attn = _MultiHeadAttention(d_k, d_v, d_model, n_branches, dropout)
        # additional parameters for BranchedAttention
        self.w_o = nn.ModuleList([Linear(d_v, d_model) for _ in range(n_branches)])
        self.w_kp = torch.rand(n_branches)
        self.w_kp = nn.Parameter(self.w_kp/self.w_kp.sum())
        self.w_a = torch.rand(n_branches)
        self.w_a = nn.Parameter(self.w_a/self.w_a.sum())

        self.pos_ffn = nn.ModuleList([
            PoswiseFeedForwardNet(d_model, d_ff//n_branches, dropout) for _ in range(n_branches)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

        init.xavier_normal(self.w_o)

    def forward(self, q, k, v, attn_mask):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_v x d_model] note (len_k == len_v)
        residual = q

        # context: a tensor of shape [b_size x len_q x n_branches * d_v]
        context, attn = self.multih_attn(q, k, v, attn_mask=attn_mask)

        # context: a list of tensors of shape [b_size x len_q x d_v] len: n_branches
        context = context.split(self.d_v, dim=-1)

        # outputs: a list of tensors of shape [b_size x len_q x d_model] len: n_branches
        outputs = [self.w_o[i](context[i]) for i in range(self.n_branches)]
        outputs = [kappa * output for kappa, output in zip(self.w_kp, outputs)]
        outputs = [pos_ffn(output) for pos_ffn, output in zip(self.pos_ffn, outputs)]
        outputs = [alpha * output for alpha, output in zip(self.w_a, outputs)]

        # output: [b_size x len_q x d_model]
        output = self.dropout(torch.stack(outputs).sum(dim=0))
        return self.layer_norm(residual + output), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, inputs):
        # inputs: [b_size x len_q x d_model]
        residual = inputs
        output = self.relu(self.conv1(inputs.transpose(1, 2)))

        # outputs: [b_size x len_q x d_model]
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)

        return self.layer_norm(residual + output)

class WeightedDecoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_branches, dropout=0.1):
        super(WeightedDecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_k, d_v, d_model, n_branches, dropout)
        self.dec_enc_attn = MultiBranchAttention(d_k, d_v, d_model, d_ff, n_branches, dropout)

    def forward(self, dec_inputs, enc_outputs, self_attn_mask, enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,
                                                        dec_inputs, attn_mask=self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs,
                                                      enc_outputs, attn_mask=enc_attn_mask)
        
        return dec_outputs, dec_self_attn, dec_enc_attn

class Encoder(nn.Module):
    def __init__(self, n_layers, d_k, d_v, d_model, d_ff, n_heads,
                 max_seq_len, src_vocab_size, dropout=0.1, weighted=False):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.pos_emb = PosEncoding(max_seq_len * 10, d_model) # TODO: *10 fix
        self.dropout_emb = nn.Dropout(dropout)
        self.layer_type = EncoderLayer if not weighted else WeightedEncoderLayer
        self.layers = nn.ModuleList(
            [self.layer_type(d_k, d_v, d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, enc_inputs, enc_inputs_len, return_attn=False):
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs += self.pos_emb(enc_inputs_len) # Adding positional encoding TODO: note
        enc_outputs = self.dropout_emb(enc_outputs)

        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            if return_attn:
                enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self, n_layers, d_k, d_v, d_model, d_ff, n_heads,
                 max_seq_len, tgt_vocab_size, dropout=0.1, weighted=False):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0 )
        self.pos_emb = PosEncoding(max_seq_len * 10, d_model) # TODO: *10 fix
        self.dropout_emb = nn.Dropout(dropout)
        self.layer_type = DecoderLayer if not weighted else WeightedDecoderLayer
        self.layers = nn.ModuleList(
            [self.layer_type(d_k, d_v, d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, dec_inputs, dec_inputs_len, enc_inputs, enc_outputs, return_attn=False):
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs += self.pos_emb(dec_inputs_len) # Adding positional encoding # TODO: note
        dec_outputs = self.dropout_emb(dec_outputs)

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)

        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_pad_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs,
                                                             self_attn_mask=dec_self_attn_mask,
                                                             enc_attn_mask=dec_enc_attn_pad_mask)
            if return_attn:
                dec_self_attns.append(dec_self_attn)
                dec_enc_attns.append(dec_enc_attn)

        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self, opt):
        super(Transformer, self).__init__()
        self.encoder = Encoder(n_layers=6, d_k=64, d_v=64, d_model=512, d_ff=2048, n_heads=8,
                               max_src_seq_len=30, src_vocab_size=1000, dropout=0.)
        self.decoder = Decoder(n_layers=6, d_k=64, d_v=64, d_model=512, d_ff=2048, n_heads=8,
                               max_tgt_seq_len=30, tgt_vocab_size=0, dropout=0.)
        self.tgt_proj = Linear(d_model=512, tgt_vocab_size=0, bias=False)
        # self.weighted_model = weighted_model

    def trainable_params(self):
        # Avoid updating the position encoding
        params = filter(lambda p: p[1].requires_grad, self.named_parameters())
        # Add a separate parameter group for the weighted_model
        param_groups = []
        base_params = {'params': [], 'type': 'base'}
        weighted_params = {'params': [], 'type': 'weighted'}
        for name, param in params:
            if 'w_kp' in name or 'w_a' in name:
                weighted_params['params'].append(param)
            else:
                base_params['params'].append(param)
        param_groups.append(base_params)
        param_groups.append(weighted_params)

        return param_groups

    def encode(self, enc_inputs, enc_inputs_len, return_attn=False):
        return self.encoder(enc_inputs, enc_inputs_len, return_attn)

    def decode(self, dec_inputs, dec_inputs_len, enc_inputs, enc_outputs, return_attn=False):
        return self.decoder(dec_inputs, dec_inputs_len, enc_inputs, enc_outputs, return_attn)

    def forward(self, enc_inputs, enc_inputs_len, dec_inputs, dec_inputs_len, return_attn=False):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs, enc_inputs_len, return_attn)
        dec_outputs, dec_self_attns, dec_enc_attns = \
            self.decoder(dec_inputs, dec_inputs_len, enc_inputs, enc_outputs, return_attn)
        dec_logits = self.tgt_proj(dec_outputs)

        return dec_logits.view(-1, dec_logits.size(-1)), \
               enc_self_attns, dec_self_attns, dec_enc_attns

    def proj_grad(self):
        pass
