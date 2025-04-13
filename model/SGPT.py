import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import jieba  # 使用结巴分词作为中文分词器
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import math
import os


# 单头自注意力（非batch版本，高效写法）
class SingleHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model):
        super(SingleHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.K = nn.Linear(d_model, d_k)
        self.Q = nn.Linear(d_model, d_k)
        self.V = nn.Linear(d_model, d_v)
        self.W = nn.Linear(d_v, d_model)
        self.normal = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_length, _ = x.size()
        K = self.K(x)
        Q = self.Q(x)
        V = self.V(x)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, V)
        y = self.W(attn_output)
        y = self.normal(y)
        y = x + y
        return y


# 多头注意力（添加一层循环，低效写法）
class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.num_heads = num_heads
        self.K = nn.Linear(d_model, num_heads * d_k)
        self.Q = nn.Linear(d_model, num_heads * d_k)
        self.V = nn.Linear(d_model, num_heads * d_v)
        self.W = nn.Linear(num_heads * d_v, d_model)
        self.normal = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_length, _ = x.size()
        head_outputs = []
        for h in range(self.num_heads):
            K_h = self.K(x)[:, h * self.d_k:(h + 1) * self.d_k]
            Q_h = self.Q(x)[:, h * self.d_k:(h + 1) * self.d_k]
            V_h = self.V(x)[:, h * self.d_v:(h + 1) * self.d_v]
            attn_scores = torch.matmul(Q_h, K_h.transpose(-2, -1)) / math.sqrt(self.d_k)
            attn_probs = torch.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_probs, V_h)
            head_outputs.append(attn_output)
        head_outputs = torch.cat(head_outputs, dim=-1)
        y = self.W(head_outputs)
        y = self.normal(y)
        y = x + y
        return y


# 多头注意力（批量版本，添加两层循环，低效写法）
class MultiHeadAttentionBatch(nn.Module):
    def __init__(self, d_k, d_v, d_model, num_heads):
        super(MultiHeadAttentionBatch, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.num_heads = num_heads
        self.K = nn.Linear(d_model, num_heads * d_k)
        self.Q = nn.Linear(d_model, num_heads * d_k)
        self.V = nn.Linear(d_model, num_heads * d_v)
        self.W = nn.Linear(num_heads * d_v, d_model)
        self.normal = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        attn_output = torch.zeros(batch_size, seq_length, self.d_model * self.num_heads).to(x.device)
        for i in range(seq_length):
            x_i = x[:, i, :]
            K = self.K(x_i)
            Q = self.Q(x_i)
            V = self.V(x_i)
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            attn_probs = torch.softmax(attn_scores, dim=-1)
            attn_output_i = torch.matmul(attn_probs, V)
            attn_output[:, i, :] = attn_output_i
        attn_output = attn_output.view(batch_size, seq_length, -1)
        y = self.W(attn_output)
        y = self.normal(y)
        y = x + y
        return y


# BLOCKS
class Block(nn.Module):
    def __init__(self, d_k, d_v, d_model, num_heads, d_diff):
        super(Block, self).__init__()
        self.attn = MultiHeadAttentionBatch(d_k, d_v, d_model, num_heads)
        self.feed = nn.Sequential(
            nn.Linear(d_model, d_diff),
            nn.ReLU(),
            nn.Linear(d_diff, d_model)
        )
        self.normal = nn.LayerNorm(d_model)

    def forward(self, x):
        y = self.attn(x)
        y = self.feed(y)
        y = x + y  # 恢复残差连接
        y = self.normal(y)
        return y


# 慢GPT（低效写法：没有使用并行计算层）
class SGPT(nn.Module):
    def __init__(self, d_k, d_v, d_model, num_heads, d_diff, n_layer, vocab_size, max_seq_len, device, pos_encoding_type='learnable'):
        super(SGPT, self).__init__()
        self.pos_encoding_type = pos_encoding_type
        # 新增词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model) 
        self.device = device
        # 新增位置编码层
        if (self.pos_encoding_type == 'learnable'):
            self.pos_encoding = nn.Embedding(max_seq_len, d_model)
        elif (self.pos_encoding_type == 'sinusoidal'):
            # 固定正弦余弦位置编码，并在设备上注册为缓冲区
            self.register_buffer('pos_encoding', self._sinusoidal_pos_encoding(max_seq_len, d_model)).to(self.device)
        self.layers = nn.ModuleList([Block(d_k, d_v, d_model, num_heads, d_diff) for _ in range(n_layer)])
        self.normal = nn.LayerNorm(d_model)
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_diff = d_diff
        self.n_layer = n_layer
        self.final_linear = nn.Linear(d_model, vocab_size)
    
    @staticmethod
    def _sinusoidal_pos_encoding(max_seq_len, d_model):
        """生成固定正弦余弦位置编码"""
        pos = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div_term)
        pe[0, :, 1::2] = torch.cos(pos * div_term)
        return pe

    def forward(self, x):
        batch_size, seq_length = x.size()
        # 1. 词嵌入
        tocken_embedding = self.embedding(x)
        # 2. 位置编码
        pos_indices = torch.arange(seq_length).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        if (self.pos_encoding_type == 'learnable'):
            pos_embedding = self.pos_encoding(pos_indices)
        elif (self.pos_encoding_type == 'sinusoidal'):
            pos_embedding = self.pos_encoding[:, :seq_length, :]
        x = tocken_embedding + pos_embedding
        
        for layer in self.layers:
            x = layer(x)
        x = self.normal(x)
        x = self.final_linear(x)
        return x