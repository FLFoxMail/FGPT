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
        attn_output = torch.zeros_like(x)
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
    def __init__(self, d_k, d_v, d_model, num_heads, d_diff, n_layer):
        super(SGPT, self).__init__()
        self.layers = nn.ModuleList([Block(d_k, d_v, d_model, num_heads, d_diff) for _ in range(n_layer)])
        self.normal = nn.LayerNorm(d_model)
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_diff = d_diff
        self.n_layer = n_layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.normal(x)
        return x