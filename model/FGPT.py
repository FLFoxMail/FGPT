import torch
import torch.nn as nn

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
        # x = [bs, seq, d_model]
        K = self.K(x).view(-1, self.num_heads, self.d_k) # [bs * seq, heads, d_k]
        Q = self.Q(x).view(-1, self.num_heads, self.d_k) # [bs * seq, heads, d_k]
        V = self.V(x).view(-1, self.num_heads, self.d_v) # [bs * seq, heads, d_v]
        
        # Q * K ^T  = [bs * seq, heads, heads]
        attention_weights = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attention_weights = torch.softmax(attention_weights, dim= -1)
        # # [bs * seq, heads, d_v] 
        y = torch.matmul(attention_weights, V).view(-1, x.size(1), self.num_heads * self.d_v) # view 后 [bs, seq, heads * d_v]
        y = self.W(y) # [bs, seq, d_model]
        y = self.normal(y)
        y = x + y
        return y
    
class Block(nn.Module):
    def __init__(self, d_k, d_v, d_model, num_heads, d_diff):
        super(Block, self).__init__()
        self.attn = MultiHeadAttention(d_k, d_v, d_model, num_heads)
        self.feed = nn.Sequential(
            nn.Linear(d_model, d_diff),
            nn.ReLU(),
            nn.Linear(d_diff, d_model)
        )
        self.normal = nn.LayerNorm(d_model)
        
    def forward(self, x):
        y = self.attn(x)
        y = self.normal(self.feed(y))
        y = x + y
        return y
    
class FGPT(nn.Module):
    def __init__(self, d_k, d_v, d_model, num_heads, d_diff, n_layer):
        super(FGPT, self).__init__()
        
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