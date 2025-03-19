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
        y = torch.softmax(y, dim=-1)
        y = self.normal(x + y)
        return y
    
d_model = 512
d_k = 32
d_v = 24
batch_size = 32
num_heads = 8
seq_len = 120

x = torch.rand(batch_size, seq_len, d_model)
model = MultiHeadAttention(d_k, d_v, d_model, num_heads) 
y = model(x)

print(y.shape)       
        