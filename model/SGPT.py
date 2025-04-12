import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import math
import os
import jieba  # 使用结巴分词作为中文分词器
from datasets import load_dataset


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

class ChineseTextDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        input_seq = self.data[idx:idx + self.seq_length]
        target_seq = self.data[idx + 1:idx + self.seq_length + 1]
        input_ids = []
        target_ids = []
        for token in input_seq:
            input_ids.append(token)
        for token in target_seq:
            target_ids.append(token)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        target_ids = torch.tensor(target_ids, dtype=torch.long)
        return input_ids, target_ids
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer

def prepare_data():
    try:
        # 假设本地 TXT 文件路径
        file_path = "/path/to/your/local/file.txt"
        if not os.path.exists(file_path):
            print(f"指定的文件 {file_path} 不存在，请检查路径。")
            return None
        with open(file_path, 'r', encoding='utf-8') as f:
            all_text = f.read()
        # 使用结巴分词进行中文分词
        tokenized_text = jieba.lcut(all_text)
        seq_length = 32
        custom_dataset = ChineseTextDataset(tokenized_text, seq_length)
        dataloader = DataLoader(custom_dataset, batch_size=64, shuffle=True)
        return dataloader
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        return None



def train(model, dataloader, optimizer, criterion, epochs):
    for epoch in range(epochs):
        total_loss = 0
        for i, (input_ids, target_ids) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(input_ids)
            output = output.view(-1, output.size(-1))
            target_ids = target_ids.view(-1)
            loss = criterion(output, target_ids)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')


def generate_text(model, start_text, max_length=50, temperature=1.0):
    model.eval()
    # 更换为中文分词器
    tokenizer = jieba.lcut
    start_tokens = tokenizer(start_text)
    input_ids = []
    for token in start_tokens:
        input_ids.append(1)  # 这里假设词汇表id为1，实际应根据词汇表来
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    generated_text = start_text

    for _ in range(max_length):
        output = model(input_ids)
        logits = output[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        next_word = "unk"  # 这里假设未知词为"unk"，实际应根据词汇表来
        generated_text += next_word

    return generated_text

        
    
    

# 使用 127.0.0.1:7890 代理
import os
import sys

# 设置环境变量
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"



if __name__ == "__main__":
    d_k = 64
    d_v = 64
    d_model = 256
    num_heads = 4
    d_diff = 512
    n_layer = 2
    epochs = 5

    dataloader = prepare_data()
    model = SGPT(d_k, d_v, d_model, num_heads, d_diff, n_layer)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train(model, dataloader, optimizer, criterion, epochs)

    start_text = "Once upon a time"
    generated = generate_text(model, start_text)
    print("Generated Text:")
    print(generated)