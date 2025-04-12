from torch.utils.data import Dataset, DataLoader
import jieba  # 使用结巴分词作为中文分词器 pip install jieba
from torchtext.data.utils import get_tokenizer # pip install torchtext
from utils.data_preperation import tokenize_text, build_vocabulary
import os
import torch
# 语料数据集（从文本里加载）继承自Dataset类
class FDataset(Dataset):
    def __init__(self, file_path, tokenizer_type='basic_chinese'):
        self.data = load_data(file_path)
        self.tokens = tokenize_text(self.data, tokenizer_type)
        self.vocab = build_vocabulary(self.tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token = self.tokens[idx]
        token_id = self.vocab[token]
        return torch.tensor(token_id, dtype=torch.long)