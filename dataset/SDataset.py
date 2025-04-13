from torch.utils.data import Dataset, DataLoader

import os
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jieba
from torchtext.data.utils import get_tokenizer
from utils.data_preperation import build_vocabulary_with_data
from utils.set_seed import set_seed
import json
import random
from tqdm import tqdm
import concurrent.futures

# 语料数据集（从文本里加载）继承自Dataset类
class SDataset(Dataset):
    def __init__(self, data_dir, tokenizer_type='basic_chinese', seq_len=32, usage_percentage=0.5,
                  train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, mode='train', vocab_path=None):
        self.data = self._load_data(data_dir)
        self.tokenizer_type = tokenizer_type
        self.seq_len = seq_len
        
        # 定义分词器
        self.tokenizer = get_tokenizer('basic_chinese') if self.tokenizer_type == 'basic_english' else jieba.lcut
        # 构建词汇表
        self.vocab_path = vocab_path
        self.vocab = self._build_vocabulary()
        self.usage_percentage = usage_percentage
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.mode = mode
        
        
        # 随机采样索引
        self.all_indices = []
        self.sample_indices = []
        self._random_sample_indices()
        # 计算每个样本的滑动窗口数量
        self.window_counts = self._calculate_window_counts()
        # 统计每个样本累计的滑动窗口数量
        self.cumulative_window_counts = self._calculate_cumulative_window_counts()
    
    def _calculate_window_counts(self):
        window_counts = []
        for index in self.sample_indices:
            text = self.data[index]
            tokens = self.tokenizer(text)
            window_count = 1 if len(tokens) <= self.seq_len else len(tokens) - self.seq_len + 1
            window_counts.append(window_count)
        return window_counts
            
    def _calculate_cumulative_window_counts(self):
        cumulative_counts = []
        cumulative_count = 0
        for count in self.window_counts:
            cumulative_count += count
            cumulative_counts.append(cumulative_count)
        return cumulative_counts
        
        
    def _random_sample_indices(self):
        num_samples = int(len(self.data) * self.usage_percentage)
        self.all_indices = random.sample(range(len(self.data)), num_samples)
        random.shuffle(self.all_indices)
        
        train_size = int(num_samples * self.train_ratio)
        val_size = int(num_samples * self.val_ratio)
        
        print(f"train_size: {train_size}, val_size: {val_size}, test_size: {num_samples - train_size - val_size}")
        
        if self.mode == 'train':
            self.sample_indices = self.all_indices[:train_size]
        elif self.mode == 'val':
            self.sample_indices = self.all_indices[train_size:train_size+val_size]
        elif self.mode == 'test':
            self.sample_indices = self.all_indices[train_size+val_size:]
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def tokenizer_line(self, text):
        return self.tokenizer(text)
                
    def _build_vocabulary(self):
        # 先从文件中读取词汇表，不存在则从数据中构建
        if os.path.exists(self.vocab_path):
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                vocab_list = [line.strip() for line in f]
            return build_vocabulary_with_data(vocab_list)
        else:
            def generate_tokens():
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self.tokenizer_line, line) for line in self.data]
                    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                        yield from future.result()

        vocab = build_vocabulary_with_data(generate_tokens())
        # 检查路径是否存在，不存在则创建
        if not os.path.exists(os.path.dirname(self.vocab_path)):
            os.makedirs(os.path.dirname(self.vocab_path))
        # 将词汇表保存为文本文件
        with open(self.vocab_path, 'w', encoding='utf-8') as f:
            for word in vocab.get_itos():
                f.write(word + '\n')
        return vocab
            
    def _load_data(self, file_path):
        all_data = []
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"file_path {file_path} does not exist.")
        for root, dirs, files in os.walk(file_path):
            for file in files:
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        all_data.append(data['title']+':'+data['text'])
        print(f"Total data size: {len(all_data)}")
        return all_data

    def __len__(self):
        # 返回累计滑动窗口数量
        return self.cumulative_window_counts[-1]

    def __getitem__(self, idx):
        # 根据累计滑动窗口数量找到对应的样本索引，并计算滑动窗口在样本中的位置
        for i in range(len(self.cumulative_window_counts)):
            if idx < self.cumulative_window_counts[i]:
                sample_index = self.sample_indices[i]
                window_index = idx - (self.cumulative_window_counts[i-1] if i > 0 else 0)
                break
        
        # 获取样本文本
        text = self.data[sample_index]
        # 对文本进行分词
        tokens = self.tokenizer(text)
    
        # 计算滑动窗口的起始位置
        start_index_x = window_index
        end_index_x = window_index + self.seq_len
        
        start_index_y = window_index + 1
        end_index_y = window_index + self.seq_len + 1

        # 截取滑动窗口
        x = tokens[start_index_x:end_index_x]
        y = tokens[start_index_y:end_index_y]
        
        # 如果滑动窗口长度小于seq_len，则进行填充
        if len(x) < self.seq_len:
            x = x + ['<pad>'] * (self.seq_len - len(x))
        if len(y) < self.seq_len:
            y = y + ['<pad>'] * (self.seq_len - len(y))

        # 将分词结果转换为词汇表索引
        x = [self.vocab[token] if token in self.vocab else self.vocab['<unk>'] for token in x]
        y = [self.vocab[token] if token in self.vocab else self.vocab['<unk>'] for token in y]

        # 将索引转换为PyTorch张量
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        return x, y

# 测试代码
# if __name__ == '__main__':
#     seed = 42
#     data_dir='data/wiki_zh1'
#     vocab_path = 'check_points/wiki_zh1/vocab.txt'
    
#     set_seed(seed)

#     dataset = SDataset(data_dir=data_dir, seq_len=16, tokenizer_type='basic_chinese', usage_percentage=1.0,
#                            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, mode='train', vocab_path=vocab_path)
#     print(len(dataset))
    
#     for i in range(10):
#         x, y = dataset[i]
#         print(x)
#         print(y)
#         print()