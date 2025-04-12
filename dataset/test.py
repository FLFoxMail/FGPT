import os
import random
import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import jieba
import json
from functools import lru_cache


class LargeDataset(Dataset):
    def __init__(self, data_dir, seq_len, tokenizer_type='basic_chinese', usage_percentage=1.0,
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42, mode='train'):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.tokenizer_type = tokenizer_type
        self.usage_percentage = usage_percentage
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.mode = mode
        self.data_files = []
        self.vocab = self._build_vocabulary()
        self._load_data_files()
        self._random_sample_indices()
        self._split_indices()
        self.window_counts = []
        self._calculate_window_counts()

    def _load_data_files(self):
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self.data_files.append(file_path)

    def _build_vocabulary(self):
        all_tokens = []
        tokenizer = get_tokenizer('basic_chinese') if self.tokenizer_type == 'basic_english' else jieba.lcut
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            text = data.get('text', '')
                            tokens = tokenizer(text)
                            all_tokens.extend(tokens)
                        except json.JSONDecodeError:
                            continue

        def yield_tokens():
            for token in all_tokens:
                yield token

        vocab = build_vocab_from_iterator(yield_tokens(), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
        vocab.set_default_index(vocab['<unk>'])
        return vocab

    def _random_sample_indices(self):
        random.seed(self.random_seed)
        num_samples = int(len(self.data_files) * self.usage_percentage)
        self.sample_indices = random.sample(range(len(self.data_files)), num_samples)

    def _split_indices(self):
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1, "The sum of ratios should be 1."
        random.seed(self.random_seed)
        random.shuffle(self.sample_indices)
        total_samples = len(self.sample_indices)
        train_size = int(self.train_ratio * total_samples)
        val_size = int(self.val_ratio * total_samples)

        self.train_indices = self.sample_indices[:train_size]
        self.val_indices = self.sample_indices[train_size:train_size + val_size]
        self.test_indices = self.sample_indices[train_size + val_size:]

        if self.mode == 'train':
            self.indices = self.train_indices
        elif self.mode == 'val':
            self.indices = self.val_indices
        elif self.mode == 'test':
            self.indices = self.test_indices
        else:
            raise ValueError("Mode should be one of 'train', 'val', or 'test'.")

    def _calculate_window_counts(self):
        for index in self.indices:
            file_path = self.data_files[index]
            data_list = self._read_file(file_path)
            for data in data_list:
                text = data.get('text', '')
                tokenizer = get_tokenizer('basic_chinese') if self.tokenizer_type == 'basic_english' else jieba.lcut
                tokens = tokenizer(text)
                token_id = [self.vocab[token] for token in tokens]
                input_len = len(token_id)
                num_windows = max(1, input_len - self.seq_len + 1)
                self.window_counts.append(num_windows)
        self.cumulative_window_counts = [0] + [sum(self.window_counts[:i + 1]) for i in range(len(self.window_counts))]

    @lru_cache(maxsize=128)
    def _read_file(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return data

    def __len__(self):
        return self.cumulative_window_counts[-1]

    def __getitem__(self, idx):
        # 找到对应的文件索引和窗口索引
        file_index = next(i for i, count in enumerate(self.cumulative_window_counts) if count > idx) - 1
        window_index = idx - self.cumulative_window_counts[file_index]

        file_path = self.data_files[self.indices[file_index]]
        data_list = self._read_file(file_path)
        data = data_list[0]  # 假设每个文件只有一条数据，可根据实际情况修改
        text = data.get('text', '')
        tokenizer = get_tokenizer('basic_chinese') if self.tokenizer_type == 'basic_english' else jieba.lcut
        tokens = tokenizer(text)
        token_id = [self.vocab[token] for token in tokens]

        # 处理数组越界情况并补全
        input_len = len(token_id)
        num_windows = max(1, input_len - self.seq_len + 1)
        if num_windows < 1:
            padding_length = self.seq_len - input_len
            token_id += [self.vocab['<pad>']] * padding_length
            num_windows = 1

        start = window_index
        x = token_id[start:start + self.seq_len - 1]
        y = token_id[start + 1:start + self.seq_len]
        if len(x) < self.seq_len - 1:
            padding_length_x = (self.seq_len - 1) - len(x)
            x += [self.vocab['<pad>']] * padding_length_x
        if len(y) < self.seq_len - 1:
            padding_length_y = (self.seq_len - 1) - len(y)
            y += [self.vocab['<pad>']] * padding_length_y

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        return x, y