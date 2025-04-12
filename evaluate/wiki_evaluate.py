from torch.utils.data import Dataset
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator
# 定义 wiki 数据集
class WikiTextDataSet(Dataset):
    def __init__(self, text_data, vocab, seq_length):

        # 初始化函数，用于初始化TextData类
        self.text_data = text_data  # 文本数据
        self.vocab = vocab  # 词汇表
        self.seq_length = seq_length  # 序列长度
        
    def __len__(self):
        # 返回text_data的长度减去1，然后除以seq_length的值
        return (len(self.text_data) - 1) // self.seq_length
    
    def __getitem__(self, index):
        # 计算起始位置
        start = index * self.seq_length
        # 计算结束位置
        end = start + self.seq_length
        # 当 end 超过最大长度时，会取 text_data的最大长度
        text_seq = self.text_data[start:end]
        # 将 text_seq 转换为 input_ids
        input_ids = self.vocab(text_seq)
        # 将 text_data 的下一个字符转换为 target_ids
        target_ids = self.vocab(self.text_data[start + 1: end + 1])
        # 返回 input_ids 和 target_ids
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)
    
def preper_data():
    tokenizer = get_tokenizer('basic_english')
    train_iter = WikiText2('./data', split="train")
    all_tokens = [tokenizer(line) for line in train_iter]
    vocab = build_vocab_from_iterator(all_tokens, specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])
    
    train_iter 