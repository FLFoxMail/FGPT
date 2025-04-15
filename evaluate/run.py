

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from performance.trace_analyzer import TraceAnalyzer

from config.Config import Config
from ModelBenchmark import ModelBenchmark
from dataset.FDataset import FDataset
from dataset.SDataset import SDataset
from torch.utils.data import DataLoader

import numpy as np


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# 固定随机种子

def getDataset(config, mode):
    if config.dataset == 'SDataset':
        dataSet = SDataset(config.data_path, tokenizer_type='basic_chinese', 
                            seq_len=config.seq_length, usage_percentage=config.usage_percentage,
                            train_ratio=config.train_ratio, val_ratio=config.val_ratio, 
                            test_ratio=config.test_ratio, mode=mode, vocab_path='./data/vocab.txt')
    elif config.dataset == 'FDataset':
        dataSet = FDataset(config.data_path, tokenizer_type='basic_chinese')
    else:
        raise ValueError(f"Invalid dataset name: {config.dataset}")
    
    return dataSet

def get_model(config):
    if config.model_type == "SGPT":
        from model.SGPT import SGPT
        model = SGPT(d_k=config.d_k, d_v=config.d_v, d_model=config.d_model, num_heads=config.num_heads, d_diff=config.d_diff, n_layer=config.n_layer, vocab_size=config.vocab_size, device=config.device, max_seq_len=config.seq_length)
    elif config.model_type == "FGPT":
        from model.FGPT import FGPT
        model = FGPT(d_k=config.d_k, d_v=config.d_v, d_model=config.d_model, num_heads=config.num_heads, d_diff=config.d_diff, n_layer=config.n_layer)
    else:
        print("Invalid model type. Please choose from 'SGPT' or 'FGPT'.")
        sys.exit(1)
        
    return model

def getDataloader(config, dataSet):
    dataloader = DataLoader(dataSet, batch_size=config.batch_size, shuffle=True)
    return dataloader

def run():
    config = Config()
    config.parse_args()
    
    # 读取vocab.txt, 获取词汇表大小
    with open(config.vocab_path, "r") as f:
        vocab_size = len(f.readlines())
    config.vocab_size = vocab_size

    model = get_model(config).to(config.device)
    # 初始化测评类
    benchmark = ModelBenchmark(model, config, getDataloader)

    # 进行性能分析
    dataSet = getDataset(config, mode='train')
    train_trace_path = benchmark.train(dataSet)
    dataSet = getDataset(config, mode='test')
    predict_trace_path = benchmark.predict(dataSet)

    # 保存结果
    benchmark.save_results()
    
    return train_trace_path, predict_trace_path
if __name__ == '__main__':
    train_trace_path, predict_trace_path = run()
    TraceAnalyzer(train_trace_path).analyze_and_plot_all(top_n=20)
    