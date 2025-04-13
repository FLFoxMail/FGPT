import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.FGPT import FGPT
import torch
import time
import random
import numpy as np
import json
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from torch.profiler import profile, record_function, ProfilerActivity
from dataset.SDataset import SDataset
from dataset.FDataset import FDataset


# 固定随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 计算模型参数量
def count_parameters(model):
    # 计算模型参数量,并除以 1e9 得到以 B 为单位的参数量
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
    return round(param_count, 3)


# 动态生成 Markdown 表格
def generate_markdown_table(results):
    markdown = "| device | dataset | model | Parameter Count (B) | Training Time (ms) | Training Throughput (SPS) | Prediction Time (ms) | Prediction Throughput (SPS) | Memory Usage (GB) |\n"
    markdown += "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    for result in results:
        markdown += f"| {result['device']} | {result['dataset']} | {result['model_type']} | {result['parameter_count']} | {result['train_time']} | {result['train_throughput']} | {result['pred_time']} | {result['pred_throughput']} | {result['memory_usage']}|\n"
    return markdown


class Config:
    def __init__(self):
        # 设置默认参数
        self.d_k = 128
        self.d_v = 128
        self.d_model = 128
        self.num_heads = 8
        self.d_diff = 1024
        self.n_layer = 2
        self.batch_size = 16
        self.seq_length = 1024
        self.data_path = "data/wiki_zh1"
        self.vocab_path = "check_points/wiki_zh1/vocab.txt"
        self.dataset = "SDataset"
        self.model_type = "SGPT"
        self.usage_percentage = 0.01
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        self.num_epochs = 1
        self.num_samples = 500
        self.device = "cuda"
        self.eval_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.num_rounds = 3
        self.vocab_size = 0


    def parse_args(self):
        if len(sys.argv) == 12:
            self.d_k = int(sys.argv[1])
            self.d_v = int(sys.argv[2])
            self.d_model = int(sys.argv[3])
            self.num_heads = int(sys.argv[4])
            self.d_diff = int(sys.argv[5])
            self.n_layer = int(sys.argv[6])
            self.batch_size = int(sys.argv[7])
            self.seq_length = int(sys.argv[8])
            self.data_path = str(sys.argv[9])
            self.dataset = str(sys.argv[10])
            self.model_type = str(sys.argv[11])
        elif len(sys.argv) != 1:
            print("Usage: python model_benchmark.py [d_k d_v d_model num_heads d_diff n_layer batch_size seq_length data_path dataset model_type]")
            sys.exit(1)

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
class ModelBenchmark:
    def __init__(self, model, config):
        set_seed(42)
        self.model = model.to(config.device)
        self.config = config
        self.result = {}
        self.fgpt_params = {
            'd_k': model.d_k,
            'd_v': model.d_v,
            'd_model': model.d_model,
            'num_heads': model.num_heads,
            'd_diff': model.d_diff,
            'n_layer': model.n_layer
        }
        self.parameter_count = count_parameters(self.model)

    def getdDtaloader(self, mode):
        if self.config.dataset == 'SDataset':
            dataSet = SDataset(self.config.data_path, tokenizer_type='basic_chinese', 
                               seq_len=self.config.seq_length, usage_percentage=self.config.usage_percentage,
                               train_ratio=self.config.train_ratio, val_ratio=self.config.val_ratio, 
                               test_ratio=self.config.test_ratio, mode=mode, vocab_path='./data/vocab.txt')
        elif self.config.dataset == 'FDataset':
            dataSet = FDataset(self.config.data_path, tokenizer_type='basic_chinese')
        else:
            raise ValueError(f"Invalid dataset name: {self.config.dataset}")
        dataloader = DataLoader(dataSet, batch_size=self.config.batch_size, shuffle=True)
        return dataloader

    def train(self):
        dataloader = self.getdDtaloader(mode='train')
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        # 构建包含 FGPT 参数、batch_size 和 seq_length 的 trace 文件名称
        trace_train_name = f"evaluate/single/trace/{self.config.device}_{self.config.dataset}_{self.config.model_type}_{self.config.eval_time}_d_k_{self.fgpt_params['d_k']}_d_v_{self.fgpt_params['d_v']}_d_model_{self.fgpt_params['d_model']}_num_heads_{self.fgpt_params['num_heads']}_d_diff_{self.fgpt_params['d_diff']}_n_layer_{self.fgpt_params['n_layer']}_batch_size_{self.config.batch_size}_seq_length_{self.config.seq_length}_trace_train.json"

        # 使用 torch.profiler 记录训练时间
        with profile(
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
            on_trace_ready=lambda prof: prof.export_chrome_trace(trace_train_name),
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            start_time = time.perf_counter()
            total_samples = 0
            for t in range(self.config.num_rounds):
                for epoch in range(self.config.num_epochs):
                    for batch_data, batch_labels in dataloader:
                        total_samples += batch_data.size(0)
                        # 移动数据
                        batch_data = batch_data.to(self.config.device)
                        batch_labels = batch_labels.to(self.config.device)
                        optimizer.zero_grad()
                        with record_function("forward"):
                            outputs = self.model(batch_data)
                        with record_function("loss"):
                            loss = criterion(outputs.view(-1, outputs.size(-1)), batch_labels.view(-1))
                        with record_function("backward"):
                            loss.backward()
                        with record_function("optimizer_step"):
                            optimizer.step()
                        prof.step()
            torch.cuda.synchronize()
            calculated_train_time = (time.perf_counter() - start_time) * 1000 / self.config.num_rounds  # 转换为毫秒
            train_throughput = total_samples / (calculated_train_time / 1000)

        # 记录显存占用
        memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Training time: {calculated_train_time:.3f} milliseconds, Memory usage: {memory_usage:.3f} GB, Training Throughput: {train_throughput:.3f} SPS")

        calculated_train_time = round(calculated_train_time, 3)
        train_throughput = round(train_throughput, 3)
        memory_usage = round(memory_usage, 3)
        # 存储结果
        self.result = {
            "eval_time": self.config.eval_time,
            "device": self.config.device,
            "train_time": calculated_train_time,
            "train_throughput": train_throughput,
            "memory_usage": memory_usage,
            "parameter_count": self.parameter_count,
            "batch_size": self.config.batch_size,
            "seq_length": self.config.seq_length,
            "model_type": self.config.model_type,
            "dataset": self.config.dataset,
        }

    def predict(self):
        dataloader = self.getdDtaloader(mode='test')

        # 构建包含 FGPT 参数、batch_size 和 seq_length 的 trace 文件名称
        trace_predict_name = f"evaluate/single/trace/{self.config.device}_{self.config.dataset}_{self.config.model_type}_{self.config.eval_time}_d_k_{self.fgpt_params['d_k']}_d_v_{self.fgpt_params['d_v']}_d_model_{self.fgpt_params['d_model']}_num_heads_{self.fgpt_params['num_heads']}_d_diff_{self.fgpt_params['d_diff']}_n_layer_{self.fgpt_params['n_layer']}_batch_size_{self.config.batch_size}_seq_length_{self.config.seq_length}_trace_predict.json"

        # 使用 torch.profiler 记录预测时间
        with profile(
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
            on_trace_ready=lambda prof: prof.export_chrome_trace(trace_predict_name),
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            start_time = time.perf_counter()
            for t in range(self.config.num_rounds):
                total_samples = 0
                with torch.no_grad():
                    for batch_data, _ in dataloader:
                        total_samples += batch_data.size(0)
                        with record_function("predict"):
                            outputs = self.model(batch_data)
                        prof.step()
            torch.cuda.synchronize()
            pred_time = (time.perf_counter() - start_time) * 1000 / self.config.num_rounds  # 转换为毫秒
            pred_throughput = total_samples / (pred_time / 1000)

        print(f"Prediction time: {pred_time:.3f} milliseconds, Prediction Throughput: {pred_throughput:.3f} SPS")

        pred_time = round(pred_time, 3)
        pred_throughput = round(pred_throughput, 3)

        self.result["pred_time"] = pred_time
        self.result["pred_throughput"] = pred_throughput

    def save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 如果 文件不存在，则创建一个新文件并写入 []
        if not os.path.exists(f"evaluate/single/res/{self.config.dataset}_{self.config.model_type}_benchmark_results.json"):
            with open(f"evaluate/single/res/{self.config.dataset}_{self.config.model_type}_benchmark_results.json", "w") as f:
                json.dump([], f, indent=4)
        with open(f"evaluate/single/res/{self.config.dataset}_{self.config.model_type}_benchmark_results.json", "r") as f:
            results = json.load(f)
        results.append(self.result)
        with open(f"evaluate/single/res/{self.config.dataset}_{self.config.model_type}_benchmark_results.json", "w") as f:
            json.dump(results, f, indent=4)
        markdown_table = generate_markdown_table(results)
        with open(f"evaluate/single/res/{self.config.dataset}_{self.config.model_type}_benchmark_results_{timestamp}.md", "w") as f:
            f.write(markdown_table)
        print("Results saved and Markdown table generated.")


if __name__ == "__main__":
    config = Config()
    config.parse_args()
    
    # 读取vocab.txt, 获取词汇表大小
    with open(config.vocab_path, "r") as f:
        vocab_size = len(f.readlines())
    config.vocab_size = vocab_size

    if config.model_type == "SGPT":
        from model.SGPT import SGPT
        model = SGPT(d_k=config.d_k, d_v=config.d_v, d_model=config.d_model, num_heads=config.num_heads, d_diff=config.d_diff, n_layer=config.n_layer, vocab_size=config.vocab_size, device=config.device, max_seq_len=config.seq_length)
    elif config.model_type == "FGPT":
        from model.FGPT import FGPT
        model = FGPT(d_k=config.d_k, d_v=config.d_v, d_model=config.d_model, num_heads=config.num_heads, d_diff=config.d_diff, n_layer=config.n_layer)
    else:
        print("Invalid model type. Please choose from 'SGPT' or 'FGPT'.")
        sys.exit(1)
    model = model.cuda()
    # 初始化测评类
    benchmark = ModelBenchmark(model, config)

    # 进行性能分析
    benchmark.train()
    benchmark.predict()

    # 保存结果
    benchmark.save_results()