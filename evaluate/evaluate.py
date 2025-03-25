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


# 固定随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 动态生成 Markdown 表格
def generate_markdown_table(results):
    markdown = "| env_name | time_stamp | d_k | d_v | d_model | num_heads | d_diff | n_layer | batch_size | seq_length | Training Time (ms) | Training Throughput (SPS) | Prediction Time (ms) | Prediction Throughput (SPS) | Memory Usage (GB) |\n"
    markdown += "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    for result in results:
        markdown += f"| {result['env_name']} | {result['eval_time']} | {result['d_k']} | {result['d_v']} | {result['d_model']} | {result['num_heads']} | {result['d_diff']} | {result['n_layer']} | {result['batch_size']} | {result['seq_length']} | {result['train_time']} | {result['train_throughput']} | {result['pred_time']} | {result['pred_throughput']} | {result['memory_usage']} |\n"
    return markdown


class ModelBenchmark:
    def __init__(self, model, batch_size, seq_length, device='cuda', seed=42):
        set_seed(seed)
        self.model = model.to(device)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.device = device
        self.result = {}
        self.fgpt_params = {
            'd_k': model.d_k,
            'd_v': model.d_v,
            'd_model': model.d_model,
            'num_heads': model.num_heads,
            'd_diff': model.d_diff,
            'n_layer': model.n_layer
        }

    def generate_random_data(self, num_samples):
        data = torch.randn(num_samples, self.seq_length, self.fgpt_params['d_model']).to(self.device)
        labels = torch.randint(0, 2, (num_samples, self.seq_length)).to(self.device)
        return DataLoader(TensorDataset(data, labels), batch_size=self.batch_size, shuffle=True)

    def train(self, num_epochs, num_samples, env_name, eval_time, num_rounds):
        dataloader = self.generate_random_data(num_samples)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        # 构建包含 FGPT 参数、batch_size 和 seq_length 的 trace 文件名称
        trace_train_name = f"evaluate/single/trace/{env_name}_{eval_time}_d_k_{self.fgpt_params['d_k']}_d_v_{self.fgpt_params['d_v']}_d_model_{self.fgpt_params['d_model']}_num_heads_{self.fgpt_params['num_heads']}_d_diff_{self.fgpt_params['d_diff']}_n_layer_{self.fgpt_params['n_layer']}_batch_size_{self.batch_size}_seq_length_{self.seq_length}_trace_train.json"

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
            for t in range(num_rounds):
                for epoch in range(num_epochs):
                    for batch_data, batch_labels in dataloader:
                        total_samples += batch_data.size(0)
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
            calculated_train_time = (time.perf_counter() - start_time) * 1000 / num_rounds  # 转换为毫秒
            train_throughput = total_samples / (calculated_train_time / 1000)

        # 记录显存占用
        memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Training time: {calculated_train_time:.3f} milliseconds, Memory usage: {memory_usage:.3f} GB, Training Throughput: {train_throughput:.3f} SPS")

        calculated_train_time = round(calculated_train_time, 3)
        train_throughput = round(train_throughput, 3)
        memory_usage = round(memory_usage, 3)
        # 存储结果
        self.result = {
            "eval_time": eval_time,
            "env_name": env_name,
            "train_time": calculated_train_time,
            "train_throughput": train_throughput,
            "memory_usage": memory_usage,
            **self.fgpt_params,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length
        }

    def predict(self, num_samples, env_name, eval_time, num_rounds):
        dataloader = self.generate_random_data(num_samples)

        # 构建包含 FGPT 参数、batch_size 和 seq_length 的 trace 文件名称
        trace_predict_name = f"evaluate/single/trace/{env_name}_{eval_time}_d_k_{self.fgpt_params['d_k']}_d_v_{self.fgpt_params['d_v']}_d_model_{self.fgpt_params['d_model']}_num_heads_{self.fgpt_params['num_heads']}_d_diff_{self.fgpt_params['d_diff']}_n_layer_{self.fgpt_params['n_layer']}_batch_size_{self.batch_size}_seq_length_{self.seq_length}_trace_predict.json"

        # 使用 torch.profiler 记录预测时间
        with profile(
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
            on_trace_ready=lambda prof: prof.export_chrome_trace(trace_predict_name),
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            start_time = time.perf_counter()
            for t in range(num_rounds):
                total_samples = 0
                with torch.no_grad():
                    for batch_data, _ in dataloader:
                        total_samples += batch_data.size(0)
                        with record_function("predict"):
                            outputs = self.model(batch_data)
                        prof.step()
            torch.cuda.synchronize()
            pred_time = (time.perf_counter() - start_time) * 1000 / num_rounds  # 转换为毫秒
            pred_throughput = total_samples / (pred_time / 1000)

        print(f"Prediction time: {pred_time:.3f} milliseconds, Prediction Throughput: {pred_throughput:.3f} SPS")

        pred_time = round(pred_time, 3)
        pred_throughput = round(pred_throughput, 3)

        self.result["pred_time"] = pred_time
        self.result["pred_throughput"] = pred_throughput

    def save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 如果 文件不存在，则创建一个新文件并写入 []
        if not os.path.exists(f"evaluate/single/res/benchmark_results.json"):
            with open(f"evaluate/single/res/benchmark_results.json", "w") as f:
                json.dump([], f, indent=4)
        with open(f"evaluate/single/res/benchmark_results.json", "r") as f:
            results = json.load(f)
        results.append(self.result)
        with open(f"evaluate/single/res/benchmark_results.json", "w") as f:
            json.dump(results, f, indent=4)
        markdown_table = generate_markdown_table(results)
        with open(f"evaluate/single/res/benchmark_results_{timestamp}.md", "w") as f:
            f.write(markdown_table)
        print("Results saved and Markdown table generated.")


# 示例用法
if __name__ == "__main__":
    d_k = 128
    d_v = 128
    d_model = 128
    num_heads = 8
    d_diff = 1024
    n_layer = 16
    batch_size = 16
    seq_length = 64

    model = FGPT(d_k=d_k, d_v=d_v, d_model=d_model, num_heads=num_heads, d_diff=d_diff, n_layer=n_layer)

    # 初始化测评类
    benchmark = ModelBenchmark(model, batch_size, seq_length)

    # 当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 当前环境
    env_name = "base"

    num_rounds = 3  # 多轮测试

    # 进行性能分析
    benchmark.train(num_epochs=1, num_samples=500, env_name=env_name, eval_time=timestamp, num_rounds=num_rounds)
    benchmark.predict(num_samples=500, env_name=env_name, eval_time=timestamp, num_rounds=num_rounds)

    # 保存结果
    benchmark.save_results()
    