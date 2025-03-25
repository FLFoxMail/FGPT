import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.FGPT import FGPT
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import random
import numpy as np
import subprocess
from torch.utils.data import DataLoader, TensorDataset
from torch.profiler import profile, record_function, ProfilerActivity
import json
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel as DDP


# 固定随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 固定 GPU 频率
def set_gpu_clock(memory_clock, graphics_clock):
    try:
        subprocess.run(["nvidia-smi", "-ac", f"{memory_clock},{graphics_clock}"], check=True)
        print(f"GPU clock set to Memory: {memory_clock}, Graphics: {graphics_clock}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to set GPU clock: {e}")


# 动态生成 Markdown 表格
def generate_markdown_table(results):
    markdown = "| env_name | time_stamp | device_count | device_name | d_k | d_v | d_model | num_heads | d_diff | n_layer | batch_size | seq_length | Total Training Time (ms) | Training Throughput (SPS) | Total Prediction Time (ms) | Prediction Throughput (SPS) | Memory Usage (GB) |\n"
    markdown += "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    for result in results:
        markdown += f"| {result.get('env_name', 'N/A')} | {result.get('eval_time', 'N/A')} | {result.get('device_count', 'N/A')} | {result.get('device_names', 'N/A')} | {result.get('d_k', 'N/A')} | {result.get('d_v', 'N/A')} | {result.get('d_model', 'N/A')} | {result.get('num_heads', 'N/A')} | {result.get('d_diff', 'N/A')} | {result.get('n_layer', 'N/A')} | {result.get('batch_size', 'N/A')} | {result.get('seq_length', 'N/A')} | {result.get('total_train_time', 'N/A')} | {result.get('train_throughput', 'N/A')} | {result.get('total_pred_time', 'N/A')} | {result.get('pred_throughput', 'N/A')} | {result.get('memory_usage', 'N/A')} |\n"
    return markdown


class ModelBenchmark:
    def __init__(self, rank, world_size, model, batch_size, seq_length, d_model, device='cuda', seed=42):
        set_seed(seed)
        self.rank = rank
        self.world_size = world_size
        self.model = model.to(device)
        self.model = DDP(self.model, device_ids=[rank])
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.d_model = d_model
        self.device = device
        self.results = []
        self.fgpt_params = {
            'd_k': model.d_k,
            'd_v': model.d_v,
            'd_model': model.d_model,
            'num_heads': model.num_heads,
            'd_diff': model.d_diff,
            'n_layer': model.n_layer
        }

    def generate_random_data(self, num_samples):
        data = torch.randn(num_samples, self.seq_length, self.d_model).to(self.device)
        labels = torch.randint(0, 2, (num_samples, self.seq_length)).to(self.device)
        dataset = TensorDataset(data, labels)
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank)
        return DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)

    def train(self, num_epochs, num_samples, env_name, eval_time, device_names, device_count, num_rounds):
        dataloader = self.generate_random_data(num_samples)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        # 构建包含 FGPT 参数、batch_size 和 seq_length 的 trace 文件名称
        trace_train_name = f"evaluate/ddp/trace/{env_name}_{eval_time}_d_k_{self.fgpt_params['d_k']}_d_v_{self.fgpt_params['d_v']}_d_model_{self.fgpt_params['d_model']}_num_heads_{self.fgpt_params['num_heads']}_d_diff_{self.fgpt_params['d_diff']}_n_layer_{self.fgpt_params['n_layer']}_batch_size_{self.batch_size}_seq_length_{self.seq_length}_trace_train_{device_count}_{'_'.join(device_names)}_{self.rank}.json"

        # 使用 torch.profiler 记录训练时间
        with profile(
                activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
                on_trace_ready=lambda prof: prof.export_chrome_trace(trace_train_name),
                record_shapes=True,
                profile_memory=True,
        ) as prof:
            start_time = time.perf_counter()
            for t in range(num_rounds):
                total_samples = 0
                for epoch in range(num_epochs):
                    dataloader.sampler.set_epoch(epoch)
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
            train_time = (time.perf_counter() - start_time) * 1000  # 转换为毫秒
            # 保留三位小数
            train_throughput = total_samples / (train_time / 1000)

        # 记录显存占用
        memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 3)
        if self.rank == 0:
            print(
                f"Training time on GPU {self.rank}: {train_time:.3f} milliseconds, Memory usage: {memory_usage:.3f} GB, Training Throughput: {train_throughput:.3f} SPS")

        memory_usage = round(memory_usage, 3)
        # 存储结果
        result = {
            "eval_time": eval_time,
            "env_name": env_name,
            "device_count": device_count,
            "device_names": '_'.join(device_names),
            "train_time": train_time,
            "train_throughput": train_throughput,
            "memory_usage": memory_usage,
            "pred_time": None,
            "pred_throughput": None,
            **self.fgpt_params,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length
        }
        self.results.append(result)
        return train_time, total_samples

    def predict(self, num_samples, env_name, eval_time, device_names, device_count, num_rounds):
        dataloader = self.generate_random_data(num_samples)

        # 构建包含 FGPT 参数、batch_size 和 seq_length 的 trace 文件名称
        trace_predict_name = f"evaluate/ddp/trace/{env_name}_{eval_time}_d_k_{self.fgpt_params['d_k']}_d_v_{self.fgpt_params['d_v']}_d_model_{self.fgpt_params['d_model']}_num_heads_{self.fgpt_params['num_heads']}_d_diff_{self.fgpt_params['d_diff']}_n_layer_{self.fgpt_params['n_layer']}_batch_size_{self.batch_size}_seq_length_{self.seq_length}_trace_predict_{device_count}_{'_'.join(device_names)}_{self.rank}.json"

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
            pred_time = (time.perf_counter() - start_time) * 1000  # 转换为毫秒
            pred_throughput = total_samples / (pred_time / 1000)

        if self.rank == 0:
            print(
                f"Prediction time on GPU {self.rank}: {pred_time:.3f} milliseconds, Prediction Throughput: {pred_throughput:.3f} SPS")

        # 存储结果
        for result in self.results:
            if result["eval_time"] == eval_time:
                result["pred_time"] = pred_time
                result["pred_throughput"] = pred_throughput
                break
        return pred_time, total_samples


def run(rank, world_size):
    # 初始化进程组
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
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
    benchmark = ModelBenchmark(rank, world_size, model, batch_size=batch_size, seq_length=seq_length, d_model=d_model, device=rank)
    # 固定 GPU 频率
    if rank == 0:
        set_gpu_clock(memory_clock=5001, graphics_clock=2100)

    # 当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 当前环境
    env_name = "base"

    num_rounds = 3  # 多轮测试
    all_train_times = []
    all_train_samples = []
    all_pred_times = []
    all_pred_samples = []

    # 进行性能分析
    train_time, train_samples = benchmark.train(num_epochs=1, num_samples=500, env_name=env_name, eval_time=timestamp,
                                                device_count=world_size, device_names=[torch.cuda.get_device_name(i) for i in
                                                                                        range(torch.cuda.device_count())], num_rounds=num_rounds)
    pred_time, pred_samples = benchmark.predict(num_samples=500, env_name=env_name, eval_time=timestamp,
                                                device_count=world_size, device_names=[torch.cuda.get_device_name(i) for i in
                                                                                        range(torch.cuda.device_count())], num_rounds=num_rounds)
    all_train_times.append(train_time)
    all_train_samples.append(train_samples)
    all_pred_times.append(pred_time)
    all_pred_samples.append(pred_samples)

    # 同步所有进程
    dist.barrier()

    if rank == 0:
        # 汇总训练时间和样本数
        total_train_time = sum(all_train_times)
        total_train_samples = sum(all_train_samples)
        train_throughput = total_train_samples / (total_train_time / 1000)

        # 汇总预测时间和样本数
        total_pred_time = sum(all_pred_times)
        total_pred_samples = sum(all_pred_samples)
        pred_throughput = total_pred_samples / (total_pred_time / 1000)

        # 更新结果字典，只更新一次
        result = benchmark.results[0]

        total_train_time = round(total_train_time, 3)
        train_throughput = round(train_throughput, 3)
        pred_throughput = round(pred_throughput, 3)
        total_pred_time = round(total_pred_time, 3)

        result["total_train_time"] = total_train_time
        result["train_throughput"] = train_throughput
        result["total_pred_time"] = total_pred_time
        result["pred_throughput"] = pred_throughput
        benchmark.results = [result]

        # 保存结果
        def save_results(results):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 如果 文件不存在，则创建一个新文件并写入 []
            if not os.path.exists(f"evaluate/ddp/res/benchmark_results.json"):
                with open(f"evaluate/ddp/res/benchmark_results.json", "w") as f:
                    json.dump([], f, indent=4)
            with open(f"evaluate/ddp/res/benchmark_results.json", "r") as f:
                existing_results = json.load(f)
            existing_results.extend(results)
            with open(f"evaluate/ddp/res/benchmark_results.json", "w") as f:
                json.dump(existing_results, f, indent=4)
            markdown_table = generate_markdown_table(existing_results)
            with open(f"evaluate/ddp/res/benchmark_results_{timestamp}.md", "w") as f:
                f.write(markdown_table)
            print("Results saved and Markdown table generated.")

        save_results(benchmark.results)

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
    