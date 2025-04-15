import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import time
import random
import numpy as np
import json
from datetime import datetime
from torch.profiler import profile, record_function, ProfilerActivity
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
    markdown = "| device | dataset | model | Parameter Count (B) | Training Time (ms) | Prediction Time (ms) | Memory Usage (GB) |\n"
    markdown += "| --- | --- | --- | --- | --- | --- | --- |\n"
    for result in results:
        markdown += f"| {result['device']} | {result['dataset']} | {result['model_type']} | {result['parameter_count']} | {result['train_time']} | {result['pred_time']} | {result['memory_usage']}|\n"
    return markdown

class ModelBenchmark:
    def __init__(self, model, config, dataloaderFun):
        self.model = model.to(config.device)
        self.config = config
        self.result = {}
        self.dataloaderFun = dataloaderFun
        self.fgpt_params = {
            'd_k': model.d_k,
            'd_v': model.d_v,
            'd_model': model.d_model,
            'num_heads': model.num_heads,
            'd_diff': model.d_diff,
            'n_layer': model.n_layer
        }
        self.parameter_count = count_parameters(self.model)

    def getDataloader(self, *args, **kwargs):
        return self.dataloaderFun(*args, **kwargs)
    
    def train(self, dataSet):
        dataloader = self.getDataloader(self.config, dataSet=dataSet)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        # 构建包含 FGPT 参数、batch_size 和 seq_length 的 trace 文件名称
        trace_train_name = f"evaluate/single/trace/train_{self.config.device}_{self.config.dataset}_{self.config.model_type}_{self.config.eval_time}_d_k_{self.fgpt_params['d_k']}_d_v_{self.fgpt_params['d_v']}_d_model_{self.fgpt_params['d_model']}_num_heads_{self.fgpt_params['num_heads']}_d_diff_{self.fgpt_params['d_diff']}_n_layer_{self.fgpt_params['n_layer']}_batch_size_{self.config.batch_size}_seq_length_{self.config.seq_length}_trace_train.json"

        # 使用 torch.profiler 记录训练时间
        with profile(
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(
                wait=1, 
                warmup=1, 
                active=self.config.num_rounds,
                repeat=self.config.epochs),
            on_trace_ready=lambda prof: prof.export_chrome_trace(trace_train_name),
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            start_time = time.perf_counter()
            for batch_data, batch_labels in dataloader:
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
            
        calculated_train_time = (time.perf_counter() - start_time) * 1000 # 转换为毫秒
        calculated_train_time = round(calculated_train_time, 3)

        # 记录显存占用
        memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 3)
        memory_usage = round(memory_usage, 3)
        # 存储结果
        self.result = {
            "eval_time": self.config.eval_time,
            "device": self.config.device,
            "train_time": calculated_train_time,
            "memory_usage": memory_usage,
            "parameter_count": self.parameter_count,
            "batch_size": self.config.batch_size,
            "seq_length": self.config.seq_length,
            "model_type": self.config.model_type,
            "dataset": self.config.dataset,
        }
        return trace_train_name

    def predict(self, dataSet):
        dataloader = self.getDataloader(self.config, dataSet=dataSet)
        # 构建包含 FGPT 参数、batch_size 和 seq_length 的 trace 文件名称
        trace_predict_name = f"evaluate/single/trace/predict_{self.config.device}_{self.config.dataset}_{self.config.model_type}_{self.config.eval_time}_d_k_{self.fgpt_params['d_k']}_d_v_{self.fgpt_params['d_v']}_d_model_{self.fgpt_params['d_model']}_num_heads_{self.fgpt_params['num_heads']}_d_diff_{self.fgpt_params['d_diff']}_n_layer_{self.fgpt_params['n_layer']}_batch_size_{self.config.batch_size}_seq_length_{self.config.seq_length}_trace_predict.json"

        # 把模型切换到评估模式
        # 使用 torch.profiler 记录预测时间
        with profile(
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(
                wait=1, 
                warmup=1, 
                active=self.config.num_rounds,
                repeat=self.config.epochs),
            on_trace_ready=lambda prof: prof.export_chrome_trace(trace_predict_name),
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            start_time = time.perf_counter()
            with torch.no_grad():
                for batch_data, _ in dataloader:
                    batch_data = batch_data.to(self.config.device)
                    with record_function("predict"):
                        outputs = self.model(batch_data)
                    prof.step()
            torch.cuda.synchronize()
            pred_time = (time.perf_counter() - start_time) * 1000 / self.config.num_rounds  # 转换为毫秒


        pred_time = round(pred_time, 3)
        self.result["pred_time"] = pred_time
        return trace_predict_name

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
        return self.result