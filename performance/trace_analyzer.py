import json
import plotly.express as px

class TraceAnalyzer:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.events = []
        self.summary = {
            "total_events": 0,
            "categories": {},
            "top_ops_by_duration": [],
        }

        # ✅ 类别含义词典(你自己填中文注释)
        self.cat_meaning_dict = {
            "Trace": "PyTorch Profiler总的追踪区间",
            "ac2g": "Autograd到CUDA的通信桥接(用于记录张量使用情况)",
            "cpu_instant_event": "CPU上的瞬时事件(如张量shape、标记等)",
            "cpu_op": "在CPU上执行的PyTorch操作(如add、mm、view等)",
            "cuda_runtime": "CUDA运行时调用(如cudaMemcpy、kernel启动)",
            "fwdbwd": "forward/backward阶段的逻辑标识",
            "user_annotation": "用户通过record_function或profiler插入的注释事件"
        }

        self.name_meaning_dict = {
            "AccumulateGrad": "梯度累积节点，将梯度累加到参数上",
            "AddBackward0": "加法操作的反向传播",
            "AddmmBackward0": "线性层中矩阵乘加(add + matmul)的反向传播",
            "BmmBackward0": "批量矩阵乘法(batch matmul)的反向传播",
            "CatBackward0": "张量拼接(concatenate)的反向传播",
            "DivBackward0": "除法操作的反向传播",
            "EmbeddingBackward0": "Embedding层的反向传播",
            "ExpandBackward0": "张量expand操作的反向传播(广播维度)",
            "LogSoftmaxBackward0": "LogSoftmax操作的反向传播",
            "NativeLayerNormBackward0": "LayerNorm层的反向传播",
            "NllLossBackward0": "负对数似然损失函数的反向传播",
            "ReluBackward0": "ReLU激活函数的反向传播",
            "ReshapeAliasBackward0": "张量reshape/alias的反向传播",
            "SliceBackward0": "切片操作的反向传播",
            "ViewBackward0": "张量视图变换(view)的反向传播",
            "SumBackward0": "对张量进行求和操作的反向传播",
            "MmBackward0": "矩阵乘法(mm)的反向传播",
            "SoftmaxBackward0": "Softmax操作的反向传播",
            "MeanBackward0": "求平均(mean)操作的反向传播",
            "SubBackward0": "减法操作的反向传播",
            "ClampBackward0": "clamp限制张量范围的反向传播",
            "CopySlices": "张量部分复制操作",
            "aten::mm": "矩阵乘法操作(matrix multiplication)",
            "aten::addmm": "加法 + 矩阵乘法(线性层)",
            "aten::nll_loss_backward": "负对数似然损失函数的反向",
            "aten::_log_softmax_backward_data": "log_softmax的反向传播实现",
            "aten::sum": "对张量进行求和",
            "aten::native_layer_norm_backward": "LayerNorm的反向",
            "aten::threshold_backward": "ReLU激活函数的反向传播",
            "aten::reshape": "张量reshape操作",
            "aten::view": "张量视图转换(不复制)",
            "ProfilerStep#177": "Profiler记录的第177步训练",
            "PyTorch Profiler (0)": "Profiler跟踪的整体包裹区间",
            "Optimizer.step#Adam.step": "执行Adam优化器的参数更新",
            "Optimizer.zero_grad#Adam.zero_grad": "Adam优化器中将梯度归零"
        }

    def load_trace(self):
        with open(self.filepath, "r") as f:
            trace_data = json.load(f)
        self.events = trace_data.get("traceEvents", [])
        self.summary["total_events"] = len(self.events)

    def analyze(self, top_n=20, include_trace=False):
        ops = []
        for event in self.events:
            category = event.get("cat", "unknown")
            if not include_trace:
                # 如果category包含Trace，trace等字样，则忽略
                if "Trace" in category or "trace" in category or "stream" in category:
                    continue  # 忽略Trace事件

            name = event.get("name", "unknown")
            if not include_trace:
                # 如果name包含Profiler等字样，则忽略
                if "Profiler" in name or "profiler" in name or "stream" in name:
                    continue  # 忽略Trace事件
            duration = event.get("dur", 0)

            if category not in self.summary["categories"]:
                self.summary["categories"][category] = {"count": 0, "total_dur": 0}
            self.summary["categories"][category]["count"] += 1
            self.summary["categories"][category]["total_dur"] += duration

            ops.append({
                "name": name,
                "category": category,
                "duration": duration,
                "timestamp": event.get("ts", 0),
                "thread": event.get("tid", ""),
                "stream": event.get("args", {}).get("stream", "")
            })

        self.summary["top_ops_by_duration"] = sorted(ops, key=lambda x: x["duration"], reverse=True)[:top_n]

    def get_category_percentages(self):
        total = sum(c["total_dur"] for c in self.summary["categories"].values())
        return {
            cat: round(data["total_dur"] / total * 100, 2) if total else 0.0
            for cat, data in self.summary["categories"].items()
        }

    def plot_pie_by_category(self):
        cat_durations = {cat: data["total_dur"] for cat, data in self.summary["categories"].items()}
        total = sum(cat_durations.values())
        labels = list(cat_durations.keys())
        sizes = [v / total * 100 for v in cat_durations.values()]
        desc = self.cat_meaning_dict

        data = [
            {
                "类别": l,
                "占比": s,
                "总时长 (μs)": cat_durations[l],
                "含义": desc.get(l, '无解释')
            }
            for l, s in zip(labels, sizes)
        ]

        fig = px.pie(
            data,
            values="占比",
            names="类别",
            title="Total Time Share by Category (cat)",
            hover_data=["总时长 (μs)", "含义"]
        )
        fig.show()

    def plot_top_ops_bar(self):
        top_ops = self.summary["top_ops_by_duration"]
        data = []
        for op in top_ops:
            name = op["name"]
            base = name.split("::")[-1].split(":")[-1].strip()
            data.append({
                "操作": name,
                "耗时 (μs)": op["duration"],
                "类型": op["category"],
                "含义": self.name_meaning_dict.get(base, "(暂无定义)")
            })

        fig = px.bar(
            data,
            x="耗时 (μs)",
            y="操作",
            orientation="h",
            hover_data=["类型", "含义"],
            title="Top 20 Ops by Duration (name + cat)",
            height=600
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        fig.show()

    def analyze_and_plot_all(self, top_n=20):
        self.load_trace()
        self.analyze(top_n=top_n)
        self.plot_pie_by_category()
        self.plot_top_ops_bar()

# test
if __name__ == "__main__":
    analyzer = TraceAnalyzer("evaluate/single/trace/train_cuda_SDataset_SGPT_20250413_173944_d_k_128_d_v_128_d_model_128_num_heads_8_d_diff_1024_n_layer_2_batch_size_16_seq_length_1024_trace_train.json")
    analyzer.analyze_and_plot_all(top_n=20)