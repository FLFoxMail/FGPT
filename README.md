# 🚀 手写高性能 GPT 实践

[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/your-repo/high-performance-gpt/blob/main/LICENSE)[![PyTorch Version](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## 项目亮点
- ​**从零实现**：自注意力 → 完整 GPT → 分布式训练（[训练过程可视化](docs/training_curve.gif)）
- ​**极致优化**：单卡吞吐量提升 ? 倍，显存占用降低 ?%
- ​**工业级实践**：支持混合精度训练 + 梯度检查点 + 模型并行
- ​**开箱即用**：提供预训练模型 ([下载链接](https://example.com/pretrained_models))

## 代办事项
[x] 实现GPT
[] 测评环境，测评指标
[] 单卡计算性能优化
[] 单卡显存性能优化
[] 多卡分布式训练优化
[] 其他优化

## 性能对比
| env_name | time_stamp | d_k | d_v | d_model | num_heads | d_diff | n_layer | batch_size | seq_length | Training Time (ms) | Training Throughput (SPS) | Prediction Time (ms) | Prediction Throughput (SPS) | Memory Usage (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 20250326_010026 | 64 | 64 | 256 | 8 | 64 | 2 | 32 | 128 | 1197.324 | 1252.793 | 327.084 | 1528.66 | 0.208 |
| base | 20250326_010034 | 128 | 128 | 512 | 8 | 128 | 4 | 32 | 128 | 3482.128 | 430.771 | 980.718 | 509.831 | 0.666 |
| base | 20250326_010052 | 256 | 256 | 1024 | 8 | 256 | 8 | 32 | 128 | 17373.476 | 86.339 | 5899.958 | 84.746 | 2.676 |


## 使用方法
1. 克隆仓库
```bash
git clone https://github.com/FLFoxMail/FGPT.git
cd FGPT
```
2. 创建虚拟环境
```bash
conda create -n fgpt python=3.12
conda activate fgpt
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

## 鸣谢
感谢以下工作对 GPT 模型的研究与实现，本项目基于这些工作进行了优化和扩展：
- [Transformers](https://github.com/huggingface/transformers)
- [大模型动力引擎——PyTorch性能与显存优化手册](http://www.tup.tsinghua.edu.cn/booksCenter/book_10581501.html#)