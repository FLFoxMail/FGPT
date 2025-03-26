# 🚀 手写高性能 GPT 实践

[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/your-repo/high-performance-gpt/blob/main/LICENSE)[![PyTorch Version](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## 项目亮点
- ​**从零实现**：自注意力 → 完整 GPT → 分布式训练（[训练过程可视化](docs/training_curve.gif)）
- ​**极致优化**：单卡吞吐量提升 ? 倍，显存占用降低 ?%
- ​**工业级实践**：支持混合精度训练 + 梯度检查点 + 模型并行
- ​**开箱即用**：提供预训练模型 ([下载链接](https://example.com/pretrained_models))

## 代办事项
[x] 实现GPT
[x] 测评环境，测评指标
[] 单卡计算性能优化
[] 单卡显存性能优化
[] 多卡分布式训练优化
[] 其他优化

## 性能对比

- 单卡环境：
| env_name | Parameter Count (B) | Training Time (ms) | Training Throughput (SPS) | Prediction Time (ms) | Prediction Throughput (SPS) | Memory Usage (GB) |
| --- | --- | --- | --- | --- | --- | --- |6|
| base | 0.001 | 209.825 | 7148.8 | 53.019 | 9430.532 | 0.208|
| base | 0.009 | 540.858 | 2773.371 | 154.5 | 3236.25 | 0.665|
| base | 0.071 | 2794.232 | 536.82 | 903.021 | 553.697 | 2.676|
| base | 0.143 | 5807.015 | 258.308 | 1812.458 | 275.868 | 5.014|
| base | 0.286 | 11623.487 | 129.049 | 3642.887 | 137.254 | 9.689|

- 多卡环境：
| env_name | Parameter Count (B) | Training Time (ms) | Training Throughput (SPS) | Prediction Time (ms) | Prediction Throughput (SPS) | Memory Usage (GB) |
| --- | --- | --- | --- | --- | --- | --- |
| base | 0.001 | 604.663 | 413.453 | 85.774 | 2914.636 | 0.214|
| base | 0.009 | 1490.268 | 167.755 | 244.134 | 1024.028 | 0.699|
| base | 0.071 | 6532.981 | 38.267 | 1336.805 | 187.013 | 2.943|
| base | 0.143 | 12880.158 | 19.41 | 2664.64 | 93.821 | 5.546|
| base | 0.286 | 25467.03 | 9.817 | 5338.4 | 46.831 | 10.754|



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