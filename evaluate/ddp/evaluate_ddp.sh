#!/bin/bash

# 定义 Python 脚本路径，该脚本用于进行模型性能基准测试
PYTHON_SCRIPT="evaluate/evaluate_ddp.py"

# 定义不同的参数组合数组，每个元素代表一组参数，参数顺序为：
# d_k d_v d_model num_heads d_diff n_layer batch_size seq_length
param_combinations=(
    "64 64 256 8 64 2 32 128"
    "128 128 512 8 128 4 32 128"
    "256 256 1024 8 256 8 32 128"
    "256 256 1024 8 256 16 32 128"
    "256 256 1024 8 256 32 32 128"
    "256 256 1024 8 256 64 32 128"
)

# 遍历参数组合数组，对每一组参数执行一次基准测试
for combination in "${param_combinations[@]}"; do
    # 将当前参数组合按空格分割为单独的参数，并存储到数组 params 中
    IFS=' ' read -r -a params <<< "$combination"
    
    # 从 params 数组中提取各个参数
    d_k="${params[0]}"
    d_v="${params[1]}"
    d_model="${params[2]}"
    num_heads="${params[3]}"
    d_diff="${params[4]}"
    n_layer="${params[5]}"
    batch_size="${params[6]}"
    seq_length="${params[7]}"

    # 输出当前正在使用的参数组合，方便调试和监控进度
    echo "Running benchmark with parameters: d_k=$d_k, d_v=$d_v, d_model=$d_model, num_heads=$num_heads, d_diff=$d_diff, n_layer=$n_layer, batch_size=$batch_size, seq_length=$seq_length"
    
    # 运行 Python 脚本并传递参数，执行模型性能基准测试
    python3 $PYTHON_SCRIPT $d_k $d_v $d_model $num_heads $d_diff $n_layer $batch_size $seq_length
    
    # 输出测试完成信息，标记一组参数的测试结束
    echo "Benchmark completed for the current parameter combination."
done
    