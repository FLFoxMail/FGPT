#!/bin/bash
set -eo pipefail

echo "===== 动态生成 requirements.txt ====="
cd "$(dirname "$0")/.." || exit 1

# 预定义版本-源映射字典（可自行扩展）
declare -A TORCH_SOURCES=(
    ["2.4.1"]="https://download.pytorch.org/whl/cu124"
    ["2.0.0"]="https://download.pytorch.org/whl/cu118"
    ["1.12.1"]="https://download.pytorch.org/whl/cu113"
    ["1.10.0"]="https://download.pytorch.org/whl/cu102"
    ["1.8.0"]="https://download.pytorch.org/whl/cu111"
)

# 生成原始 requirements.txt
pipreqs . --force --encoding utf-8 --savepath requirements.txt 2>/dev/null || {
    echo "生成失败！请检查依赖关系。"
    exit 1
}

# 逐行处理 torch 版本
grep "^torch==" requirements.txt | while IFS= read -r line; do
    version=$(echo "$line" | cut -d= -f3 | cut -d+ -f1)  # 提取纯净版本号（如 2.4.1）
    source_url="${TORCH_SOURCES[$version]}"
    
    if [[ -n "$source_url" ]]; then
        # 替换为带源的版本声明（保留原始行并追加源）
        sed -i "s|^$line$|$line --index-url $source_url|" requirements.txt
    else
        echo "警告: 未找到版本 $version 的预配置源，保留原始行。"
    fi
done

echo "生成成功！"
cat requirements.txt