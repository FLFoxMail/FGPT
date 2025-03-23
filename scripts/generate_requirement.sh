#!/bin/bash
set -eo pipefail

echo "===== 安全依赖扫描 ====="
cd "$(dirname "$0")/.." || { echo "路径错误！"; exit 1; }


# 安装依赖

# 安装最新版 pipreqs
pip install --upgrade pipreqs -i https://pypi.tuna.tsinghua.edu.cn/simple

# 构建参数数组
args=("." "--force") # --force 强制覆盖
args+=("--encoding" "utf-8" "--savepath" "requirements.tmp")

# 执行扫描命令 并配置代理
pipreqs "${args[@]}" --proxy http://127.0.0.1:7890

# 检查是否生成 requirements.txt
if [ ! -f "requirements.tmp" ]; then
  echo "生成失败！"
  exit 1
fi

# 移动临时文件
mv requirements.tmp requirements.txt

# # 添加 pytorch 官方源 -extra-index-url  https://download.pytorch.org/whl/124 和 阿里云源 - extra-index-url https://mirrors.aliyun.com/pypi/simple/
# if ! grep -q "^--index-url" requirements.txt; then
#   sed -i '1i--index-url https://download.pytorch.org/whl/cu124' requirements.txt
# fi

# 只给 torch==2.4.1 添加官方源
# 判断是否包含 torch==2.4.1
if grep -q "^torch==2.4.1$" requirements.txt; then
  # 判断是否包含 --index-url
  if ! grep -q "^--index-url" requirements.txt; then
    # 给 torch==2.4.1 所在行末尾添加 --index-url https://download.pytorch.org/whl/cu124
    sed -i '/^torch==2.4.1$/ s/$/ --index-url https:\/\/download.pytorch.org\/whl\/cu124/' requirements.txt
  fi
fi

echo "生成成功！"
cat requirements.txt