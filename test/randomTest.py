import random

random.seed(42)
print("Hash:", hash("hello"))

# glob 随机性测试
import glob

# 获取文件列表
files = glob.glob("/home/fl/code/python/FGPT/*.ipynb")

# 打印文件列表
print(files)