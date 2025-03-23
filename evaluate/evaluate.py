import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.profiler import profile, record_function, ProfilerActivity
from model.FGPT import FGPT
from utils.set_seed import set_seed
from tqdm import tqdm
# 设置随机种子
# set_seed(42)
# print(torch.cuda.is_available())

d_k = 64
d_v = 64
d_model = 256
d_ff = 1024
N = 6
H = 8

batch_size = 32
sql_len = 1024
model = FGPT(d_k, d_v, d_model, H, d_ff, N).to('cuda')
x = torch.randn(batch_size, sql_len, d_model).to('cuda')
y = torch.randn(batch_size, sql_len, d_model).to('cuda')
# 定义损失函数
criterion = torch.nn.MSELoss()
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True
) as prof:
    with record_function("FGPT"):
        # 批量计算
        for i in tqdm(range(0, sql_len, batch_size)):
            x_batch = x[:, i:i+batch_size, :]
            y_batch = y[:, i:i+batch_size, :]
            # 前向传播
            output = model(x_batch)
            # 计算损失
            loss = criterion(output, y_batch)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=16))