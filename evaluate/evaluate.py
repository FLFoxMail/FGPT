# import torch
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from torch.profiler import profile, record_function, ProfilerActivity
# from model.FGPT import FGPT
# from utils.set_seed import set_seed
# from tqdm import tqdm
# # 设置随机种子
# # set_seed(42)
# # print(torch.cuda.is_available())

# d_k = 64
# d_v = 64
# d_model = 256
# d_ff = 1024
# N = 6
# H = 8

# batch_size = 32
# sql_len = 1024
# model = FGPT(d_k, d_v, d_model, H, d_ff, N).to('cuda')
# x = torch.randn(batch_size, sql_len, d_model).to('cuda')
# y = torch.randn(batch_size, sql_len, d_model).to('cuda')
# # 定义损失函数
# criterion = torch.nn.MSELoss()
# # 定义优化器
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#             profile_memory=True,
#             record_shapes=True
# ) as prof:
#     with record_function("FGPT"):
#         # 批量计算
#         for i in tqdm(range(0, sql_len, batch_size)):
#             x_batch = x[:, i:i+batch_size, :]
#             y_batch = y[:, i:i+batch_size, :]
#             # 前向传播
#             output = model(x_batch)
#             # 计算损失
#             loss = criterion(output, y_batch)
#             # 反向传播
#             optimizer.zero_grad()
#             loss.backward()
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=16))
################################################################################
################################################################################
# import torch
# from torch.profiler import profile, record_function, ProfilerActivity

# # 定义一个简单的自定义模型（替代 ResNet-18）
# class SimpleModel(torch.nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)  # 输入通道3，输出通道16
#         self.relu = torch.nn.ReLU()
#         self.pool = torch.nn.MaxPool2d(2)                               # 下采样为 112x112
#         self.fc = torch.nn.Linear(16 * 112 * 112, 10)                       # 输出10类分类

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)       # 展平
#         x = self.fc(x)
#         return x

# # 创建模型并移动到 GPU
# model = SimpleModel().cuda()

# # 创建一个随机输入张量
# input_tensor = torch.randn(1, 3, 224, 224).cuda()

# # 使用 torch.profiler 进行性能分析
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#             #  schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
#             #  on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
#              record_shapes=True,
#              profile_memory=True,
#              with_stack=True) as prof:
#     with record_function("model_inference"):
#         # 执行模型推理
#         output = model(input_tensor)
        
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))


###############################################################################
###############################################################################
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
# 
model = models.resnet18().cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
              record_shapes=True,
              profile_memory=True,) as prof:
    with record_function("model_inference"):
        model(inputs)
        
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))