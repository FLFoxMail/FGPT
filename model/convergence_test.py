import torch
import torch.nn as nn
from model import FGPT

d_model = 768
d_k = 768
d_v = 768
batch_size = 2
num_heads = 8
seq_len = 1024
d_diff = 128
n_layer = 32
epoch = 20
lr = 0.05
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


x = torch.rand(batch_size, seq_len, d_model).to(device)
y = torch.rand(batch_size, seq_len, d_model).to(device)
memory_before = torch.cuda.memory_allocated()
model = FGPT(d_k, d_v, d_model, num_heads, d_diff, n_layer)
model = model.to(device)


total_params = sum(p.numel() for p in model.parameters())
print(f"总参数数量：{total_params/1e9:.2f} B")

criterion = nn.MSELoss()
optimaizer = torch.optim.Adam(model.parameters(), lr)


for i in range(epoch):
    # 清空梯度
    optimaizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimaizer.step()
    if i % 2 == 0:
        print(f"Epoch: {i + 1}, Loss: {loss.item()}")
        

        