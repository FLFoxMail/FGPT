import torch
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

def numpy_heavy_computation(x):
    size_inner = 100000
    res = x
    for _ in range(2):
        size_0 = res.shape[0]
        size_1 = res.shape[1]
        matrix_a = np.random.rand(size_0, size_inner)
        matrix_b = np.random.rand(size_inner, size_1)
        res = np.dot(matrix_a, matrix_b)
    
    return res

def run(data, model):
    processed_data = torch.from_numpy(data).float()
    numpy_heavy_computation(processed_data)
    data_tensor = torch.tensor(processed_data[:10,:10], dtype=torch.float32, device='cuda')
    output = model(data_tensor)
    return output

def main():
    model = SimpleModel().to('cuda')
    data = np.random.rand(100,100)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True
    ) as prof:
        for i in range(2):
            run(data, model)
    
    torch.cuda.synchronize()
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=16))
    prof.export_chrome_trace("evaluate/trace_without_numpy.json")
    
main()