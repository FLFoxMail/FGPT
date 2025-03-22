import torch
import random
import numpy as np
import glob
import os

def set_seed(seed):
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 约束 GPU 算子随机性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False