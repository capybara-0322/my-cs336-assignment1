import torch
import random
from torch import nn
import numpy as np

def data_loading(dataset, batch_size, context_length, device = None):
    max_start = len(dataset) - context_length - 1
    starts = [random.randint(0, max_start) for _ in range(batch_size)]

    x_np = np.array([dataset[s: s + context_length] for s in starts])
    y_np = np.array([dataset[s+1 : s+context_length+1] for s in starts])

    x = torch.from_numpy(x_np).long().to(device)
    y = torch.from_numpy(y_np).long().to(device)

    return x, y

