import torch
from torch import nn
from einops import reduce

def cross_entropy(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x = input
    max_x = reduce(x, "... vocab_size -> ... 1", "max")
    x = x - max_x
    log_sum_x = torch.log(reduce(torch.exp(x), "... vocab_size -> ... 1", "sum"))
    correct_x = torch.gather(x, dim=-1, index = target.unsqueeze(-1))
    loss = log_sum_x - correct_x
    return loss.mean()






