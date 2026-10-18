from typing import Any
import torch
from torch import nn
from einops import rearrange, einsum, reduce



class RMSNorm(nn.Module):


    def __init__(self, d_model: int, eps: float, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.g = nn.Parameter(torch.empty([d_model], device=device, dtype=dtype))


    @classmethod
    def from_weight(cls, weight: torch.Tensor, eps):
        rmsnorm_model = cls(weight.shape[0],eps,weight.device,weight.dtype)
        with torch.no_grad():
            rmsnorm_model.g.copy_(weight)
        return rmsnorm_model
    

    def forward(self, x):
        in_type = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(reduce(x**2,"... d_model -> ... 1", "mean") + self.eps)
        result = (x/rms)*self.g
        return result.to(in_type)

    








