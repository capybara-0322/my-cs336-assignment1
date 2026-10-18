from typing import Any

import torch.nn as nn
import torch
from einops import rearrange, einsum

class Linear(nn.Module):

    def __init__(self, in_features, out_features, device = None, dtype = None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty((out_features,in_features),device = device, dtype = dtype))
        std = (2/(in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight,mean=0.0,std=std, a=-3 * std, b=3*std)


    @classmethod
    def from_weight(cls, weight: torch.Tensor):
        d_out, d_in = weight.shape
        model = cls(d_in,d_out,weight.device,weight.dtype)
        with torch.no_grad():
            model.weight.copy_(weight)

        return model



    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "out input,... input -> ... out")
