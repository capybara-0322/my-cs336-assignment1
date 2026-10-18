import torch
from torch import nn
from cs336_basics.softmax import softmax
from einops import einsum
import math

def scaled_dot_product_attention(keys: torch.Tensor, queries: torch.Tensor, values: torch.Tensor, mask: None | torch.Tensor = None) -> torch.Tensor:
    qk = einsum(queries,keys,"... seq1 dk, ... seq2 dk -> ... seq1 seq2")
    qk = qk / math.sqrt(keys.shape[-1])
    if mask is not None:
        qk = qk.masked_fill(~mask, -torch.inf)
    weight = softmax(qk,dim = -1)
    return einsum(weight, values, "... seq1 seq2, ... seq2 dv -> ... seq1 dv")


