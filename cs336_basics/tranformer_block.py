from typing import Any

import torch
from torch import nn
from cs336_basics.multihead_self_attention import MultiheadSelfAttention
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import SwiGLU
from cs336_basics.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, device = None, dtype = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn = MultiheadSelfAttention(self.d_model, num_heads, use_rope=True, 
                                                max_seq_len = max_seq_len, theta=theta,
                                                device=device, dtype=dtype)
        self.ln1 = RMSNorm(self.d_model, 1e-5, device=device, dtype=dtype)
        self.ffn = SwiGLU(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNorm(self.d_model, 1e-5, device=device, dtype=dtype)


    def forward(self, x):
        y1 = x + self.attn(self.ln1(x))
        y2 = y1 + self.ffn(self.ln2(y1))
        return y2

        









