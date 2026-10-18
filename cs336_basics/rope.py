from typing import Any

import torch
from einops import einsum
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) -> None:
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        assert d_k % 2 == 0

        inv_freq = 1.0 / self.theta**(torch.arange(0, d_k, 2, device = device, dtype = torch.float32)/d_k)
        positions = torch.arange(0, max_seq_len, device=device, dtype=torch.float32)
        freqs = einsum(inv_freq, positions, "d_model_d2, seq_len -> seq_len d_model_d2")

        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        # shape: (seq_len, d_model_d2)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def forward(self, x:torch.Tensor, token_positions:torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (..., seq_len, d_k)
            token_positions: Tensor of shape (..., seq_len) containing the token positions for each element in the batch.
        """
        
        
        x_odd = x[...,1::2]
        x_even = x[...,0::2]

        cos = self.cos_cached[token_positions].unsqueeze(-3)
        sin = self.sin_cached[token_positions].unsqueeze(-3)

        out_odd = sin * x_even + cos * x_odd
        out_even = cos * x_even - sin * x_odd

        out = torch.empty_like(x)
        out[..., 1::2] = out_odd
        out[..., 0::2] = out_even
        return out




