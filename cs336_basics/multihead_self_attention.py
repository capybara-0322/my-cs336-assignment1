from typing import Any

import torch
from torch import nn
from cs336_basics.linear import Linear
from cs336_basics.attention import scaled_dot_product_attention
from cs336_basics.rope import RotaryPositionalEmbedding
from einops import rearrange


class MultiheadSelfAttention(nn.Module):
    def __init__(self, 
                d_model: int,
                num_heads: int, 
                use_rope = False, 
                max_seq_len = None,
                theta = None,
                device = None,
                dtype = None
                ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model%num_heads == 0
        self.d_k = d_model//num_heads
        self.d_v = d_model//num_heads
        self.device = device
        self.dtype = dtype
        self.use_rope = use_rope
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.wq = Linear(self.d_model, self.d_k*self.num_heads,device=device,dtype=dtype)
        self.wk = Linear(self.d_model, self.d_k*self.num_heads,device=device,dtype=dtype)
        self.wv = Linear(self.d_model, self.d_k*self.num_heads,device=device,dtype=dtype)
        self.wo = Linear(self.d_k*self.num_heads, self.d_model,device=device,dtype=dtype)
        if use_rope == True:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device)
        else:
            self.rope = None

    def generate_mask(self, dim: int):
        return torch.tril(torch.ones(dim,dim,dtype=torch.bool,device=self.device))

    

    def forward(self, x, token_positions = None):
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)
        Q = rearrange(Q, "... t (h d) -> ... h t d", h = self.num_heads)
        K = rearrange(K, "... t (h d) -> ... h t d", h = self.num_heads)
        V = rearrange(V, "... t (h d) -> ... h t d", h = self.num_heads)
        mask = self.generate_mask(x.shape[-2])
        if self.use_rope == True:
            if token_positions is None:
                token_positions = torch.arange(x.shape[-2], device=x.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)


        out = scaled_dot_product_attention(K,Q,V,mask)
        out = rearrange(out, "... h t d -> ... t (h d)")
        return self.wo(out)

        

        