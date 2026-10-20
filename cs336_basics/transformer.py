import torch
from torch import nn

from cs336_basics.embedding import Embedding
from cs336_basics.tranformer_block import TransformerBlock
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.linear import Linear




class TransformerLM(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, vocab_size, num_layers,device = None, dtype = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype

        self.token_embeddings = Embedding(self.vocab_size, self.d_model, self.device, self.dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(self.d_model, self.num_heads, self.d_ff, self.max_seq_len, self.theta, self.device, self.dtype)
            for _ in range(self.num_layers)
        ])
        self.ln_final = RMSNorm(self.d_model, 1e-5, self.device, self.dtype)
        self.lm_head = Linear(self.d_model, self.vocab_size)

    def forward(self, x):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
        
        
