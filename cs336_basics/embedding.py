from torch import nn
import torch
from einops import rearrange, einsum


class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()

        self.vocab_size = num_embeddings
        self.d_model = embedding_dim
        self.weight = nn.Parameter(torch.empty((num_embeddings,embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, a=-3, b=3)

    @classmethod
    def from_weight(cls, weight:torch.Tensor):
        embedding_mode = cls(weight.shape[0], weight.shape[1])
        with torch.no_grad():
            embedding_mode.weight.copy_(weight)
        return embedding_mode
    

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

        
