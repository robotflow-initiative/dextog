import math
import torch
from torch import nn
from torch.nn import init

class TimeEmbedding(nn.Module):

    def __init__(self, T: int, d_model: int, dim: int):
        super().__init__()
        emb = torch.arange(0, d_model, step = 2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim = -1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view((T, d_model))

        self.time_embedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.time_embedding(t)
        return emb
