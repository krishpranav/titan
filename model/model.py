'''
@filename: model.py
@date: 03-04-2025
@author: Krisna Pranav
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import GELUActivation

class ScaledRoPEAttention(nn.Module):
    def __init__(self, embed_dim, num_headers):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_headers = num_headers
        self.head_dim = embed_dim // num_headers
        assert self.embed_dim % self.num_headers == 0, "Embedding size must be divisible"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        