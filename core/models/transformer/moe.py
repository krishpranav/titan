from torch import nn
from torch.distrubted as dist

class MoELayer(nn.Module):
    def __init__(self, hidden_dim, num_experts=16, expert_capacity=64):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
        ])

    def forward(self, x):
        gates = F.softmax(self.get(x), dim=-1)
        expert_weights, expert_indicies = torch.topk(gates, k=2, dim=-1)