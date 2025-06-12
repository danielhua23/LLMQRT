import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        inv = torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states * inv
        return self.weight * hidden_states.to(input_dtype)    
