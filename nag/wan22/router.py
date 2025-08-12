"""
MoE专家路由网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MoERouter(nn.Module):
    """MoE专家路由网络"""
    
    def __init__(self, dim, num_experts=8):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(dim, num_experts)
        self.noise_threshold = 0.5
        
    def forward(self, x, noise_level):
        """根据噪声水平选择专家"""
        gate_logits = self.gate(x.mean(dim=1))
        
        if noise_level > self.noise_threshold:
            # 高噪声：选择粗加工专家
            expert_weights = F.softmax(gate_logits, dim=-1)
            selected_expert = torch.argmax(expert_weights, dim=-1)
        else:
            # 低噪声：选择精修专家
            expert_weights = F.softmin(gate_logits, dim=-1)
            selected_expert = torch.argmin(expert_weights, dim=-1)
            
        return selected_expert, expert_weights
