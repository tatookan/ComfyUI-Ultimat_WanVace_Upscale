"""
WAN2.2 MoE专用注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy.ldm.modules.attention import optimized_attention
from .router import MoERouter

class NAGWan22MoECrossAttention(nn.Module):
    """WAN2.2 MoE交叉注意力"""
    
    def __init__(self, base_attention, num_experts=8):
        super().__init__()
        self.base_attention = base_attention
        self.router = MoERouter(base_attention.dim, num_experts)
        self.expert_weights = nn.ParameterList([
            nn.Parameter(torch.randn(base_attention.dim, base_attention.dim) * 0.02)
            for _ in range(num_experts)
        ])
        
    def forward(self, x, context, noise_level, **kwargs):
        # 路由选择专家
        expert_idx, weights = self.router(x, noise_level)
        
        # 获取对应专家的权重
        expert_weight = self.expert_weights[expert_idx]
        
        # 应用专家权重
        modified_x = torch.matmul(x, expert_weight)
        
        # 执行基础注意力
        output = self.base_attention(modified_x, context, **kwargs)
        
        return output
