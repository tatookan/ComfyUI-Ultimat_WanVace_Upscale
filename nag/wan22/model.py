"""
WAN2.2 MoE模型适配器
支持Mixture of Experts架构和双阶段采样
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import MethodType
from functools import partial

import comfy
from comfy.ldm.wan.model import Wan22Model, VaceWan22Model
from comfy.ldm.modules.attention import optimized_attention

from ..utils import nag, cat_context, check_nag_activation


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


class NAGWan22CrossAttention(nn.Module):
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


class NAGWan22Model(Wan22Model):
    """WAN2.2 MoE模型适配器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.moe_config = {
            'num_experts': 8,
            'high_noise_ratio': 0.75,
            'expert_dropout': 0.1,
            'load_balancing': 0.01
        }
        
    def forward(self, x, timestep, context, **kwargs):
        # 检测采样阶段
        noise_level = self.detect_noise_level(timestep)
        
        # 应用MoE适配
        if hasattr(self, 'moe_layers'):
            return self.moe_forward(x, timestep, context, noise_level, **kwargs)
        else:
            return super().forward(x, timestep, context, **kwargs)
    
    def detect_noise_level(self, timestep):
        """检测当前噪声水平"""
        # 基于timestep计算噪声水平
        max_timestep = 1000
        noise_level = timestep / max_timestep
        return noise_level
    
    def moe_forward(self, x, timestep, context, noise_level, **kwargs):
        """MoE前向传播"""
        # 动态替换注意力层
        self._setup_moe_attention(noise_level)
        
        # 执行标准前向
        output = super().forward(x, timestep, context, **kwargs)
        
        return output
    
    def _setup_moe_attention(self, noise_level):
        """设置MoE注意力"""
        for name, module in self.named_modules():
            if "cross_attn" in name:
                # 包装为MoE版本
                if not hasattr(module, '_moe_wrapped'):
                    module._original_forward = module.forward
                    module.forward = MethodType(
                        lambda self, *args, **kw: NAGWan22CrossAttention(
                            self, self.moe_config['num_experts']
                        ).forward(*args, noise_level=noise_level, **kw),
                        module
                    )
                    module._moe_wrapped = True
