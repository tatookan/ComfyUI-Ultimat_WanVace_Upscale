"""
WAN2.2 MoE专用采样器
"""

import torch
from comfy.samplers import KSampler
from .wan22.config import MoEConfig

class KSamplerWithMoE(KSampler):
    """WAN2.2 MoE专用采样器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.moe_config = MoEConfig()
        
    def sample(self, noise, positive, negative, **kwargs):
        # 阶段1：高噪声粗采样
        high_noise_result = self._high_noise_stage(noise, positive, negative, **kwargs)
        
        # 阶段2：低噪声精修
        final_result = self._low_noise_stage(high_noise_result, positive, negative, **kwargs)
        
        return final_result
        
    def _high_noise_stage(self, noise, positive, negative, **kwargs):
        """高噪声采样阶段"""
        steps = int(kwargs.get('steps', 20) * self.moe_config.high_noise_ratio)
        kwargs['steps'] = steps
        return super().sample(noise, positive, negative, **kwargs)
        
    def _low_noise_stage(self, noise, positive, negative, **kwargs):
        """低噪声采样阶段"""
        steps = kwargs.get('steps', 20) - int(kwargs.get('steps', 20) * self.moe_config.high_noise_ratio)
        kwargs['steps'] = steps
        return super().sample(noise, positive, negative, **kwargs)
