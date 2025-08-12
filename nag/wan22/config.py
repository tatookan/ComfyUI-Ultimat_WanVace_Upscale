"""
WAN2.2 MoE配置
"""

class MoEConfig:
    def __init__(self):
        self.num_experts = 8
        self.high_noise_ratio = 0.75
        self.expert_dropout = 0.1
        self.load_balancing = 0.01
