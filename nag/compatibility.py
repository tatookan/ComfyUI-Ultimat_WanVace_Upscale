"""
WAN版本兼容性适配器
"""

from comfy.ldm.wan.model import WanModel, VaceWanModel, Wan22Model, VaceWan22Model
from .samplers import KSamplerWithNAG
from .moe_sampler import KSamplerWithMoE
from .wan.model import NAGWanModel, NAGVaceWanModel
from .wan22.model import NAGWan22Model, NAGVaceWan22Model as NAGVaceWan22Model_

class WanCompatibilityAdapter:
    """WAN版本兼容性适配器"""
    
    VERSION_MAP = {
        "wan21": {
            "model_class": [WanModel, VaceWanModel],
            "sampler_class": KSamplerWithNAG,
            "adapter_class": {
                "base": NAGWanModel,
                "vace": NAGVaceWanModel
            }
        },
        "wan22": {
            "model_class": [Wan22Model, VaceWan22Model],
            "sampler_class": KSamplerWithMoE,
            "adapter_class": {
                "base": NAGWan22Model,
                "vace": NAGVaceWan22Model_
            }
        }
    }
    
    def __init__(self, model):
        self.model = model
        self.version = self.detect_version()
        self.config = self.VERSION_MAP[self.version]
        
    def detect_version(self):
        """自动检测WAN版本"""
        model_type = type(self.model.model.diffusion_model)
        if model_type in self.VERSION_MAP["wan22"]["model_class"]:
            return "wan22"
        elif model_type in self.VERSION_MAP["wan21"]["model_class"]:
            return "wan21"
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    def get_sampler(self, *args, **kwargs):
        """根据版本返回合适的采样器"""
        return self.config["sampler_class"](*args, **kwargs)
        
    def get_model_adapter(self, *args, **kwargs):
        """根据版本返回合适的模型适配器"""
        if isinstance(self.model.model.diffusion_model, (VaceWanModel, VaceWan22Model)):
            return self.config["adapter_class"]["vace"](*args, **kwargs)
        else:
            return self.config["adapter_class"]["base"](*args, **kwargs)
