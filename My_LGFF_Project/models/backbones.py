# lgff/models/backbones.py
"""目标：为 LGFF 提供统一的 backbone 构建入口
要求 Codex：

在 FFB6D 项目中查找 resnet, backbone 相关模块，看能否直接重用 ResNet18/34；

确保返回最后一层 feature map 的 shape 和 LGFFBase 预期一致
"""
import torch.nn as nn

# TODO: 按需从 FFB6D 或其他项目里引 MobileNet / ResNet / 轻量 YOLO block

def build_backbone(name: str, **kwargs) -> nn.Module:
    """
    Args:
        name: 'resnet18', 'mobilenet_v2', 'lgff_tiny' ...
    Returns:
        nn.Module: CNN backbone, 输出特征 F_rgb [B,C,H',W']
    """
    if name == "resnet18":
        # TODO: 从 ffb6d.models.resnet 导入并适配
        ...
    elif name == "mobilenet_v2":
        # TODO: 按需实现或导入
        ...
    else:
        raise NotImplementedError
