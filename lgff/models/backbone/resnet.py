"""
ResNet Backbone for LGFF.
Wraps torchvision ResNet with dilated convolutions for dense prediction.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights


class ResNetExtractor(nn.Module):
    """
    ResNet 特征提取器，修改了 stride/dilation 以支持 output_stride=8。
    """

    def __init__(
            self,
            arch: str = "resnet34",
            output_stride: int = 8,
            pretrained: bool = True,
            freeze_bn: bool = True,
    ) -> None:
        super().__init__()

        # 1. 加载预训练模型
        if arch == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            orig_model = models.resnet18(weights=weights)
            self.out_channels = 512
        elif arch == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            orig_model = models.resnet34(weights=weights)
            self.out_channels = 512
        elif arch == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            orig_model = models.resnet50(weights=weights)
            self.out_channels = 2048
        else:
            raise ValueError(f"Unknown ResNet arch: {arch}")

        # 2. 提取层级结构
        # ResNet 结构: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4
        self.conv1 = orig_model.conv1
        self.bn1 = orig_model.bn1
        self.relu = orig_model.relu
        self.maxpool = orig_model.maxpool
        self.layer1 = orig_model.layer1
        self.layer2 = orig_model.layer2
        self.layer3 = orig_model.layer3
        self.layer4 = orig_model.layer4

        # 3. 修改 Stride 和 Dilation 以适配 Output Stride
        # Layer1: OS=4 (Default)
        # Layer2: OS=8 (Default)
        # Layer3: OS=16 -> 改为 OS=8 (stride=1, dilation=2)
        # Layer4: OS=32 -> 改为 OS=8 (stride=1, dilation=4)

        if output_stride == 8:
            self._make_dilated(self.layer3, stride=1, dilation=2)
            self._make_dilated(self.layer4, stride=1, dilation=4)
        elif output_stride == 16:
            self._make_dilated(self.layer4, stride=1, dilation=2)

        # 4. 冻结 BN
        if freeze_bn:
            self._freeze_bn()

    def _make_dilated(self, layer: nn.Sequential, stride: int, dilation: int) -> None:
        """将 ResNet Layer 的下采样改为 dilation"""
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                m.stride = (1, 1) if m.stride == (2, 2) else m.stride  # 取消下采样
                if m.kernel_size == (3, 3):
                    m.dilation = (dilation, dilation)
                    m.padding = (dilation, dilation)  # 保持尺寸
            elif isinstance(m, nn.Sequential) and hasattr(m, 'downsample'):
                # 处理 Bottleneck/BasicBlock 的 downsample 分支
                if m.downsample is not None:
                    for sub_m in m.downsample.modules():
                        if isinstance(sub_m, nn.Conv2d) and sub_m.stride == (2, 2):
                            sub_m.stride = (1, 1)

        # 重新修正 ResNet Block 内部逻辑 (更严谨的做法是替换 stride，但上述暴力修改通常有效)
        # 对于 torchvision resnet，layer[0] 是下采样块。
        layer[0].conv1.stride = (stride, stride)
        layer[0].conv2.dilation = (dilation, dilation)
        layer[0].conv2.padding = (dilation, dilation)
        if hasattr(layer[0], 'downsample') and layer[0].downsample is not None:
            layer[0].downsample[0].stride = (stride, stride)

    def _freeze_bn(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x