"""
MobileNetV3 Backbone for LGFF (Lightweight Geometric-Feature Fusion).

Fixes & Features:
- [Critical Fix] Correctly detects stride in InvertedResidual blocks with expansion layers.
  (Previously failed to detect stride=2 if it was in the 2nd convolution of the block).
- Robust stride/dilation modification for 3x3 and 5x5 kernels.
- Supports output_stride in {8,16,32}, with effective_output_stride recorded.
- Optional return_intermediate for low-level features (skip/multi-scale friendly).
"""
from __future__ import annotations

import warnings
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights
from torchvision.models.mobilenetv3 import InvertedResidual


class MobileNetV3Extractor(nn.Module):
    def __init__(
        self,
        arch: str = "small",
        output_stride: int = 8,
        pretrained: bool = True,
        freeze_bn: bool = True,
        return_intermediate: bool = False,
        low_level_index: int = 2,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            arch: 'small' or 'large'
            output_stride: target downsampling factor, in {8,16,32}
            pretrained: whether to load ImageNet pretrained weights
            freeze_bn: freeze BatchNorm (recommended for small batch sizes)
            return_intermediate:
                - False: forward -> high-level feature only
                - True:  forward -> (high-level, low-level)
            low_level_index:
                - index in self.features from which to grab low-level feature
            verbose:
                - whether to print modification logs
        """
        super().__init__()
        self.arch = arch
        self.return_intermediate = return_intermediate
        self.low_level_index = low_level_index
        self.verbose = verbose

        # 1. Output Stride Check
        if output_stride not in (8, 16, 32):
            warnings.warn(
                f"[MobileNetV3Extractor] Unsupported output_stride={output_stride}, "
                "only {8,16,32} are recommended. Falling back to 8.",
                RuntimeWarning,
            )
            output_stride = 8
        self.target_os = output_stride

        # 2. Load Base Model
        if arch == "small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            base_model = models.mobilenet_v3_small(weights=weights)
        elif arch == "large":
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            base_model = models.mobilenet_v3_large(weights=weights)
        else:
            raise ValueError(f"[MobileNetV3Extractor] Unknown arch: {arch}")

        self.features = base_model.features

        # 简单防御：low_level_index 范围检查
        if self.return_intermediate and self.low_level_index >= len(self.features):
            raise ValueError(
                f"[MobileNetV3Extractor] low_level_index={self.low_level_index} "
                f"out of range (len(features)={len(self.features)})"
            )

        # 3. Modify Structure for Dilation
        self._modify_structure_for_dilation(self.features, self.target_os, verbose=self.verbose)

        # 4. Get Output Channels
        self.out_channels = self._get_out_channels()

        # 5. Freeze BN
        if freeze_bn:
            self._freeze_bn()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _get_out_channels(self) -> int:
        """Run a dummy forward to get final channel dimension."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.features(dummy)
        return out.shape[1]

    def _freeze_bn(self) -> None:
        """Freeze all BatchNorm layers."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)

    def _get_layer_stride(self, layer: nn.Module) -> int:
        """
        [关键修复] 遍历 layer 内部所有 Conv2d，检测是否存在下采样。

        InvertedResidual 结构示例：
            1x1 expand (stride=1)
            3x3 / 5x5 depthwise (stride=1 or 2)
            1x1 project (stride=1)

        如果只看第一个卷积，会漏掉 depthwise 的 stride=2。
        当前逻辑：只要模块内部有任意 Conv2d 的 stride>1，则认为这一层是下采样层。
        """
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                if m.stride[0] > 1:
                    return m.stride[0]
        return 1

    def _modify_structure_for_dilation(
        self,
        features: nn.Sequential,
        target_os: int,
        verbose: bool = False
    ) -> None:
        """
        主逻辑：达到 target_os 以后，禁止进一步下采样，改用 dilation 扩大感受野。
        """
        current_os = 1
        current_dilation = 1

        if verbose:
            print(f"[MobileNetV3Extractor] Modifying structure for target OS={target_os}...")

        for i, layer in enumerate(features):
            # 1. 探测该层 stride
            layer_stride = self._get_layer_stride(layer)

            # 2. 修改逻辑
            if current_os >= target_os:
                # 已经达到目标 OS，后续不再下采样
                if layer_stride == 2:
                    self._set_module_stride(layer, 1)
                    current_dilation *= 2

                # 为保持感受野，应用当前累积的 dilation
                if current_dilation > 1:
                    self._set_module_dilation(layer, current_dilation)
            else:
                # 还没达到目标 OS，允许正常下采样
                if layer_stride == 2:
                    current_os *= 2

        # 记录实际输出步长
        self.effective_output_stride = current_os

        if verbose:
            print(
                f"[MobileNetV3Extractor] Modification done. "
                f"effective_output_stride={self.effective_output_stride}"
            )

    # 静态辅助函数：修改 stride/dilation
    @staticmethod
    def _set_module_stride(module: nn.Module, stride: int) -> None:
        """递归将模块内所有 strided conv 改为 stride=1。"""
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                if m.stride[0] > 1:
                    m.stride = (stride, stride)

    @staticmethod
    def _set_module_dilation(module: nn.Module, dilation: int) -> None:
        """递归设置 dilation 和 padding。"""
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                # 仅大核卷积 (k>1) 需要 dilation (3x3, 5x5 等)
                if m.kernel_size[0] > 1:
                    m.dilation = (dilation, dilation)
                    pad = (m.kernel_size[0] - 1) * dilation // 2
                    m.padding = (pad, pad)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Returns:
            - if return_intermediate == False:
                  feat_high: [B, C, H/os, W/os]
            - if return_intermediate == True:
                  (feat_high, feat_low)
        """
        if not self.return_intermediate:
            return self.features(x)

        feat_low = None
        out = x
        for i, layer in enumerate(self.features):
            out = layer(out)
            if i == self.low_level_index:
                feat_low = out

        # 防御：如果 low_level_index 太大（理论上在 __init__ 已经挡掉了）
        assert feat_low is not None, (
            f"[MobileNetV3Extractor] feat_low is None. low_level_index={self.low_level_index} "
            f"may be out of range (len(features)={len(self.features)})."
        )

        return out, feat_low


# ======================================================================
#  Test Block
# ======================================================================
if __name__ == "__main__":
    print("Testing MobileNetV3Extractor Robustness...")
    dummy_in = torch.randn(2, 3, 128, 128)

    # 1. Small, OS=8
    model = MobileNetV3Extractor(
        arch="small",
        output_stride=8,
        pretrained=False,
        return_intermediate=False,
        verbose=True,
    )
    out = model(dummy_in)
    print(f"[Small OS=8] Input: {dummy_in.shape} -> Output: {out.shape}")

    # 使用实际 effective_output_stride 做检查，避免写死 8
    expected_dim = 128 // model.effective_output_stride
    assert out.shape[2] == expected_dim, (
        f"Error: Expected spatial dim={expected_dim}, got {out.shape[2]} "
        f"(effective_output_stride={model.effective_output_stride})"
    )

    # 2. Large, OS=16 + intermediate
    model_large = MobileNetV3Extractor(
        arch="large",
        output_stride=16,
        pretrained=False,
        return_intermediate=True,
        low_level_index=2,
        verbose=True,
    )
    out_high, out_low = model_large(dummy_in)
    print(f"[Large OS=16] High: {out_high.shape}, Low: {out_low.shape}")
    expected_dim_large = 128 // model_large.effective_output_stride
    assert out_high.shape[2] == expected_dim_large, (
        f"Error: Expected spatial dim={expected_dim_large}, got {out_high.shape[2]} "
        f"(effective_output_stride={model_large.effective_output_stride})"
    )

    print("All tests passed! ✅")
