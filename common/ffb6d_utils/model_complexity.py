"""网络复杂度统计工具，负责计算参数量与FLOPs等指标。"""
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile


class ModelComplexityLogger:
    """Utility to log parameter counts and forward GFLOPs once per stage."""

    def __init__(self):
        self._logged_stages = set()

    @staticmethod
    def _prepare_inputs(batch, device):
        prepared = {}
        for key, value in batch.items():
            tensor = None
            if isinstance(value, torch.Tensor):
                tensor = value
            elif isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value)
            else:
                continue

            if tensor.dtype in [torch.uint8, torch.float16, torch.float32, torch.float64]:
                tensor = tensor.float()
            elif tensor.dtype in [torch.int32, torch.int64, torch.int16, torch.int8]:
                tensor = tensor.long()
            prepared[key] = tensor.to(device, non_blocking=True)
        return prepared

    @staticmethod
    def _count_parameters(model):
        base_model = model.module if hasattr(model, "module") else model
        total_params = sum(p.numel() for p in base_model.parameters())
        # param_mb 只是一个粗略换算：假定 float32 * 4 bytes
        param_mb = total_params * 4 / (1024 ** 2)
        return total_params, param_mb

    @staticmethod
    def _infer_batch_size(inputs: dict) -> int:
        """
        尝试从输入字典中推断 batch size：
        取第一个形状 >=1 的 tensor 的 dim0 作为 batch_size。
        """
        for v in inputs.values():
            if isinstance(v, torch.Tensor) and v.ndim >= 1:
                return int(v.shape[0])
        return 1  # 实在推不出来就当 1

    @staticmethod
    def _measure_flops(model, inputs):
        """
        使用 torch.profiler 统计一次 forward 的总 FLOPs（当前 batch），
        并换算为：
          - gflops_total: 当前 batch 的总 GFLOPs
          - gflops_per_sample: 单样本 GFLOPs（gflops_total / batch_size）
        """
        base_model = model.module if hasattr(model, "module") else model

        # 推断 batch size（用于 per-sample 归一化）
        batch_size = ModelComplexityLogger._infer_batch_size(inputs)

        activities = [ProfilerActivity.CPU]
        if any(
            isinstance(t, torch.Tensor) and t.is_cuda
            for t in inputs.values()
        ):
            activities.append(ProfilerActivity.CUDA)

        with torch.no_grad():
            with profile(
                activities=activities,
                record_shapes=True,
                with_flops=True
            ) as prof:
                base_model(inputs)

        total_flops = sum(
            evt.flops for evt in prof.key_averages() if evt.flops is not None
        )

        gflops_total = total_flops / 1e9
        if batch_size > 0:
            gflops_per_sample = gflops_total / batch_size
        else:
            gflops_per_sample = gflops_total

        return gflops_total, gflops_per_sample, batch_size

    def maybe_log(self, model, batch, stage="unknown"):
        """
        只在每个 stage 第一次调用时打印：
          - 参数量（总参数 & 约等于 MB）
          - 当前 batch 的 forward 总 GFLOPs
          - 单样本 GFLOPs（per image），更接近 YOLO 报法
        """
        if stage in self._logged_stages:
            return None

        device = next(model.parameters()).device
        prepared = self._prepare_inputs(batch, device)

        was_training = model.training
        model.eval()

        params, param_mb = self._count_parameters(model)
        try:
            gflops_total, gflops_per_sample, batch_size = self._measure_flops(
                model, prepared
            )
        except Exception as exc:  # pragma: no cover - profiler issues are not fatal
            gflops_total, gflops_per_sample, batch_size = None, None, None
            print(f"[ModelComplexity] Failed to compute FLOPs: {exc}")

        if was_training:
            model.train()

        self._logged_stages.add(stage)

        msg_parts = [
            f"[ModelComplexity] Stage: {stage}",
            f"Parameters: {params:,} ({param_mb:.2f} MB)",
        ]
        if gflops_total is not None:
            msg_parts.append(
                f"Forward GFLOPs (batch) = {gflops_total:.3f}"
            )
        if gflops_per_sample is not None and batch_size is not None:
            msg_parts.append(
                f"Forward GFLOPs (per sample) = {gflops_per_sample:.3f} (B={batch_size})"
            )

        print(" | ".join(msg_parts))

        return {
            "params": params,
            "param_mb": param_mb,
            "gflops": gflops_total,
            "gflops_per_sample": gflops_per_sample,
            "batch_size": batch_size,
        }
