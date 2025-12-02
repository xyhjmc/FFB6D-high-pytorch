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
        return total_params, total_params * 4 / (1024 ** 2)

    @staticmethod
    def _measure_flops(model, inputs):
        base_model = model.module if hasattr(model, "module") else model
        activities = [ProfilerActivity.CPU]
        if any(tensor.is_cuda for tensor in inputs.values() if isinstance(tensor, torch.Tensor)):
            activities.append(ProfilerActivity.CUDA)
        with torch.no_grad():
            with profile(activities=activities, record_shapes=True, with_flops=True) as prof:
                base_model(inputs)
        total_flops = sum(evt.flops for evt in prof.key_averages() if evt.flops is not None)
        return total_flops / 1e9

    def maybe_log(self, model, batch, stage="unknown"):
        if stage in self._logged_stages:
            return None

        device = next(model.parameters()).device
        prepared = self._prepare_inputs(batch, device)

        was_training = model.training
        model.eval()

        params, param_mb = self._count_parameters(model)
        try:
            gflops = self._measure_flops(model, prepared)
        except Exception as exc:  # pragma: no cover - profiler issues are not fatal
            gflops = None
            print(f"[ModelComplexity] Failed to compute FLOPs: {exc}")

        if was_training:
            model.train()

        self._logged_stages.add(stage)

        msg_parts = [
            f"[ModelComplexity] Stage: {stage}",
            f"Parameters: {params:,} ({param_mb:.2f} MB)",
        ]
        if gflops is not None:
            msg_parts.append(f"Forward GFLOPs: {gflops:.3f}")
        print(" | ".join(msg_parts))
        return {"params": params, "param_mb": param_mb, "gflops": gflops}
