"""
Benchmark script for LGFF Single-Class model (SC).

This tool measures:
  1) Single-batch forward time (ms per batch, ms per sample).
  2) Full-split throughput:
       - wall-clock time (includes DataLoader + model forward),
       - model-only time (only forward, averaged over all batches).

Usage examples:
    python lgff/bench_sc.py \
        --config lgff/configs/linemod_ape_sc_mnl_all.yaml \
        --checkpoint lgff/output/linemod_ape_sc_mnl_all/checkpoint_best.pth \
        --split test \
        --batch-size 1

    # 或者只给 --config，让脚本自己从工作目录推断 checkpoint：
    python lgff/bench_sc.py --config lgff/configs/linemod_ape_sc_mnl_all.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

# 确保从项目根目录执行脚本时可以 import 到 lgff
sys.path.append(os.getcwd())

from lgff.utils.config import LGFFConfig, load_config, merge_cfg_from_checkpoint
from lgff.utils.geometry import GeometryToolkit
from lgff.utils.logger import setup_logger, get_logger
from lgff.datasets.single_loader import SingleObjectDataset
from lgff.models.lgff_sc import LGFF_SC
from lgff.eval_sc import load_model_weights, resolve_checkpoint_path
from common.ffb6d_utils.model_complexity import ModelComplexityLogger


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LGFF Single-Class model")

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (same as train/eval).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to model checkpoint (.pth). If not set, will try "
            "<work_dir>/checkpoint_best.pth."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to benchmark on (default: cfg.val_split or 'test').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for benchmarking (default: cfg.batch_size).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override dataloader workers (default: cfg.num_workers).",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=20,
        help="Number of warmup iterations for single-batch timing.",
    )
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=100,
        help="Number of iterations for single-batch timing.",
    )

    args, _ = parser.parse_known_args()
    return args


# ----------------------------------------------------------------------
# Benchmark helpers
# ----------------------------------------------------------------------
def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def benchmark_single_batch(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
    warmup: int = 20,
    iters: int = 100,
) -> Dict[str, float]:
    """
    测量单个 batch 的 forward 时间（不包含 DataLoader），更贴近 YOLO 报法。

    Returns:
        {
            "batch_size": ...,
            "ms_per_batch": ...,
            "ms_per_sample": ...
        }
    """
    model.eval()
    batch = _move_batch_to_device(batch, device)

    # 估计 batch size（以 rgb 的第 0 维为准）
    if "rgb" in batch and isinstance(batch["rgb"], torch.Tensor):
        bs = int(batch["rgb"].shape[0])
    else:
        # 兜底：找一个有 batch 维度的 Tensor
        bs = None
        for v in batch.values():
            if isinstance(v, torch.Tensor) and v.ndim >= 1:
                bs = int(v.shape[0])
                break
        if bs is None:
            bs = 1

    with torch.no_grad():
        # 预热
        for _ in range(warmup):
            _ = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # 正式计时
        start = time.perf_counter()
        for _ in range(iters):
            _ = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

    total_s = end - start
    ms_per_batch = (total_s / iters) * 1000.0
    ms_per_sample = ms_per_batch / max(1, bs)

    return {
        "batch_size": float(bs),
        "ms_per_batch": float(ms_per_batch),
        "ms_per_sample": float(ms_per_sample),
    }


def benchmark_full_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    在整个 split 上跑一遍 eval forward，分别统计：
      - wall-clock：DataLoader + model 全流程平均时间
      - model-only：仅 forward 部分的平均时间

    Returns:
        {
            "num_samples": ...,
            "num_batches": ...,
            "wall_ms_per_sample": ...,
            "wall_ms_per_batch": ...,
            "model_ms_per_sample": ...,
            "model_ms_per_batch": ...,
        }
    """
    model.eval()

    num_samples = 0
    num_batches = 0
    total_wall_s = 0.0
    total_model_s = 0.0

    with torch.no_grad():
        # wall-clock 计时：从开始迭代 loader 到结束
        wall_start = time.perf_counter()

        for batch in loader:
            num_batches += 1

            bs = batch["rgb"].shape[0] if isinstance(batch.get("rgb"), torch.Tensor) else 1
            num_samples += bs

            batch = _move_batch_to_device(batch, device)

            # 模型前向时间（仅 forward）
            if device.type == "cuda":
                torch.cuda.synchronize()
            model_start = time.perf_counter()

            _ = model(batch)

            if device.type == "cuda":
                torch.cuda.synchronize()
            model_end = time.perf_counter()

            total_model_s += (model_end - model_start)

        if device.type == "cuda":
            torch.cuda.synchronize()
        wall_end = time.perf_counter()

        total_wall_s = wall_end - wall_start

    if num_batches == 0 or num_samples == 0:
        return {
            "num_samples": 0.0,
            "num_batches": 0.0,
            "wall_ms_per_sample": float("nan"),
            "wall_ms_per_batch": float("nan"),
            "model_ms_per_sample": float("nan"),
            "model_ms_per_batch": float("nan"),
        }

    wall_ms_per_batch = (total_wall_s / num_batches) * 1000.0
    wall_ms_per_sample = (total_wall_s / num_samples) * 1000.0
    model_ms_per_batch = (total_model_s / num_batches) * 1000.0
    model_ms_per_sample = (total_model_s / num_samples) * 1000.0

    return {
        "num_samples": float(num_samples),
        "num_batches": float(num_batches),
        "wall_ms_per_sample": float(wall_ms_per_sample),
        "wall_ms_per_batch": float(wall_ms_per_batch),
        "model_ms_per_sample": float(model_ms_per_sample),
        "model_ms_per_batch": float(model_ms_per_batch),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # 1) 先用全局的 config loader 解析 YAML / CLI（与 train/eval 一致）
    cfg_cli: LGFFConfig = load_config()

    # 2) 从 checkpoint 恢复训练时的 config，保证 backbone/维度完全一致
    if args.checkpoint is not None:
        ckpt_path = resolve_checkpoint_path(args.checkpoint)
    else:
        # 若未显式指定 checkpoint，则默认工作目录下的 checkpoint_best.pth
        work_dir = getattr(cfg_cli, "work_dir", None) or getattr(cfg_cli, "log_dir", "output")
        ckpt_path = os.path.join(work_dir, "checkpoint_best.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    cfg = merge_cfg_from_checkpoint(cfg_cli, checkpoint.get("config"))

    # 3) 工作目录 + 日志
    work_dir = getattr(cfg, "work_dir", None) or getattr(cfg, "log_dir", "output")
    os.makedirs(work_dir, exist_ok=True)
    log_file = os.path.join(work_dir, "bench.log")
    setup_logger(log_file, name="lgff.bench")
    logger = get_logger("lgff.bench")

    logger.info(f"Using checkpoint: {ckpt_path}")

    # 4) Dataset & Dataloader
    split = args.split or getattr(cfg, "val_split", "test")
    batch_size = args.batch_size or getattr(cfg, "batch_size", 8)
    num_workers = args.num_workers or getattr(cfg, "num_workers", 4)

    geometry = GeometryToolkit()

    dataset = SingleObjectDataset(cfg, split=split)
    if len(dataset) == 0:
        logger.error(f"Dataset split={split} is empty. Nothing to benchmark.")
        return

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    logger.info(
        f"Benchmarking split={split} | dataset_size={len(dataset)} | "
        f"batch_size={batch_size} | num_workers={num_workers}"
    )

    # 5) Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LGFF_SC(cfg, geometry)
    load_model_weights(model, ckpt_path, device, checkpoint=checkpoint)
    model = model.to(device)

    # 6) 可选：打印一次复杂度（含 per-sample GFLOPs）
    complexity_logger = ModelComplexityLogger()
    try:
        example_batch = next(iter(loader))
        comp_info = complexity_logger.maybe_log(model, example_batch, stage="bench")
        if comp_info:
            logger.info(
                " | ".join(
                    [
                        "[ModelComplexity] Stage=bench",
                        f"Params: {comp_info['params']:,} "
                        f"({comp_info['param_mb']:.2f} MB)",
                        (
                            f"GFLOPs (batch) = {comp_info['gflops']:.3f}, "
                            f"GFLOPs (per sample) = "
                            f"{comp_info.get('gflops_per_sample', float('nan')):.3f}"
                            if comp_info.get("gflops") is not None
                            else "GFLOPs: N/A",
                        ),
                    ]
                )
            )
    except StopIteration:
        logger.warning("Loader is empty during complexity logging.")
        example_batch = None
    except Exception as exc:
        logger.warning(f"Model complexity logging failed: {exc}")
        example_batch = None

    # 7) 单 batch 推理时间（仅 forward）
    if example_batch is None:
        example_batch = next(iter(loader))

    single_stats = benchmark_single_batch(
        model=model,
        batch=example_batch,
        device=device,
        warmup=args.warmup_iters,
        iters=args.bench_iters,
    )
    logger.info(
        "[SingleBatch] "
        f"B={int(single_stats['batch_size'])} | "
        f"forward = {single_stats['ms_per_batch']:.3f} ms / batch "
        f"({single_stats['ms_per_sample']:.3f} ms / sample)"
    )

    # 8) 整个 split 上的吞吐（包含 DataLoader & forward）
    full_stats = benchmark_full_split(model=model, loader=loader, device=device)
    logger.info(
        "[FullSplit] "
        f"num_samples={int(full_stats['num_samples'])}, "
        f"num_batches={int(full_stats['num_batches'])}"
    )
    logger.info(
        "[FullSplit] Wall-clock: "
        f"{full_stats['wall_ms_per_batch']:.3f} ms / batch, "
        f"{full_stats['wall_ms_per_sample']:.3f} ms / sample"
    )
    logger.info(
        "[FullSplit] Model-only: "
        f"{full_stats['model_ms_per_batch']:.3f} ms / batch, "
        f"{full_stats['model_ms_per_sample']:.3f} ms / sample"
    )

    # 9) 控制台也打印一份简洁 summary，方便你直接抄进笔记
    print("=" * 80)
    print(f"[Benchmark Summary] split={split}, device={device.type}")
    print(
        f"Single batch: B={int(single_stats['batch_size'])}, "
        f"{single_stats['ms_per_batch']:.3f} ms / batch, "
        f"{single_stats['ms_per_sample']:.3f} ms / sample"
    )
    print(
        f"Full split (wall-clock): "
        f"{full_stats['wall_ms_per_batch']:.3f} ms / batch, "
        f"{full_stats['wall_ms_per_sample']:.3f} ms / sample"
    )
    print(
        f"Full split (model-only): "
        f"{full_stats['model_ms_per_batch']:.3f} ms / batch, "
        f"{full_stats['model_ms_per_sample']:.3f} ms / sample"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
