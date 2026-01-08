"""
Professional Benchmark Script for LGFF-Seg (Segmentation Enhanced).
Measures Latency and FPS for Batch=1 (Real-time Simulation).

Breakdown:
1. Network Inference (Backbone + Fusion + Seg Head + Pose Heads)
2. Solver (Mask Generation + Masked Confidence + SVD/Fusion)
"""
import argparse
import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

# 路径设置
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# [CHANGED] 引入 Seg 相关的模块
from lgff.utils.config_seg import load_config, merge_cfg_from_checkpoint, LGFFConfigSeg
from lgff.utils.geometry import GeometryToolkit
from lgff.models.lgff_sc_seg import LGFF_SC_SEG
# [CHANGED] 引入实际推理用的融合函数
from lgff.utils.pose_metrics_seg import fuse_pose_from_outputs
from lgff.eval_sc import load_model_weights, resolve_checkpoint_path


def parse_args():
    parser = argparse.ArgumentParser(description="LGFF-Seg Latency Benchmark")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--iter", type=int, default=500, help="Number of iterations for averaging")
    parser.add_argument("--n-points", type=int, default=1024, help="Number of points (e.g. 512, 1024)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--use-pnp", action="store_true", default=True, help="Force enable PnP mode for benchmark")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"\n[Benchmark] Setting up device: {device}")

    # 1. Load Config & Model
    print("[Benchmark] Loading Seg model...")
    # [CHANGED] 使用 ConfigSeg
    cfg_cli = load_config()
    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 手动合并配置 (简化版 logic)
    ckpt_cfg = ckpt.get("config", {})
    cfg = LGFFConfigSeg()
    cfg.update(ckpt_cfg)

    # 强制覆盖配置以模拟推理环境
    cfg.num_points = args.n_points
    cfg.eval_use_pnp = args.use_pnp  # 强制开启 PnP 以测试最重负载
    cfg.pose_fusion_use_valid_mask = True # 强制开启 Mask 过滤
    cfg.pose_fusion_valid_mask_source = "seg"

    print(f"[Benchmark] Configured N_POINTS = {cfg.num_points}")
    print(f"[Benchmark] Mode: {'PnP (SVD)' if cfg.eval_use_pnp else 'Dense Regression'}")

    geometry = GeometryToolkit()
    # [CHANGED] 实例化 Seg 模型
    model = LGFF_SC_SEG(cfg, geometry).to(device)
    load_model_weights(model, ckpt_path, device, checkpoint=ckpt)
    model.eval()

    # 2. Construct Dummy Input (Batch=1)
    print("[Benchmark] Generating dummy inputs...")
    B = 1
    N = args.n_points
    K_kps = getattr(cfg, "num_keypoints", 8)
    H, W = 128, 128

    dummy_K = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)

    # 模拟 `choose` 索引 (随机采样)
    dummy_choose = torch.randint(0, H*W, (B, N), device=device, dtype=torch.int64)

    dummy_batch = {
        "rgb": torch.randn(B, 3, H, W, device=device, dtype=torch.float32),
        "point_cloud": torch.randn(B, N, 3, device=device, dtype=torch.float32),
        "choose": dummy_choose,
        "cls_id": torch.tensor([1], device=device, dtype=torch.int64),
        "intrinsic": dummy_K,

        # [CRITICAL] 注入 model_kps 供 SVD 解算使用
        "kp3d_model": torch.randn(B, K_kps, 3, device=device, dtype=torch.float32),
        # 补充 Seg 模型可能需要的字段
        "points": torch.randn(B, N, 3, device=device, dtype=torch.float32),
    }
    # points 通常是 point_cloud 的别名，确保都有
    dummy_batch["points"] = dummy_batch["point_cloud"]

    # 3. Warm-up
    print("[Benchmark] Warming up GPU (50 iters)...")
    with torch.no_grad():
        for _ in range(50):
            # 注入 model_kps 到 outputs 模拟 evaluator 行为
            outs = model(dummy_batch)
            outs["model_kps"] = dummy_batch["kp3d_model"]
            torch.cuda.synchronize()

    # 4. Benchmarking Loop
    print(f"[Benchmark] Running {args.iter} iterations...")

    t_net_list = []
    t_solver_list = []
    t_total_list = []

    starter = torch.cuda.Event(enable_timing=True)
    ender_net = torch.cuda.Event(enable_timing=True)
    ender_total = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in tqdm(range(args.iter)):
            torch.cuda.synchronize()
            starter.record()

            # --- [Stage 1] Network Inference ---
            outputs = model(dummy_batch)

            # 手动注入 model_kps (因为这是 Dataset 的责任，这里模拟它已经存在)
            outputs["model_kps"] = dummy_batch["kp3d_model"]

            ender_net.record()

            # --- [Stage 2] Solver Logic (Seg Specific) ---
            # 我们需要手动模拟 Evaluator 中的 Mask 处理逻辑，因为这部分耗时不能忽略

            # A. Compute Mask (Sigmoid + Gather + TopK)
            pred_seg = outputs.get("pred_mask_logits", outputs.get("pred_seg", None))
            if pred_seg is not None:
                # 1. Sigmoid
                seg_probs = torch.sigmoid(pred_seg) # [B, 1, H, W]
                # 2. Gather (2D -> 3D mapping)
                seg_flat = seg_probs.view(B, -1)
                point_probs = torch.gather(seg_flat, 1, dummy_batch["choose"]) # [B, N]
                # 3. Threshold
                valid_mask = point_probs > 0.5

                # (忽略 Top-K safety net 的微小分支预测耗时，主要算矩阵操作)
            else:
                valid_mask = None

            # B. Pose Fusion / SVD
            # 调用实际的融合函数 (它内部会做 SVD 或 Dense 回归)
            _ = fuse_pose_from_outputs(
                outputs,
                geometry,
                cfg,
                stage="eval",
                valid_mask=valid_mask
            )

            ender_total.record()
            torch.cuda.synchronize()

            curr_net_time = starter.elapsed_time(ender_net)
            curr_total_time = starter.elapsed_time(ender_total)
            curr_solver_time = curr_total_time - curr_net_time

            t_net_list.append(curr_net_time)
            t_solver_list.append(curr_solver_time)
            t_total_list.append(curr_total_time)

    # 5. Stats
    def print_stats(name, data):
        data = np.array(data)
        mean = np.mean(data)
        p50 = np.median(data)
        p95 = np.percentile(data, 95)
        print(f"  -> {name:<12}: Mean={mean:.2f}ms | P50={p50:.2f}ms | P95={p95:.2f}ms")
        return mean

    print("\n" + "=" * 50)
    print(f"BENCHMARK REPORT (Seg Model, Batch=1, N={args.n_points})")
    print("=" * 50)

    mean_net = print_stats("Network", t_net_list)
    mean_solver = print_stats("Seg+Solver", t_solver_list)
    mean_total = print_stats("Total", t_total_list)

    fps = 1000.0 / mean_total
    print("-" * 50)
    print(f"  Final FPS    : {fps:.2f} Frames/sec")
    print("-" * 50)

    net_pct = (mean_net / mean_total) * 100
    sol_pct = (mean_solver / mean_total) * 100
    print(f"  Breakdown    : Network {net_pct:.1f}% | Solver {sol_pct:.1f}%")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()