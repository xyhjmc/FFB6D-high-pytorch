"""
Professional Benchmark Script for LGFF-Lite.
Measures Latency and FPS for Batch=1 (Real-time Simulation).

Breakdown:
1. Network Inference (Backbone + Fusion + Heads)
2. Solver (Hybrid PnP / Regression + Logic Branching)
"""
import argparse
import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
# # 确保从项目根目录执行脚本时可以 import 到 lgff
# sys.path.append(os.getcwd())
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 项目模块导入
from lgff.utils.config import load_config, merge_cfg_from_checkpoint
from lgff.utils.geometry import GeometryToolkit
from lgff.models.lgff_sc import LGFF_SC
from lgff.utils.pnp_solver import PnPSolver
from lgff.eval_sc import load_model_weights, resolve_checkpoint_path


def parse_args():
    parser = argparse.ArgumentParser(description="LGFF-Lite Latency Benchmark")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--iter", type=int, default=500, help="Number of iterations for averaging")
    parser.add_argument("--n-points", type=int, default=1024, help="Number of points (e.g. 512, 1024)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"\n[Benchmark] Setting up device: {device}")

    # 1. Load Config & Model
    print("[Benchmark] Loading model...")
    cfg_cli = load_config()
    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = merge_cfg_from_checkpoint(cfg_cli, ckpt.get("config"))

    # 强制覆盖点数配置，方便测试不同点数的影响
    cfg.n_sample_points = args.n_points
    print(f"[Benchmark] Configured N_POINTS = {cfg.n_sample_points}")

    geometry = GeometryToolkit()
    model = LGFF_SC(cfg, geometry).to(device)
    load_model_weights(model, ckpt_path, device, checkpoint=ckpt)
    model.eval()

    # 2. Setup Solver
    # 确保使用最新的并行化 PnP
    pnp_solver = PnPSolver(cfg)

    # 3. Construct Dummy Input (Batch=1)
    print("[Benchmark] Generating dummy inputs...")
    B = 1
    N = args.n_points
    K = getattr(cfg, "n_kps", 8)

    # 创建一个单位矩阵作为伪造的内参
    dummy_K = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)

    dummy_batch = {
        "rgb": torch.randn(B, 3, 128, 128, device=device, dtype=torch.float32),
        "point_cloud": torch.randn(B, N, 3, device=device, dtype=torch.float32),  # 之前修过的
        "choose": torch.zeros(B, N, device=device, dtype=torch.int64),
        "cls_id": torch.tensor([1], device=device, dtype=torch.int64),
        "kp3d_model": torch.randn(B, K, 3, device=device, dtype=torch.float32),

        # [新增] 补上缺失的相机内参 [B, 3, 3]
        "intrinsic": dummy_K,
    }

    # 4. Warm-up (极重要：让 GPU 频率跑起来)
    print("[Benchmark] Warming up GPU (50 iters)...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_batch)
            torch.cuda.synchronize()

    # 5. Benchmarking Loop
    print(f"[Benchmark] Running {args.iter} iterations...")

    # 时间记录容器 (ms)
    t_net_list = []
    t_solver_list = []
    t_total_list = []

    # 定义 CUDA 事件
    starter = torch.cuda.Event(enable_timing=True)
    ender_net = torch.cuda.Event(enable_timing=True)
    ender_total = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in tqdm(range(args.iter)):
            # --- Start Timing ---
            torch.cuda.synchronize()
            starter.record()

            # [Stage 1] Network Inference
            outputs = model(dummy_batch)

            # Record Network Time
            ender_net.record()

            # [Stage 2] Solver Logic (Hybrid Strategy Simulation)
            # 模拟 evaluator 中的逻辑

            # A. Regression Base
            # 简单模拟处理过程 (Reshape/Permute)
            pred_trans = outputs["pred_trans"]
            pred_kp_ofs = outputs["pred_kp_ofs"]
            pred_conf = outputs["pred_conf"]
            cloud_input = outputs.get("points", dummy_batch["point_cloud"])
            # B. PnP Execution
            # 即使是 Batch=1，我们也要跑一遍 PnP 来测算它的耗时
            # 这里调用 solve_batch
            _ = pnp_solver.solve_batch(
                points=cloud_input,
                pred_kp_ofs=pred_kp_ofs,
                pred_conf=pred_conf,
                model_kps=dummy_batch["kp3d_model"]
            )

            # C. Hybrid Combination Logic
            # 简单的张量切片和拼接操作 (耗时极短，但为了严谨算在内)
            _ = pred_trans.mean(dim=1)

            # --- End Timing ---
            ender_total.record()
            torch.cuda.synchronize()

            # 计算耗时 (ms)
            curr_net_time = starter.elapsed_time(ender_net)
            curr_total_time = starter.elapsed_time(ender_total)
            curr_solver_time = curr_total_time - curr_net_time

            t_net_list.append(curr_net_time)
            t_solver_list.append(curr_solver_time)
            t_total_list.append(curr_total_time)

    # 6. Statistical Analysis
    def print_stats(name, data):
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        p50 = np.median(data)
        p95 = np.percentile(data, 95)
        print(f"  -> {name:<10}: Mean={mean:.2f}ms | P50={p50:.2f}ms | P95={p95:.2f}ms | Std={std:.2f}")
        return mean

    print("\n" + "=" * 40)
    print(f"BENCHMARK REPORT (Batch=1, N={args.n_points})")
    print("=" * 40)

    mean_net = print_stats("Network", t_net_list)
    mean_solver = print_stats("Solver+PnP", t_solver_list)
    mean_total = print_stats("Total", t_total_list)

    fps = 1000.0 / mean_total
    print("-" * 40)
    print(f"  Final FPS   : {fps:.2f} Frames/sec")
    print("-" * 40)

    # 占比分析
    net_pct = (mean_net / mean_total) * 100
    sol_pct = (mean_solver / mean_total) * 100
    print(f"  Breakdown   : Network {net_pct:.1f}% | Solver {sol_pct:.1f}%")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()