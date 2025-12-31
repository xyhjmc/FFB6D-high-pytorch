"""
Hyper-Fast Batch-Parallel Rigid PnP Solver for LGFF.

Optimizations:
1. Batch Parallelism: Removed the outer loop over batch size.
   Processes (B * RANSAC_ITER) hypotheses simultaneously.
2. Tensorized Aggregation: Computes weighted means for the entire batch in one go.
3. Zero-Overhead RANSAC: Uses advanced indexing to gather points without Python loops.

Performance Target: ~30it/s (Same as model inference speed).
"""

from __future__ import annotations

import logging
import torch

class PnPSolver:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.logger = logging.getLogger("lgff.pnp")

        # 1. 投票聚合配置
        self.vote_top_k = int(getattr(cfg, "pnp_vote_top_k", 256))

        # 2. RANSAC 配置
        # 每个样本并行采样的假设数量
        self.ransac_iter = int(getattr(cfg, "pnp_ransac_iter", 100))
        # 内点阈值 (米)
        self.ransac_inlier_th = float(getattr(cfg, "pnp_ransac_inlier_th", 0.01))

    def solve_batch(
        self,
        points: torch.Tensor,       # [B, N, 3]
        pred_kp_ofs: torch.Tensor,  # [B, K, N, 3]
        pred_conf: torch.Tensor,    # [B, N, 1]
        model_kps: torch.Tensor,    # [B, K, 3] or [K, 3]
    ) -> torch.Tensor:
        """
        全 Batch 并行解算，无 Python 循环。
        Returns:
            pred_rt: [B, 3, 4]
        """
        B, N, _ = points.shape
        K = pred_kp_ofs.shape[1]
        device = points.device

        # --- 1. 全局 Batch 恢复投票坐标 ---
        # points: [B, 1, N, 3] + pred_kp_ofs: [B, K, N, 3] -> votes: [B, K, N, 3]
        votes = points.unsqueeze(1) + pred_kp_ofs

        # 处理 model_kps 的维度，确保它是 [B, K, 3]
        if model_kps.dim() == 2:
            model_kps = model_kps.unsqueeze(0).expand(B, -1, -1)

        # --- Step A: Batch 极速聚合 (Weighted Mean) ---
        # 输入: [B, K, N, 3] 和 [B, N, 1]
        # 输出: [B, K, 3]
        pred_kps_cam = self._batch_fast_aggregate(votes, pred_conf)

        # --- Step B: Batch 向量化 RANSAC ---
        # 输入: [B, K, 3] 和 [B, K, 3]
        # 输出: [B, 3, 4]
        pred_rt = self._batch_ransac(model_kps, pred_kps_cam)

        return pred_rt

    def _batch_fast_aggregate(self, votes: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
        """
        全 Batch 并行聚合。
        votes: [B, K, N, 3]
        conf:  [B, N, 1]
        """
        B, K, N, _ = votes.shape

        if conf is None:
            return votes.mean(dim=2) # [B, K, 3]

        # 1. 选取 Top-K 索引 (在 N 维度)
        # c: [B, N]
        c = conf.view(B, N)
        k_count = min(self.vote_top_k, N)

        # top_vals: [B, TopK], top_idx: [B, TopK]
        top_vals, top_idx = torch.topk(c, k=k_count, dim=1)

        # 2. 归一化权重
        # weights: [B, 1, TopK, 1] (准备广播到 K)
        weights = top_vals / (top_vals.sum(dim=1, keepdim=True) + 1e-6)
        weights = weights.view(B, 1, k_count, 1)

        # 3. Gather 投票点
        # 我们需要在 dim=2 (N) 上 gather。
        # top_idx expanded: [B, K, TopK, 3]
        idx_exp = top_idx.view(B, 1, k_count, 1).expand(B, K, k_count, 3)

        # votes_top: [B, K, TopK, 3]
        votes_top = torch.gather(votes, 2, idx_exp)

        # 4. 加权平均
        # sum([B, K, TopK, 3] * [B, 1, TopK, 1]) -> [B, K, 3]
        centers = (votes_top * weights).sum(dim=2)

        return centers

    def _batch_ransac(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """
        全 Batch RANSAC。同时处理 B 个样本，每个样本 Iter 个假设。
        Total Parallelism = B * Iter.

        src: [B, K, 3] Model
        dst: [B, K, 3] Camera
        """
        B, K, _ = src.shape
        Iter = self.ransac_iter
        device = src.device

        # -----------------------------------------------------------
        # 1. 并行生成假设 (Hypothesis Generation)
        # -----------------------------------------------------------
        # 我们需要从 [0...K-1] 中选 4 个点。
        # indices: [B, Iter, 4]
        rand_idx = torch.randint(0, K, (B, Iter, 4), device=device)

        # Gather src points:
        # src expands to [B, 1, K, 3] -> broadcast gather
        # src_batch: [B, Iter, 4, 3]
        # 为了使用 gather，我们需要把 indices 扩展到适配维度
        # idx_exp: [B, Iter, 4, 3]
        idx_exp = rand_idx.unsqueeze(-1).expand(-1, -1, -1, 3)

        src_expand = src.unsqueeze(1).expand(-1, Iter, -1, -1) # [B, Iter, K, 3]
        dst_expand = dst.unsqueeze(1).expand(-1, Iter, -1, -1) # [B, Iter, K, 3]

        # 采样 4 个点用于解算
        src_samples = torch.gather(src_expand, 2, idx_exp) # [B, Iter, 4, 3]
        dst_samples = torch.gather(dst_expand, 2, idx_exp) # [B, Iter, 4, 3]

        # -----------------------------------------------------------
        # 2. 并行解算姿态 (Batch Kabsch)
        # -----------------------------------------------------------
        # 为了复用 _kabsch_batch，我们将 B 和 Iter 维度合并
        # input: [B*Iter, 4, 3]
        flat_src = src_samples.reshape(B * Iter, 4, 3)
        flat_dst = dst_samples.reshape(B * Iter, 4, 3)

        # R: [B*Iter, 3, 3], t: [B*Iter, 3, 1]
        flat_R, flat_t = self._kabsch_batch(flat_src, flat_dst)

        # 恢复维度
        # R_hyp: [B, Iter, 3, 3]
        # t_hyp: [B, Iter, 3, 1]
        R_hyp = flat_R.view(B, Iter, 3, 3)
        t_hyp = flat_t.view(B, Iter, 3, 1)

        # -----------------------------------------------------------
        # 3. 并行验证 (Verification)
        # -----------------------------------------------------------
        # 将原始 src [B, K, 3] 全部变换
        # src: [B, K, 3] -> [B, 1, 3, K] (transpose for matmul)
        src_T = src.transpose(1, 2).unsqueeze(1) # [B, 1, 3, K]

        # R_hyp: [B, Iter, 3, 3]
        # t_hyp: [B, Iter, 3, 1]
        # pred:  [B, Iter, 3, K]
        pred_trans = torch.matmul(R_hyp, src_T) + t_hyp

        # 转换回 [B, Iter, K, 3] 以计算距离
        pred_trans = pred_trans.transpose(2, 3)

        # dst: [B, K, 3] -> [B, 1, K, 3]
        diff = pred_trans - dst.unsqueeze(1)
        dist = torch.norm(diff, dim=3) # [B, Iter, K]

        # 统计内点: [B, Iter]
        num_inliers = (dist < self.ransac_inlier_th).sum(dim=2)

        # -----------------------------------------------------------
        # 4. 选择最佳模型 (Selection)
        # -----------------------------------------------------------
        # best_iter_idx: [B]
        best_iter_idx = torch.argmax(num_inliers, dim=1)

        # Gather best R and t
        # R_hyp: [B, Iter, 3, 3] -> gather using best_iter_idx
        # idx_r: [B, 1, 1, 1] -> [B, 1, 3, 3]
        idx_view = best_iter_idx.view(B, 1, 1, 1)

        best_R = torch.gather(R_hyp, 1, idx_view.expand(-1, -1, 3, 3)).squeeze(1) # [B, 3, 3]
        best_t = torch.gather(t_hyp, 1, idx_view.expand(-1, -1, 3, 1)).squeeze(1) # [B, 3, 1]

        # [Note]: 为了极速，我们这里略过了 "Refinement using all inliers" 步骤。
        # 因为在 Batch 模式下，每个样本的内点数量不同，无法高效向量化 Refinement。
        # 考虑到我们跑了 100+ 次假设，Best Hypothesis 通常已经非常接近最优解。

        # 返回 [B, 3, 4]
        return torch.cat([best_R, best_t], dim=2)

    @staticmethod
    def _kabsch_batch(P: torch.Tensor, Q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        标准 Batch Kabsch 算法
        P, Q: [N_batch, N_points, 3]
        Returns: R [N_batch, 3, 3], t [N_batch, 3, 1]
        """
        # 均值
        mu_p = P.mean(dim=1, keepdim=True)
        mu_q = Q.mean(dim=1, keepdim=True)

        # 去中心化
        p_c = P - mu_p
        q_c = Q - mu_q

        # 协方差矩阵 H = P^T * Q
        H = torch.matmul(p_c.transpose(1, 2), q_c)

        # SVD
        u, _, vt = torch.linalg.svd(H)

        # R = V U^T
        R = torch.matmul(vt.transpose(1, 2), u.transpose(1, 2))

        # 修正反射
        det = torch.linalg.det(R)
        mask = det < 0
        if mask.any():
            vt[mask, 2, :] *= -1
            R[mask] = torch.matmul(vt[mask].transpose(1, 2), u[mask].transpose(1, 2))

        t = mu_q.transpose(1, 2) - torch.matmul(R, mu_p.transpose(1, 2))
        return R, t