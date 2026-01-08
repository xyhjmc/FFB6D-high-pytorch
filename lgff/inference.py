"""
Interactive Inference Demo for LGFF-Seg (High-Performance PnP Version).

Features:
1. Loads images from a folder sequentially.
2. User draws a box -> Press SPACE to infer.
3. Uses the CUSTOM Batch-Parallel PnP Solver (RANSAC) provided by user.
4. Projects 3D Box back onto the original image.
"""
import argparse
import glob
import logging
import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from plyfile import PlyData

sys.path.append(os.getcwd())

from lgff.utils.config_seg import load_config, merge_cfg_from_checkpoint
from lgff.utils.geometry import GeometryToolkit
from lgff.models.lgff_sc_seg import LGFF_SC_SEG


# ==============================================================================
# 1. USER PROVIDED PNP SOLVER (Embedded here for standalone usage)
# ==============================================================================
class PnPSolver:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        # self.logger = logging.getLogger("lgff.pnp") # Suppress logger for demo

        # 1. Voting Config
        self.vote_top_k = int(getattr(cfg, "pnp_vote_top_k", 256))

        # 2. RANSAC Config
        self.ransac_iter = int(getattr(cfg, "pnp_ransac_iter", 100))
        self.ransac_inlier_th = float(getattr(cfg, "pnp_ransac_inlier_th", 0.01))

    def solve_batch(
            self,
            points: torch.Tensor,  # [B, N, 3]
            pred_kp_ofs: torch.Tensor,  # [B, K, N, 3]
            pred_conf: torch.Tensor,  # [B, N, 1]
            model_kps: torch.Tensor,  # [B, K, 3] or [K, 3]
    ) -> torch.Tensor:
        B, N, _ = points.shape
        K = pred_kp_ofs.shape[1]

        # --- 1. Global Vote Recovery ---
        # points: [B, 1, N, 3] + pred_kp_ofs: [B, K, N, 3] -> votes: [B, K, N, 3]
        votes = points.unsqueeze(1) + pred_kp_ofs

        if model_kps.dim() == 2:
            model_kps = model_kps.unsqueeze(0).expand(B, -1, -1)

        # --- Step A: Fast Aggregation ---
        pred_kps_cam = self._batch_fast_aggregate(votes, pred_conf)

        # --- Step B: Batch RANSAC ---
        pred_rt = self._batch_ransac(model_kps, pred_kps_cam)

        return pred_rt

    def _batch_fast_aggregate(self, votes: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
        B, K, N, _ = votes.shape
        if conf is None:
            return votes.mean(dim=2)

        c = conf.view(B, N)
        k_count = min(self.vote_top_k, N)
        top_vals, top_idx = torch.topk(c, k=k_count, dim=1)

        weights = top_vals / (top_vals.sum(dim=1, keepdim=True) + 1e-6)
        weights = weights.view(B, 1, k_count, 1)

        idx_exp = top_idx.view(B, 1, k_count, 1).expand(B, K, k_count, 3)
        votes_top = torch.gather(votes, 2, idx_exp)
        centers = (votes_top * weights).sum(dim=2)
        return centers

    def _batch_ransac(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        B, K, _ = src.shape
        Iter = self.ransac_iter
        device = src.device

        rand_idx = torch.randint(0, K, (B, Iter, 4), device=device)
        idx_exp = rand_idx.unsqueeze(-1).expand(-1, -1, -1, 3)

        src_expand = src.unsqueeze(1).expand(-1, Iter, -1, -1)
        dst_expand = dst.unsqueeze(1).expand(-1, Iter, -1, -1)

        src_samples = torch.gather(src_expand, 2, idx_exp)
        dst_samples = torch.gather(dst_expand, 2, idx_exp)

        flat_src = src_samples.reshape(B * Iter, 4, 3)
        flat_dst = dst_samples.reshape(B * Iter, 4, 3)

        flat_R, flat_t = self._kabsch_batch(flat_src, flat_dst)

        R_hyp = flat_R.view(B, Iter, 3, 3)
        t_hyp = flat_t.view(B, Iter, 3, 1)

        src_T = src.transpose(1, 2).unsqueeze(1)
        pred_trans = torch.matmul(R_hyp, src_T) + t_hyp
        pred_trans = pred_trans.transpose(2, 3)

        diff = pred_trans - dst.unsqueeze(1)
        dist = torch.norm(diff, dim=3)
        num_inliers = (dist < self.ransac_inlier_th).sum(dim=2)

        best_iter_idx = torch.argmax(num_inliers, dim=1)
        idx_view = best_iter_idx.view(B, 1, 1, 1)

        best_R = torch.gather(R_hyp, 1, idx_view.expand(-1, -1, 3, 3)).squeeze(1)
        best_t = torch.gather(t_hyp, 1, idx_view.expand(-1, -1, 3, 1)).squeeze(1)

        return torch.cat([best_R, best_t], dim=2)

    @staticmethod
    def _kabsch_batch(P: torch.Tensor, Q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu_p = P.mean(dim=1, keepdim=True)
        mu_q = Q.mean(dim=1, keepdim=True)
        p_c = P - mu_p
        q_c = Q - mu_q
        H = torch.matmul(p_c.transpose(1, 2), q_c)
        u, _, vt = torch.linalg.svd(H)
        R = torch.matmul(vt.transpose(1, 2), u.transpose(1, 2))
        det = torch.linalg.det(R)
        mask = det < 0
        if mask.any():
            vt[mask, 2, :] *= -1
            R[mask] = torch.matmul(vt[mask].transpose(1, 2), u[mask].transpose(1, 2))
        t = mu_q.transpose(1, 2) - torch.matmul(R, mu_p.transpose(1, 2))
        return R, t


# ==============================================================================
# 2. UI Helper
# ==============================================================================
class BoxSelector:
    def __init__(self, window_name="Select Object"):
        self.window_name = window_name
        self.image = None
        self.start_point = None
        self.end_point = None
        self.drawing = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing: self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)

    def select_box(self, image):
        self.image = image.copy()
        self.start_point = None
        self.end_point = None
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        print("\n[Controls] Draw Box -> SPACE to confirm | 'c' Clear | 'd' Skip | 'q' Quit")

        while True:
            temp_img = self.image.copy()
            if self.start_point and self.end_point:
                cv2.rectangle(temp_img, self.start_point, self.end_point, (0, 255, 255), 2)
            cv2.imshow(self.window_name, temp_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if self.start_point and self.end_point:
                    x1, y1 = self.start_point
                    x2, y2 = self.end_point
                    xmin, xmax = sorted([x1, x2])
                    ymin, ymax = sorted([y1, y2])
                    if (xmax - xmin) > 5: return (xmin, ymin, xmax, ymax)
            elif key == ord('c'):
                self.start_point = None
            elif key == ord('d'):
                return None
            elif key == ord('q'):
                sys.exit(0)
        cv2.destroyWindow(self.window_name)


# ==============================================================================
# 3. Preprocessing
# ==============================================================================
class InteractivePreprocessor:
    def __init__(self, cfg, device):
        self.device = device
        self.resize_hw = 128
        self.num_points = int(cfg.num_points)
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]

    def process(self, rgb, depth, bbox, K_original, depth_scale):
        xmin, ymin, xmax, ymax = bbox
        # Square Crop + Padding
        h, w = ymax - ymin, xmax - xmin
        size = int(max(h, w) * 1.1)
        cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
        H_img, W_img = depth.shape
        rmin = max(0, cy - size // 2)
        rmax = min(H_img, rmin + size)
        cmin = max(0, cx - size // 2)
        cmax = min(W_img, cmin + size)
        real_h, real_w = rmax - rmin, cmax - cmin

        rgb_crop = rgb[rmin:rmax, cmin:cmax]
        depth_crop = depth[rmin:rmax, cmin:cmax].astype(np.float32) / depth_scale
        rgb_rez = cv2.resize(rgb_crop, (self.resize_hw, self.resize_hw))
        depth_rez = cv2.resize(depth_crop, (self.resize_hw, self.resize_hw), interpolation=cv2.INTER_NEAREST)

        # Adjust K
        K_new = K_original.copy()
        K_new[0, 2] -= cmin
        K_new[1, 2] -= rmin
        ratio_w, ratio_h = self.resize_hw / real_w, self.resize_hw / real_h
        K_new[0, 0] *= ratio_w;
        K_new[1, 1] *= ratio_h
        K_new[0, 2] *= ratio_w;
        K_new[1, 2] *= ratio_h

        # Cloud
        ys, xs = np.meshgrid(np.arange(self.resize_hw), np.arange(self.resize_hw), indexing='ij')
        X = (xs - K_new[0, 2]) * depth_rez / K_new[0, 0]
        Y = (ys - K_new[1, 2]) * depth_rez / K_new[1, 1]
        pts_3d = np.stack([X, Y, depth_rez], axis=-1).reshape(-1, 3)

        # Sample
        valid = (depth_rez > 0.01).reshape(-1)
        valid_idx = np.where(valid)[0]
        if len(valid_idx) >= self.num_points:
            choose = np.random.choice(valid_idx, self.num_points, replace=False)
        else:
            choose = np.zeros(self.num_points, dtype=int)
            if len(valid_idx) > 0: choose[:len(valid_idx)] = valid_idx

        pts_sel = pts_3d[choose]
        rgb_norm = cv2.cvtColor(rgb_rez, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        for i in range(3): rgb_norm[..., i] = (rgb_norm[..., i] - self.norm_mean[i]) / self.norm_std[i]

        # [FIX] Ensure "point_cloud" key exists (Model expects it)
        pt_tensor = torch.from_numpy(pts_sel).unsqueeze(0).float().to(self.device)

        batch = {
            "rgb": torch.from_numpy(rgb_norm).permute(2, 0, 1).unsqueeze(0).float().to(self.device),
            "point_cloud": pt_tensor,  # <--- 关键修复：这是模型要求的名字
            "points": pt_tensor,  # <--- 别名，以防万一
            "choose": torch.from_numpy(choose).unsqueeze(0).long().to(self.device),
            "intrinsic": torch.from_numpy(K_new).unsqueeze(0).float().to(self.device),
            "cls_id": torch.tensor([0], device=self.device)
        }
        return batch


# ==============================================================================
# 4. Helpers
# ==============================================================================
def load_ply_model(path):
    ply = PlyData.read(path)
    v = ply["vertex"].data
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32) / 1000.0


def build_box(pts):
    mins, maxs = pts.min(0), pts.max(0)
    center, half = (mins + maxs) / 2, (maxs - mins) / 2
    verts = np.array([center + half * np.array([x, y, z]) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    return verts, edges


def draw_projected_box(img, R, t, K, verts, edges):
    """
    鲁棒的 3D 框投影绘制函数。
    包含 Z轴裁剪 和 坐标范围限制，防止 OpenCV 崩溃。
    """
    H, W = img.shape[:2]

    # 1. 变换到相机坐标系
    # verts: [8, 3], R: [3, 3], t: [3] -> pts_c: [8, 3]
    pts_c = np.dot(verts, R.T) + t

    # 2. [关键修复] 防止除以零或物体在相机背面
    # 强制 Z > 0.01 (1cm)，否则投影会乱飞
    pts_c[:, 2] = np.clip(pts_c[:, 2], 0.01, None)

    # 3. 投影到像素坐标
    # u = (X * fx) / Z + cx
    u = (pts_c[:, 0] * K[0, 0]) / pts_c[:, 2] + K[0, 2]
    # v = (Y * fy) / Z + cy
    v = (pts_c[:, 1] * K[1, 1]) / pts_c[:, 2] + K[1, 2]

    # 4. [关键修复] 防止整数溢出
    # OpenCV draw functions crash with very large coordinates (e.g. > 32768)
    # 我们把坐标限制在图像周围一个安全的缓冲区内 (-2000 ~ W+2000)
    pad = 2000
    u = np.clip(u, -pad, W + pad)
    v = np.clip(v, -pad, H + pad)

    # 5. 四舍五入转整数
    p2d = np.stack([u, v], axis=1)
    p2d = np.round(p2d).astype(int)

    # 6. 画线
    for s, e in edges:
        p1 = tuple(p2d[s])
        p2 = tuple(p2d[e])
        cv2.line(img, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)

    # 7. 画坐标轴 (Center, X, Y, Z)
    # 重新计算坐标轴的中心和方向
    # 假设 verts 和 edges 是配套生成的 box，我们需要单独算 axis
    # 这里为了通用性，我们只画框。如果需要画轴，逻辑同上。

    return img


# ... (前面的 PnPSolver, BoxSelector, InteractivePreprocessor, Helpers 保持不变) ...

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", required=False, default="/home/xyh/datasets/minepose/test/000002/rgb",
                        help="RGB Folder")
    parser.add_argument("--depth-dir", required=False, default="/home/xyh/datasets/minepose/test/000002/depth",
                        help="Depth Folder")
    parser.add_argument("--checkpoint", required=False,
                        default="/home/xyh/PycharmProjects/FFB6D-high-pytorch/output/minepose_shovel_sc_mnl_all_seg/checkpoint_best.pth")
    parser.add_argument("--ply-path", required=False, default="/home/xyh/datasets/minepose/models/obj_000002.ply")
    parser.add_argument("--kps-path", required=False,
                        default="/home/xyh/PycharmProjects/FFB6D-high-pytorch/lgff/data/minepose_shovel_crop128_all/keypoints/obj_000002.npy")
    parser.add_argument("--out-dir", default="demo_results")

    # Intrinsics (LineMod default)
    # parser.add_argument("--fx", type=float, default=572.4114)
    # parser.add_argument("--fy", type=float, default=573.57043)
    # parser.add_argument("--cx", type=float, default=325.2611)
    # parser.add_argument("--cy", type=float, default=242.04899)
    parser.add_argument("--fx", type=float, default=614.362)
    parser.add_argument("--fy", type=float, default=613.514)
    parser.add_argument("--cx", type=float, default=311.294)
    parser.add_argument("--cy", type=float, default=236.754)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda")

    # 1. Load Model & Solver
    print("[Init] Loading...")
    cfg_cli = load_config()

    # Load Checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    # Merge Config
    if "config" in ckpt:
        cfg = merge_cfg_from_checkpoint(cfg_cli, ckpt.get("config"))
    else:
        cfg = cfg_cli  # Fallback

    # Init Solver
    pnp_solver = PnPSolver(cfg)

    # Init Model
    model = LGFF_SC_SEG(cfg, GeometryToolkit()).to(device)

    # -----------------------------------------------------------
    # [FIX] Robust State Dict Loading (增加了对 "state_dict" 的检查)
    # -----------------------------------------------------------
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:  # <--- 关键修复：您的 checkpoint 可能是这种格式
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # 处理 'module.' 前缀 (DDP 训练遗留)
    new_state_dict = {}
    for k, v in state_dict.items():
        # 有些权重文件里可能混杂了无关的键，只保留模型相关的
        if k == "epoch" or k == "optimizer" or k == "config":
            continue

        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # strict=True 确保每一个参数都必须完美匹配，这就不会出现参数没加载进去的情况了
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("[Init] Model weights loaded successfully.")
    except Exception as e:
        print(f"[Warn] Strict loading failed, trying strict=False. Error: {e}")
        model.load_state_dict(new_state_dict, strict=False)

    model.eval()

    kp3d = np.load(args.kps_path).astype(np.float32)[:8]
    kp3d_tensor = torch.from_numpy(kp3d).unsqueeze(0).to(device)  # [1, K, 3]
    verts, edges = build_box(load_ply_model(args.ply_path))
    K_orig = np.array([[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1]], dtype=np.float32)

    processor = InteractivePreprocessor(cfg, device)
    selector = BoxSelector()

    # 2. Iterate Files
    img_files = sorted(glob.glob(os.path.join(args.img_dir, "*.png")) + glob.glob(os.path.join(args.img_dir, "*.jpg")))

    if len(img_files) == 0:
        print(f"[Error] No images found in {args.img_dir}")
        return

    for fpath in img_files:
        fname = os.path.basename(fpath)
        dpath = os.path.join(args.depth_dir, fname)
        if not os.path.exists(dpath):
            # Try png <-> jpg
            if fname.endswith(".jpg"):
                dpath = os.path.join(args.depth_dir, fname.replace(".jpg", ".png"))
            elif fname.endswith(".png"):
                dpath = os.path.join(args.depth_dir, fname.replace(".png", ".jpg"))

        if not os.path.exists(dpath):
            print(f"[Skip] Depth not found: {dpath}")
            continue

        rgb = cv2.imread(fpath)
        depth = cv2.imread(dpath, cv2.IMREAD_UNCHANGED)

        print(f"Processing: {fname}")
        bbox = selector.select_box(rgb)
        if bbox is None: continue

        try:
            # Preprocess
            batch = processor.process(rgb, depth, bbox, K_orig, args.depth_scale)

            with torch.no_grad():
                out = model(batch)

                # --- Seg Mask Logic ---
                pred_mask_logits = out.get("pred_mask_logits", None)
                if isinstance(pred_mask_logits, torch.Tensor):
                    probs = torch.sigmoid(pred_mask_logits).view(1, -1)
                    point_probs = torch.gather(probs, 1, batch["choose"])
                    valid_mask = (point_probs > 0.5).float().unsqueeze(-1)
                else:
                    valid_mask = torch.ones_like(batch["choose"], dtype=torch.float32).unsqueeze(-1)

                # --- Prepare PnP Inputs ---
                pred_conf = out.get("pred_conf", valid_mask)
                pred_conf = pred_conf * valid_mask

                kp_ofs = out["pred_kp_ofs"]
                if kp_ofs.shape[-1] != 3:
                    kp_ofs = kp_ofs.permute(0, 1, 3, 2)

                # --- Run Custom PnPSolver ---
                pose = pnp_solver.solve_batch(
                    points=batch["points"],
                    pred_kp_ofs=kp_ofs,
                    pred_conf=pred_conf,
                    model_kps=kp3d_tensor
                )

                R_pred = pose[0, :3, :3].cpu().numpy()
                t_pred = pose[0, :3, 3].cpu().numpy()

            # Draw & Save
            res = draw_projected_box(rgb.copy(), R_pred, t_pred, K_orig, verts, edges)
            cv2.rectangle(res, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 1)
            cv2.putText(res, f"Z: {t_pred[2]:.2f}m", (bbox[0], bbox[1] - 5), 0, 0.6, (0, 255, 0), 2)

            cv2.imshow("Result", res)
            cv2.waitKey(2000)
            cv2.imwrite(os.path.join(args.out_dir, f"res_{fname}"), res)

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
