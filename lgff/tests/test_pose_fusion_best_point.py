import torch

from lgff.utils.pose_metrics import fuse_pose_from_outputs


class _Cfg:
    # 固定为 best_point 路线，Top-K 配置应被忽略
    train_use_best_point = True
    eval_use_best_point = True
    eval_topk = 8
    pose_fusion_topk = 16
    num_points = 64
    train_topk = 5


class _DummyGeometry:
    @staticmethod
    def quat_to_rot(quat: torch.Tensor) -> torch.Tensor:
        quat = torch.nn.functional.normalize(quat, dim=-1)
        w, x, y, z = quat.unbind(-1)
        B = quat.shape[0]
        rot = torch.zeros((B, 3, 3), device=quat.device, dtype=quat.dtype)
        rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
        rot[:, 0, 1] = 2 * (x * y - z * w)
        rot[:, 0, 2] = 2 * (x * z + y * w)
        rot[:, 1, 0] = 2 * (x * y + z * w)
        rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
        rot[:, 1, 2] = 2 * (y * z - x * w)
        rot[:, 2, 0] = 2 * (x * z - y * w)
        rot[:, 2, 1] = 2 * (y * z + x * w)
        rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return rot


def test_best_point_fusion_ignores_topk_changes():
    torch.manual_seed(0)
    cfg = _Cfg()
    geometry = _DummyGeometry()

    B, N = 2, 32
    outputs = {
        "pred_quat": torch.randn(B, N, 4),
        "pred_trans": torch.randn(B, N, 3),
        "pred_conf": torch.randn(B, N, 1),
    }

    pose_a = fuse_pose_from_outputs(outputs, geometry, cfg, stage="eval")

    # 修改 topk 相关配置，不应影响 best_point 结果
    cfg.eval_topk = 2
    cfg.pose_fusion_topk = 4
    pose_b = fuse_pose_from_outputs(outputs, geometry, cfg, stage="eval")

    assert torch.allclose(pose_a, pose_b, atol=1e-7)


def test_best_point_fusion_train_path():
    torch.manual_seed(0)
    cfg = _Cfg()
    geometry = _DummyGeometry()

    B, N = 1, 16
    outputs = {
        "pred_quat": torch.randn(B, N, 4),
        "pred_trans": torch.randn(B, N, 3),
        "pred_conf": torch.randn(B, N, 1),
    }

    pose_a = fuse_pose_from_outputs(outputs, geometry, cfg, stage="train")
    cfg.train_topk = 1
    pose_b = fuse_pose_from_outputs(outputs, geometry, cfg, stage="train")

    assert torch.allclose(pose_a, pose_b, atol=1e-7)
