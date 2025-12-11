# LGFF 单物体 6D 姿态估计几何/评估一致性审查


## LINEMOD/BOP GT 与 RGB/深度端到端对齐自检
- 可按以下最小脚本验证：
  1. `ds = SingleObjectDataset(cfg, split="test")`; `sample = ds[np.random.randint(len(ds))]`，确保 `obj_id` 为 ape。
  2. 使用 `sample["model_points"]` 与 `sample["pose"]` 通过 `GeometryToolkit.project_points`/`basic_utils.project_p3d` 投影到 `sample["rgb"]` 尺寸的像素平面（`sample["intrinsic"]` 已按 resize 缩放）；观察 CAD 轮廓与 RGB 物体边界对齐情况。
  3. 将 `sample["points"]`（相机系深度点）用同一 `pose` 投影，对比 CAD 投影与真实点云分布差异。
- 数据集读取已将 `cam_t_m2c` mm->m，并根据原始 H/W 对 `cam_K` 进行同步缩放；`depth_scale` 逐帧存储并在 `_load_depth` 时转换到米，减小 off-by-one 风险。仍需注意 `cls_id` 始终固定 0，若下游按 BOP `obj_id` 计算对称性需额外传递原始 ID。【F:lgff/datasets/single_loader.py†L402-L468】

## 评估与可视化数值闭环
- `EvaluatorSC` 将 per-image 指标写入 `per_image_metrics.csv`，字段与内部 `metrics_meter` 一致；`viz_sc` 的标题/日志直接使用 `EvaluatorSC` 的输出并未重复实现 ADD/ADD-S/t/rot 计算，但当前未自动 cross-check CSV 对应行。建议在 `viz_sc` 中若提供 `--per-image-csv` 路径，则根据 `scene_id/im_id` 查表并在可视化标题中打印同一行数据，避免手算差异；同时将 ADD/ADD-S 的计算统一调用 `compute_batch_pose_metrics`，减少与手写 `torch.cdist` 版本的重复维护。【F:lgff/engines/evaluator_sc.py†L327-L365】【F:lgff/viz_sc.py†L303-L364】

## Loss 量纲与权重平衡（基于 metrics_history 的后续检查建议）
- 当前训练日志/`metrics_history.csv` 未在仓库中，无法直接统计各分量量级；推荐在训练脚本中追加一个“loss 量纲快照”函数：在代表性 epoch（如 5、20、末尾）记录 `loss_add`、`loss_t`、`loss_conf`、`loss_add_cad`、`loss_kp_of` 的 batch 均值，再乘以 `lambda_*` 估算总损失贡献比例。可以直接在 `TrainerSC._record_epoch_metrics` 附加一个 CSV/JSON 表，或用 TensorBoard 多 scalar 记录。【F:lgff/engines/trainer_sc.py†L322-L406】【F:lgff/losses/lgff_loss.py†L16-L112】
- 经验建议：以 LINEMOD ape 为例，ROI 几何误差（单位 m）通常在 0.01~0.05，平移 L1（含 z 轴权重 2）在同量级。若观察到 `loss_conf`（权重 0.1）或 `loss_kp_of`（权重 0.6）长期 <1e-3，则可将 `lambda_conf`/`lambda_kp_of` 提升 2~3 倍；若 `loss_add_cad`（权重 0.0 默认关闭）一旦开启在 1e-1 量级，则需把 `lambda_add_cad` 控制在 0.1 以内，避免压制 ROI 分支。【F:lgff/losses/lgff_loss.py†L29-L71】
- 如需自动平衡，可在每个 epoch 记录加权前后的数值并打印“占比柱状”摘要：`weight * loss / loss_total`。当某分支占比>70% 持续多个 epoch 或 <5% 持续多个 epoch 时，提示调权重或重新归一化。还可对平移/几何分支分别做单位注释（全部为米）。

## 阈值单位（米/毫米）一致性复核
- 所有阈值常量都以米实现：`summarize_pose_metrics` 的 `acc_adds<5mm/10mm/...` 使用 0.005、0.010 等小数，`acc_t<10mm/20mm` 使用 0.010、0.020；CMD 阈值 `cmd_threshold_m` 默认 0.02 m，与 Trainer/Evaluator 共用。【F:lgff/utils/pose_metrics.py†L149-L206】
- `EvaluatorSC` 的绝对阈值列表 `acc_abs_adds_thresholds` 也以米存储，并在 per-image CSV 中写入 `succ_adds_10mm` 等字段时乘 1000 仅用于列名，数值比较仍是米；日志打印时的 mm 转换仅用于展示。【F:lgff/engines/evaluator_sc.py†L72-L109】【F:lgff/engines/evaluator_sc.py†L230-L279】【F:lgff/viz_sc.py†L517-L542】
- 暂未发现硬编码的 “5/10” 整数直接参与距离比较；后续若新增阈值，需保持米制，列名或可视化再乘 1000 以显示毫米。

## 数据增强对内参/姿态的影响
- 训练集仅应用颜色增强（`ColorJitter`），不会改变几何；几何相关的 resize 在 `_load_rgb/_load_depth` 内固定到 `resize_w/resize_h`，并在 `__getitem__` 中同步按原始尺寸比例缩放 `K`，保证深度反投影和投影一致。【F:lgff/datasets/single_loader.py†L65-L118】【F:lgff/datasets/single_loader.py†L298-L383】
- 未实现 random crop/affine/flip 等会改变像素几何的增强；因此 GT pose 与深度/内参无额外变换。验证/测试 split 仍沿用同样的 resize（无随机性），不会破坏 BOP GT 对齐。【F:lgff/datasets/single_loader.py†L278-L352】【F:lgff/datasets/single_loader.py†L383-L444】
- 若未来加入随机仿射/裁剪，应同时更新 `K`（fx/cx, fy/cy）与 mask/pose；可参考 `_load_depth` 内的缩放逻辑封装为单独函数，供新增强调用。

## 随机性控制与复现性
- `train_lgff_sc.py` 只设置了 Python `random` 与 PyTorch（含 CUDA）种子，未固定 NumPy 或 DataLoader worker 的随机性；`deterministic=False` 也让 cudnn 算子保持非确定性。【F:lgff/train_lgff_sc.py†L4-L88】
- 若需要严格复现，可在启动时添加：
  - `np.random.seed(seed)`；
  - DataLoader 增加 `worker_init_fn` 或 `generator=torch.Generator().manual_seed(seed)`；
  - 根据需求设置 `torch.use_deterministic_algorithms(True)` 并关闭部分非确定算子。
- 推荐在 `set_random_seed` 中补充 NumPy 种子和可选 `torch.backends.cuda.matmul.allow_tf32=False`，并在构建 DataLoader 时传入 `worker_init_fn`，使每个 worker 拥有基于全局种子的偏移，提升实验可复现性。
