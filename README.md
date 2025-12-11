# LGFF Single-Object 6D Pose Estimation (PyTorch 2.x)

This repository contains a lightweight, high-PyTorch version of the single-object **LGFF** (FFB6D-inspired) 6D pose estimator. It targets BOP/LINEMOD objects (e.g., `ape`) with unified meter units, confidence-aware pose fusion, and visualization utilities.

## Coordinate & Unit Conventions
- **Camera frame, meters everywhere.** Depth from `scene_camera.json` is converted with `depth_m = depth_raw * depth_scale_scene / cfg.depth_scale`; CAD meshes are divided by `1000` when loaded.
- **Poses** are `[R|t]` with `R` (3×3) in the left block and `t` (meters) in the last column; all transforms map object points into the camera frame.
- **Points**
  - `points` / ROI depth points: camera frame (m), cropped/resized together with RGB and `K`.
  - `model_points`: CAD object frame (m), broadcast to batch when needed.
- **Intrinsics**: if RGB/depth are resized, `K` is scaled accordingly; projection helpers use `cam_scale=1.0` (meters) explicitly.

## Data Preparation
1. Place BOP-format data under `datasets/bop` (or set `cfg.dataset_root`).
2. Ensure `scene_camera.json` contains `cam_K` and `depth_scale`; the loader applies per-scene `depth_scale` and resizes RGB/depth/masks consistently.
3. Optional: provide `keypoints/obj_xxxxxx.npy` for predefined object-frame keypoints; otherwise they are sampled from the CAD points.

## Training
```bash
python lgff/train_lgff_sc.py --config lgff/configs/linemod_ape_sc_resnet34.yaml \
    --work-dir output/linemod_ape_sc
```
Key notes:
- Checkpoints store the **full config**; gradients use PyTorch AMP + GradScaler + configurable grad clip.
- Pose fusion during training/validation mirrors evaluation (`train_use_best_point` / `eval_use_best_point` or weighted Top-K via `pose_fusion_topk`).
- Loss terms include ROI DenseFusion-style geometry, translation, confidence regularization, optional CAD-level ADD/ADD-S, and optional keypoint offset supervision.

## Evaluation
```bash
python lgff/eval_sc.py --checkpoint output/linemod_ape_sc  # directory or .pth
```
Behavior:
- If a directory is provided, the evaluator loads `checkpoint_best.pth` first, falling back to `checkpoint_last.pth`.
- The config inside the checkpoint rebuilds the model/backbone; CLI overrides (e.g., batch size, val split) are kept.
- Metrics: ADD, ADD-S (per-sample symmetry), translation error, rotation error, CMD (<2 cm), and thresholded accuracies; per-image CSV is saved alongside logs.

## Visualization
```bash
python lgff/viz_sc.py --checkpoint output/linemod_ape_sc --num-samples 8 --show
```
Outputs per-sample images with:
- RGB overlays (cube + axes projected with `cam_scale=1.0`, meters).
- 3D scatter of CAD@GT vs CAD@Pred vs raw ROI depth.
- Title reports ADD/ADD-S, translation/rotation errors, CMD distance, and symmetry flag.

## Extending / Switching Backbones
- Backbones and head widths come from the checkpoint config; change them in YAML/`--opt` **before training**.
- Pose fusion strategy is configurable per stage (`train_use_best_point`, `eval_use_best_point`, `viz_use_best_point`, `pose_fusion_topk`, `loss_use_fused_pose`).
- Toggle loss branches via `lambda_add`, `lambda_add_cad`, `lambda_kp_of`, `lambda_conf`, `lambda_t`; monitor per-branch magnitudes in `metrics_history.csv`.

## Reproducibility
- Default seed is 42; set `--seed` or `cfg.seed` and `cfg.deterministic=True` for stricter CuDNN determinism.
- Dataloaders rescale intrinsics with image resizing; no random geometric augmentation is applied in evaluation/visualization paths.

## Repository Layout
- `lgff/datasets/single_loader.py`: BOP data loader with unit-consistent RGB/depth/mask/K handling and keypoint targets.
- `lgff/models/lgff_sc.py`: Single-class LGFF network.
- `lgff/losses/lgff_loss.py`: Multi-branch loss (ROI/CAD ADD/ADD-S, translation, confidence, keypoint offset) with NaN/Inf guards.
- `lgff/engines/trainer_sc.py`: AMP training loop, validation with unified metrics, checkpoint/history logging.
- `lgff/engines/evaluator_sc.py`: Inference-time pose fusion, ADD/ADD-S/CMD metrics, per-image CSV export.
- `lgff/viz_sc.py`: Visualization helper aligned with evaluator math.

Enjoy robust, unit-consistent 6D pose experiments! 中文用户：关键注释均在对应代码位置，强调“米制 / 相机坐标系 / 对称性判定 / 姿态融合策略”。

---

## Original FFB6D README (reference)

# FFB6D
This is the official source code for the **CVPR2021 Oral** work, **FFB6D: A Full Flow Biderectional Fusion Network for 6D Pose Estimation**. ([Arxiv](https://arxiv.org/abs/2103.02242), [Video_Bilibili](https://www.bilibili.com/video/BV1YU4y1a7Kp?from=search&seid=8306279574921937158), [Video_YouTube](https://www.youtube.com/watch?v=SSi2TnyD6Is))

## Table of Content

- [FFB6D](#ffb6d)
  - [Table of Content](#table-of-content)
  - [Introduction & Citation](#introduction--citation)
  - [Demo Video](#demo-video)
  - [Installation](#installation)
  - [Code Structure](#code-structure)
  - [Datasets](#datasets)
  - [Training and evaluating](#training-and-evaluating)
    - [Training on the LineMOD Dataset](#training-on-the-linemod-dataset)
    - [Evaluating on the LineMOD Dataset](#evaluating-on-the-linemod-dataset)
    - [Demo/visualizaion on the LineMOD Dataset](#demovisualizaion-on-the-linemod-dataset)
    - [Training on the YCB-Video Dataset](#training-on-the-ycb-video-dataset)
    - [Evaluating on the YCB-Video Dataset](#evaluating-on-the-ycb-video-dataset)
    - [Demo/visualization on the YCB-Video Dataset](#demovisualization-on-the-ycb-video-dataset)
    - [Octahedron pose visualization demo](#octahedron-pose-visualization-demo)
  - [Results](#results)
  - [Adaptation to New Dataset](#adaptation-to-new-dataset)
  - [License](#license)

## Introduction & Citation
<div align=center><img width="100%" src="figs/FFB6D_overview.png"/></div>

[FFB6D](https://arxiv.org/abs/2103.02242v1) is a general framework for representation learning from a single RGBD image, and we applied it to the 6D pose estimation task by cascading downstream prediction headers for instance semantic segmentation and 3D keypoint voting prediction from PVN3D([Arxiv](https://arxiv.org/abs/1911.04231), [Code](https://github.com/ethnhe/PVN3D), [Video](https://www.bilibili.com/video/av89408773/)).
At the representation learning stage of FFB6D, we build **bidirectional** fusion modules in the **full flow** of the two networks, where fusion is applied to each encoding and decoding layer. In this way, the two networks can leverage local and global complementary information from the other one to obtain better representations. Moreover, at the output representation stage, we designed a simple but effective 3D keypoints selection algorithm considering the texture and geometry information of objects, which simplifies keypoint localization for precise pose estimation.

Please cite [FFB6D](https://arxiv.org/abs/2103.02242v1) & [PVN3D](https://arxiv.org/abs/1911.04231) if you use this repository in your publications:

```
@InProceedings{He_2021_CVPR,
author = {He, Yisheng and Huang, Haibin and Fan, Haoqiang and Chen, Qifeng and Sun, Jian},
title = {FFB6D: A Full Flow Bidirectional Fusion Network for 6D Pose Estimation},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2021}
}

@InProceedings{He_2020_CVPR,
author = {He, Yisheng and Sun, Wei and Huang, Haibin and Liu, Jianran and Fan, Haoqiang and Sun, Jian},
title = {PVN3D: A Deep Point-Wise 3D Keypoints Voting Network for 6DoF Pose Estimation},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

## Demo Video
See our demo video on [YouTube](https://www.youtube.com/watch?v=SSi2TnyD6Is) or [bilibili](https://www.bilibili.com/video/BV1YU4y1a7Kp?from=search&seid=8306279574921937158).
## Installation
- Use Python **3.10** and a CUDA-enabled GPU (tested with CUDA 12.6 / RTX 4060).
- Install PyTorch **2.X** with the CUDA 12.6 wheels, e.g.:
  ```shell
  pip3 install torch==2.9.1+cu126 torchvision==0.24.1+cu126 --index-url https://download.pytorch.org/whl/cu126
  ```
- Install dependencies from requirement.txt:
  ```shell
  pip3 install -r requirement.txt
  ```
- Surface normal estimation is handled directly in PyTorch and will run on GPU when available; no external normalSpeed dependency is needed.
- Install tkinter through ``sudo apt install python3-tk``

- Custom CUDA/C++ operators are no longer required. KNN search, grid subsampling, and FPS have been reimplemented in PyTorch/NumPy so the project builds cleanly on PyTorch 2.X.

## Code Structure
<details>
  <summary>[Click to expand]</summary>

- **ffb6d**
  - **ffb6d/common.py**: Common configuration of dataset and models, eg. dataset path, keypoints path, batch size and so on.
  - **ffb6d/datasets**
    - **ffb6d/datasets/linemod/**
      - **ffb6d/datasets/linemod/linemod_dataset.py**: Data loader for LineMOD dataset.
      - **ffb6d/datasets/linemod/dataset_config/models_info.yml**: Object model info of LineMOD dataset.
      - **ffb6d/datasets/linemod/kps_orb9_fps**
        - **ffb6d/datasets/linemod/kps_orb9_fps/{obj_name}_8_kps.txt**: ORB-FPS 3D keypoints of an object in the object coordinate system.
        - **ffb6d/datasets/linemod/kps_orb9_fps/{obj_name}_corners.txt**: 8 corners of the 3D bounding box of an object in the object coordinate system.
    - **ffb6d/datasets/ycb**
      - **ffb6d/datasets/ycb/ycb_dataset.py**： Data loader for YCB_Video dataset.
        - **ffb6d/datasets/ycb/dataset_config/classes.txt**: Object list of YCB_Video dataset.
        - **ffb6d/datasets/ycb/dataset_config/radius.txt**: Radius of each object in YCB_Video dataset.
        - **ffb6d/datasets/ycb/dataset_config/train_data_list.txt**: Training set of YCB_Video datset.
        - **ffb6d/datasets/ycb/dataset_config/test_data_list.txt**: Testing set of YCB_Video dataset.
      - **ffb6d/datasets/ycb/ycb_kps**
        - **ffb6d/datasets/ycb/ycb_kps/{obj_name}_8_kps.txt**: ORB-FPS 3D keypoints of an object in the object coordinate system.
        - **ffb6d/datasets/ycb/ycb_kps/{obj_name}_corners.txt**: 8 corners of the 3D bounding box of an object in the object coordinate system.
  - **ffb6d/models**
    - **ffb6d/models/ffb6d.py**: Network architecture of the proposed FFB6D.
    - **ffb6d/models/cnn**
      - **ffb6d/models/cnn/extractors.py**: Resnet backbones.
      - **ffb6d/models/cnn/pspnet.py**: PSPNet decoder.
      - **ffb6d/models/cnn/ResNet_pretrained_mdl**: Resnet pretraiend model weights.
    - **ffb6d/models/loss.py**: loss calculation for training of FFB6D model.
    - **ffb6d/models/pytorch_utils.py**: pytorch basic network modules.
    - **ffb6d/models/RandLA/**: pytorch version of RandLA-Net from [RandLA-Net-pytorch](https://github.com/qiqihaer/RandLA-Net-pytorch)
  - **ffb6d/utils**
    - **ffb6d/utils/basic_utils.py**: basic functions for data processing, visualization and so on.
    - **ffb6d/utils/meanshift_pytorch.py**: pytorch version of meanshift algorithm for 3D center point and keypoints voting.
    - **ffb6d/utils/pvn3d_eval_utils_kpls.py**: Object pose esitimation from predicted center/keypoints offset and evaluation metrics.
    - **ffb6d/utils/ip_basic**: Image Processing for Basic Depth Completion from [ip_basic](https://github.com/kujason/ip_basic).
    - **ffb6d/utils/dataset_tools**
      - **ffb6d/utils/dataset_tools/DSTOOL_README.md**: README for dataset tools.
      - **ffb6d/utils/dataset_tools/requirement.txt**: Python3 requirement for dataset tools.
      - **ffb6d/utils/dataset_tools/gen_obj_info.py**: Generate object info, including SIFT-FPS 3d keypoints, radius etc.
      - **ffb6d/utils/dataset_tools/rgbd_rnder_sift_kp3ds.py**: Render rgbd images from mesh and extract textured 3d keypoints (SIFT/ORB).
      - **ffb6d/utils/dataset_tools/utils.py**: Basic utils for mesh, pose, image and system processing.
      - **ffb6d/utils/dataset_tools/fps**: Furthest point sampling algorithm.
      - **ffb6d/utils/dataset_tools/example_mesh**: Example mesh models.
  - **ffb6d/train_ycb.py**: Training & Evaluating code of FFB6D models for the YCB_Video dataset.
  - **ffb6d/demo.py**: Demo code for visualization.
  - **ffb6d/train_ycb.sh**: Bash scripts to start the training on the YCB_Video dataset.
  - **ffb6d/test_ycb.sh**: Bash scripts to start the testing on the YCB_Video dataset.
  - **ffb6d/demo_ycb.sh**: Bash scripts to start the demo on the YCB_Video_dataset.
  - **ffb6d/train_lm.py**: Training & Evaluating code of FFB6D models for the LineMOD dataset.
  - **ffb6d/train_lm.sh**: Bash scripts to start the training on the LineMOD dataset.
  - **ffb6d/test_lm.sh**: Bash scripts to start the testing on the LineMOD dataset.
  - **ffb6d/demo_lm.sh**: Bash scripts to start the demo on the LineMOD dataset.

## Datasets
We follow the datasets used in PVN3D and PoseCNN, namely **LINEMOD** and **YCB-Video** datasets.

### LineMOD
After downloading the LineMOD dataset and the corresponding object models provided by the BOP challenge, please organize the files in the following file structure:
```
├─datasets
│  ├─linemod
│  │  ├─benchvise
│  │  │  ├─data
│  │  │  ├─mask
│  │  │  ├─mask_visib
│  │  │  ├─scene_gt.json
│  │  │  ├─scene_gt_info.json
│  │  │  ├─scene_camera.json
│  │  ├─...
│  │  ├─ape
│  │  │  ├─data
│  │  │  ├─mask
│  │  │  ├─mask_visib
│  │  │  ├─scene_gt.json
│  │  │  ├─scene_gt_info.json
│  │  │  ├─scene_camera.json
│  ├─models
│  │  ├─obj_000001.ply
│  │  ├─...
│  │  ├─obj_000015.ply
```
Then download our processed files from the [Link](https://drive.google.com/file/d/14N455lIZxZkHqOR2xH3iRf7wrphu4rqy/view?usp=sharing) and put them under the directory `datasets/linemod/`.
```
datasets/linemod
│   keypoints
│   list.txt
│   models_info.json
```

### YCB-Video
Download the YCB-Video dataset including all the provided files (color images, depth maps, camera poses, segmentation masks, 3D models) from the official [project page](https://rse-lab.cs.washington.edu/projects/posecnn/). After downloading, extract the files under the directory `datasets/ycb/`.
Download the processed files from the [Link](https://drive.google.com/file/d/1yhzRGgIhjgRMqMRAwWomJDoMa20o2aPC/view?usp=sharing) and put them under the directory `datasets/ycb/`.
```
datasets/ycb
│   classes.txt
│   radius.txt
│   test_data_list.txt
│   train_data_list.txt
│   val_data_list.txt
│   ycb_meshes_google
│   ycb_kps
│   ycb_video_data
```

## Training and evaluating
Download our checkpoints (trained using PyTorch 1.1 and CUDA9.2) and put them in `output/` directory

LineMOD checkpoints: [Link](https://drive.google.com/drive/folders/13m-cCpq63NBVsoTUoQPmMXEcL40QePQa)

YCB-Video checkpoints: [Link](https://drive.google.com/drive/folders/1pG_tDjK8FqpMTT7o4nxwBFtAjkqZOGOO)

### Training on the LineMOD Dataset
```
python3 -m torch.distributed.launch --nproc_per_node=2 --use_env train_lm.py
```

### Evaluating on the LineMOD Dataset
```
python3 -m torch.distributed.launch --nproc_per_node=2 --use_env test_lm.py
```

### Demo/visualizaion on the LineMOD Dataset
```
python3 demo_lm.py
```

### Training on the YCB-Video Dataset
```
python3 -m torch.distributed.launch --nproc_per_node=2 --use_env train_ycb.py
```

### Evaluating on the YCB-Video Dataset
```
python3 -m torch.distributed.launch --nproc_per_node=2 --use_env test_ycb.py
```

### Demo/visualization on the YCB-Video Dataset
```
python3 demo_ycb.py
```

### Octahedron pose visualization demo
```
python3 demo.py
```

## Results
<details>
  <summary>[Click to expand]</summary>

### Results on LineMOD dataset. Success Rate of ADD(-S) AUC

|   Methods  |Average |Ape | Benchvise | Camera | Can | Cat | Driller | Duck | Eggbox | Glue | Holepuncher |
|---|---|---|---|---|---|---|---|---|---|---|---|
|PVN3D | 86.27 | 97.28 | 99.78 | 86.87 | 95.66 | 99.78 | 99.80 | 95.67 | 99.78 | 99.79 | 92.35|
|FFB6D(ours) | **95.70** | **99.79** | **100.0** | **93.85** | **98.42** | **99.87** | **100.** | **99.78** | **100.** | **99.83** | **97.43**|

### Results on LineMOD dataset. (3D 50mm)<br>

| Methods |Average |Ape | Benchvise | Camera | Can | Cat | Driller | Duck | Eggbox | Glue | Holepuncher |
|---|---|---|---|---|---|---|---|---|---|---|---|
|PVN3D | 98.61 | 98.51 | 99.90 | 99.67 | 99.40 | 99.33 | 99.99 | 98.20 | 99.46 | 97.57 | 93.59|
|FFB6D(ours)| **99.72** | **99.38** | **100.00** | **99.84** | **99.82** | **99.82** | **100.00** | **99.39** | **99.71** | **99.52** | **98.65**|

### Results on YCB-Video dataset. ADD(-S)

|Methods |PoseCNN | Densefusion(w/o) | Densefusion(w/) | Densefusion(w/,re-trained) | DPOD | PVN3D |MaskedFusion | FFB6D(ours)|
|---|---|---|---|---|---|---|---|---|
|ADD(-S) | 75.7 | 79.0 | 92.5 | 86.2 | 95.2 | 95.5 | 96.8 | **96.6**|

### ADD(-S) AUC breakdown by object on the YCB-Video dataset
|Class |PoseCNN | DenseFusion(w/o ICP) | DenseFusion(with ICP) | PVN3D |MaskedFusion | FFB6D(ours) |
|---|---|---|---|---|---|---|
|002_master_chef_can | 95.8 | 96.4 | 95.2 | 97.8 | 97.4 | **97.6**|
|003_cracker_box | 92.7 | 92.5 | 95.5 | 99.7 | **99.7** | 99.6|
|004_sugar_box | 98.2 | 97.5 | 98.2 | **99.6** | 99.3 | 99.2|
|005_tomato_soup_can | 94.6 | 94.4 | 95.3 | **98.3** | 97.6 | 98.1|
|006_mustard_bottle | 97.0 | 96.4 | 97.0 | **99.7** | 99.2 | 99.6|
|007_tuna_fish_can | 93.8 | 94.3 | 95.1 | **99.5** | 99.3 | 99.4|
|008_pudding_box | 94.3 | 94.4 | 97.1 | **99.6** | 99.3 | 99.2|
|009_gelatin_box | 97.1 | 96.6 | 98.2 | **99.8** | **99.8** | 99.6|
|010_potted_meat_can | 85.5 | 89.3 | 91.3 | **97.7** | **97.7** | 97.3|
|011_banana | 84.7 | 84.7 | 91.2 | 95.8 | 96.8 | **96.9**|
|019_pitcher_base | 89.3 | 90.8 | 91.4 | **97.8** | 97.5 | 97.6|
|021_bleach_cleanser | 90.9 | 90.5 | 91.8 | **97.9** | 96.9 | 97.7|
|024_bowl | 78.0 | 84.7 | **88.2** | 84.6 | 90.1 | 87.5|
|025_mug | 81.8 | 80.4 | 81.7 | 95.8 | 96.3 | **96.6**|
|035_power_drill | 71.5 | 71.1 | 85.2 | **95.2** | 93.4 | 93.6|
|036_wood_block | 50.2 | 59.2 | **71.0** | 68.2 | 70.9 | 60.8|
|037_scissors | 75.4 | 75.1 | 81.0 | 88.6 | **90.1** | 86.5|
|040_large_marker | 76.9 | 72.5 | 74.4 | 89.6 | 88.3 | **90.9**|
|051_large_clamp | 58.7 | 63.5 | 75.2 | 81.8 | 80.1 | **83.8**|
|052_extra_large_clamp | 73.3 | 74.4 | 83.0 | 86.3 | **87.8** | 87.4|
|061_foam_brick | 51.8 | 51.6 | 76.3 | 80.5 | 82.0 | **88.6**|
|Average | 75.7 | 79.0 | 86.2 | 95.5 | 96.8 | **96.6**|

### Comparison with self-implemented DenseFusion version on YCB-Video dataset. ADD(-S) AUC
|Class | Our Implementation | Official Implementation |
|---|---|---|
|soup can | 90.80 | 92.53|
|mustard | 90.24 | 89.80|
|drill | 86.58 | 79.41|
|horse | 46.06 | 37.56|
|helmet | 82.45 | 70.52|
|puncher | 77.40 | 63.33|
|Average | **78.26** | 71.36|

## Adaptation to New Dataset
If you are interested in adapting FFB6D to new dataset, you can read [Adaptation to New Dataset](https://github.com/ethnhe/FFB6D/blob/master/docs/DSTOOL_README.md).

## License
This project is released under the [S-Lab License 1.0](./LICENSE).
