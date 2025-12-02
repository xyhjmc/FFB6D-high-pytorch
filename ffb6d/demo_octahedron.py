#!/usr/bin/env python3
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import os
import argparse
import tqdm
import cv2
import torch
import numpy as np
import pickle as pkl
from glob import glob

from common import Config, ConfigRandLA
from models.ffb6d import FFB6D
from datasets.ycb.ycb_dataset import Dataset as YCB_Dataset
from datasets.linemod.linemod_dataset import Dataset as LM_Dataset
from utils.pvn3d_eval_utils_kpls import cal_frame_poses, cal_frame_poses_lm
from utils.basic_utils import Basic_Utils

try:
    from neupeak.utils.webcv2 import imshow, waitKey
except ImportError:
    from cv2 import imshow, waitKey


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Visualize predicted poses as cube pose boxes with axes"
    )
    parser.add_argument("-checkpoint", type=str, default=None, help="Checkpoint to eval")
    parser.add_argument(
        "-dataset",
        type=str,
        default="linemod",
        help="Target dataset, ycb or linemod. (linemod as default).",
    )
    parser.add_argument(
        "-cls",
        type=str,
        default="ape",
        help=(
            "Target object to eval in LineMOD dataset. (ape, benchvise, cam, can,"
            "cat, driller, duck, eggbox, glue, holepuncher, iron, lamp, phone)"
        ),
    )
    parser.add_argument("-show", action="store_true", help="View from imshow or not.")
    parser.add_argument(
        "-num_workers",
        type=int,
        default=None,
        help="Number of dataloader workers (default: 0 when showing, else 4).",
    )
    parser.add_argument(
        "-input_dir",
        type=str,
        default=None,
        help="Optional directory of inputs to visualize (pkl files or image stems).",
    )
    parser.add_argument(
        "-input_image",
        type=str,
        default=None,
        help="Optional single image or pickle path to visualize.",
    )
    parser.add_argument(
        "-scale",
        type=float,
        default=1.1,
        help="Multiplicative scale applied to the canonical cube half-edge length.",
    )
    return parser


def ensure_fd(fd):
    if not os.path.exists(fd):
        os.system("mkdir -p {}".format(fd))


def load_checkpoint(model=None, optimizer=None, filename="checkpoint"):
    filename = "{}.pth.tar".format(filename)

    assert os.path.isfile(filename), "==> Checkpoint '{}' not found".format(filename)
    print("==> Loading from checkpoint '{}'".format(filename))
    try:
        checkpoint = torch.load(filename)
    except Exception:
        checkpoint = pkl.load(open(filename, "rb"))
    epoch = checkpoint.get("epoch", 0)
    it = checkpoint.get("it", 0.0)
    best_prec = checkpoint.get("best_prec", None)
    if model is not None and checkpoint["model_state"] is not None:
        ck_st = checkpoint["model_state"]
        if "module" in list(ck_st.keys())[0]:
            tmp_ck_st = {}
            for k, v in ck_st.items():
                tmp_ck_st[k.replace("module.", "")] = v
            ck_st = tmp_ck_st
        model.load_state_dict(ck_st)
    if optimizer is not None and checkpoint["optimizer_state"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    print("==> Done")
    return it, epoch, best_prec


class CubeProjector:
    def __init__(self, bs_utils, dataset, scale=1.1):
        self.bs_utils = bs_utils
        self.dataset = dataset
        self.scale = scale
        self.radius_cache = {}
        self.cube_cache = {}

    def _canonical_radius(self, obj_id):
        if obj_id in self.radius_cache:
            return self.radius_cache[obj_id]
        mesh_pts = self.bs_utils.get_pointxyz(obj_id, ds_type=self.dataset)
        radius = np.linalg.norm(mesh_pts, axis=1).max()
        self.radius_cache[obj_id] = radius
        return radius

    def _canonical_vertices(self, obj_id):
        if obj_id in self.cube_cache:
            return self.cube_cache[obj_id]
        r = self._canonical_radius(obj_id) * self.scale
        coords = [-r, r]
        verts = np.array(
            [
                [x, y, z]
                for x in coords
                for y in coords
                for z in coords
            ],
            dtype=np.float32,
        )
        edges = [
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        ]
        axis_len = r * 1.2
        axes = np.array(
            [
                [0.0, 0.0, 0.0],
                [axis_len, 0.0, 0.0],
                [0.0, axis_len, 0.0],
                [0.0, 0.0, axis_len],
            ],
            dtype=np.float32,
        )
        self.cube_cache[obj_id] = (verts, edges, axes)
        return verts, edges, axes

    def draw(self, img, pose, obj_id, K):
        verts, edges, axes = self._canonical_vertices(obj_id)
        rotation = pose[:, :3]
        translation = pose[:, 3]
        verts_cam = np.dot(verts, rotation.T) + translation
        axes_cam = np.dot(axes, rotation.T) + translation
        verts_2d = self.bs_utils.project_p3d(verts_cam, 1.0, K)
        axes_2d = self.bs_utils.project_p3d(axes_cam, 1.0, K)

        color = self.bs_utils.get_label_color(obj_id, n_obj=22, mode=2)
        for i0, i1 in edges:
            p0 = tuple(verts_2d[i0].tolist())
            p1 = tuple(verts_2d[i1].tolist())
            img = cv2.line(img, p0, p1, color=color, thickness=2)
        img = self.bs_utils.draw_p2ds(img, verts_2d, r=3, color=color)

        axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for i, c in enumerate(axis_colors, start=1):
            p0 = tuple(axes_2d[0].tolist())
            p1 = tuple(axes_2d[i].tolist())
            img = cv2.arrowedLine(img, p0, p1, color=c, thickness=2, tipLength=0.1)

        img = self.draw_info_box(img, pose)
        return img

    @staticmethod
    def format_homogeneous_matrix(pose):
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = pose[:, :3]
        T[:3, 3] = pose[:, 3]
        return "\n".join(["\t".join(["{:.6f}".format(v) for v in row]) for row in T])

    def draw_info_box(self, img, pose):
        text_lines = ["camera_T_object"]
        text_lines.extend(self.format_homogeneous_matrix(pose).split("\n"))

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        line_height = 0
        box_width = 0
        for line in text_lines:
            (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
            box_width = max(box_width, w)
            line_height = max(line_height, h)
        padding = 6
        line_gap = 4
        box_height = len(text_lines) * (line_height + line_gap) + padding * 2 - line_gap

        x0, y0 = 10, 10
        x1, y1 = x0 + box_width + padding * 2, y0 + box_height

        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, dst=img)

        for i, line in enumerate(text_lines):
            y = y0 + padding + (line_height + line_gap) * (i + 1) - line_gap
            cv2.putText(
                img,
                line,
                (x0 + padding, y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                lineType=cv2.LINE_AA,
            )
        return img


def resolve_custom_inputs(args):
    if args.input_dir is None and args.input_image is None:
        return None

    custom_list = []
    if args.input_dir is not None:
        if not os.path.isdir(args.input_dir):
            raise FileNotFoundError("Input directory not found: {}".format(args.input_dir))
        pkl_files = sorted(glob(os.path.join(args.input_dir, "*.pkl")))
        if len(pkl_files) > 0:
            custom_list.extend(pkl_files)
        png_files = sorted(glob(os.path.join(args.input_dir, "*.png")))
        stems = [os.path.splitext(os.path.basename(p))[0] for p in png_files]
        custom_list.extend(stems)

    if args.input_image is not None:
        if args.input_image.endswith(".pkl"):
            custom_list.append(args.input_image)
        else:
            custom_list.append(os.path.splitext(os.path.basename(args.input_image))[0])

    seen = set()
    uniq_list = []
    for item in custom_list:
        if item not in seen:
            uniq_list.append(item)
            seen.add(item)
    return uniq_list if len(uniq_list) > 0 else None


def cal_view_pred_pose(args, config, bs_utils, projector, model, data, epoch=0, obj_id=-1):
    model.eval()
    with torch.set_grad_enabled(False):
        cu_dt = {}
        for key in data.keys():
            if data[key].dtype in [np.float32, np.uint8]:
                cu_dt[key] = torch.from_numpy(data[key].astype(np.float32)).cuda()
            elif data[key].dtype in [np.int32, np.uint32]:
                cu_dt[key] = torch.LongTensor(data[key].astype(np.int32)).cuda()
            elif data[key].dtype in [torch.uint8, torch.float32]:
                cu_dt[key] = data[key].float().cuda()
            elif data[key].dtype in [torch.int32, torch.int16]:
                cu_dt[key] = data[key].long().cuda()
        end_points = model(cu_dt)
        _, classes_rgbd = torch.max(end_points["pred_rgbd_segs"], 1)

        pcld = cu_dt["cld_rgb_nrm"][:, :3, :].permute(0, 2, 1).contiguous()
        if args.dataset == "ycb":
            pred_cls_ids, pred_pose_lst, _ = cal_frame_poses(
                pcld[0],
                classes_rgbd[0],
                end_points["pred_ctr_ofs"][0],
                end_points["pred_kp_ofs"][0],
                True,
                config.n_objects,
                True,
                None,
                None,
            )
        else:
            pred_pose_lst = cal_frame_poses_lm(
                pcld[0],
                classes_rgbd[0],
                end_points["pred_ctr_ofs"][0],
                end_points["pred_kp_ofs"][0],
                True,
                config.n_objects,
                False,
                obj_id,
            )
            pred_cls_ids = np.array([[1]])

        np_rgb = cu_dt["rgb"].cpu().numpy().astype("uint8")[0].transpose(1, 2, 0).copy()
        if args.dataset == "ycb":
            np_rgb = np_rgb[:, :, ::-1].copy()
        ori_rgb = np_rgb.copy()
        pose_logs = []
        for cls_id in cu_dt["cls_ids"][0].cpu().numpy():
            idx = np.where(pred_cls_ids == cls_id)[0]
            if len(idx) == 0:
                continue
            pose = pred_pose_lst[idx[0]]
            if args.dataset == "ycb":
                obj_id = int(cls_id[0])
            if args.dataset == "ycb":
                K = config.intrinsic_matrix["ycb_K1"]
            else:
                K = config.intrinsic_matrix["linemod"]
            np_rgb = projector.draw(np_rgb, pose, obj_id, K)
            pose_logs.append(
                "Object {} pose (camera_T_object):\n{}".format(
                    obj_id, projector.format_homogeneous_matrix(pose)
                )
            )
        vis_dir = os.path.join(config.log_eval_dir, "pose_vis_cube")
        ensure_fd(vis_dir)
        f_pth = os.path.join(vis_dir, "{}.jpg".format(epoch))
        log_pth = os.path.join(vis_dir, "{}_pose.txt".format(epoch))
        if args.dataset == "ycb":
            bgr = np_rgb
            ori_bgr = ori_rgb
        else:
            bgr = np_rgb[:, :, ::-1]
            ori_bgr = ori_rgb[:, :, ::-1]
        cv2.imwrite(f_pth, bgr)
        if pose_logs:
            with open(log_pth, "w") as f:
                f.write("\n\n".join(pose_logs))
        if args.show:
            imshow("cube_pose_rgb", bgr)
            imshow("original_rgb", ori_bgr)
            key = waitKey()
            window_closed = False
            try:
                window_closed = cv2.getWindowProperty(
                    "cube_pose_rgb", cv2.WND_PROP_VISIBLE
                ) < 1
            except Exception:
                pass
            if key in [ord("q"), 27] or window_closed:
                return False
    if epoch == 0:
        print("\n\nResults saved in {}".format(vis_dir))
    return True


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.dataset == "ycb":
        config = Config(ds_name=args.dataset)
        obj_id = -1
        dataset = "ycb"
    else:
        config = Config(ds_name=args.dataset, cls_type=args.cls)
        obj_id = config.lm_obj_dict[args.cls]
        dataset = "linemod"
    bs_utils = Basic_Utils(config)
    projector = CubeProjector(bs_utils, dataset, scale=args.scale)

    if args.dataset == "ycb":
        test_ds = YCB_Dataset("test")
    else:
        test_ds = LM_Dataset("test", cls_type=args.cls)
    custom_items = resolve_custom_inputs(args)
    if custom_items is not None:
        test_ds.all_lst = custom_items
        print("Overriding test list with {} custom item(s).".format(len(custom_items)))
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = 0 if args.show else 4
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=config.test_mini_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    rndla_cfg = ConfigRandLA
    model = FFB6D(
        n_classes=config.n_objects,
        n_pts=config.n_sample_points,
        rndla_cfg=rndla_cfg,
        n_kps=config.n_keypoints,
    )
    model.cuda()

    if args.checkpoint is not None:
        load_checkpoint(model, None, filename=args.checkpoint[:-8])

    for i, data in tqdm.tqdm(enumerate(test_loader), leave=False, desc="val"):
        should_continue = cal_view_pred_pose(
            args, config, bs_utils, projector, model, data, epoch=i, obj_id=obj_id
        )
        if should_continue is False:
            print("Visualization interrupted by user close.")
            break


if __name__ == "__main__":
    main()

# vim: ts=4 sw=4 sts=4 expandtab
