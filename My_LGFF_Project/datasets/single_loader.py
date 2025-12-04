
"""
目标：单类 BOP 格式 loader，只对应一个物体（如 helmet）
让 Codex：
打开 FFB6D 项目里的 YCBV / LM / BOP 数据集类，例如：
ffb6d/datasets/bop_dataset.py 或类似文件。
仿照其结构实现 SingleObjectDataset

"""


class SingleObjectDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, object_name, split, config):
        """
        root_dir: data/helmet/ 这种 BOP 风格根目录
        object_name: 'helmet'
        split: 'train' or 'test'
        config: 传入一些尺寸、采样参数
        """
        # TODO: 建立样本索引（图像路径、深度路径、mask、pose、K 等）
        ...

    def __getitem__(self, idx):
        """
        返回一个 dict，约定字段名如下：
          'rgb':        [3,H,W] tensor
          'depth':      [1,H,W] tensor
          'K':          [3,3]   tensor
          'points_xyz': [N,3]   tensor (已采样)
          'points_uv':  [N,2]   tensor (原图像素坐标)
          'kp_3d_gt':   [K,3]   tensor
          'kp_2d_gt':   [K,2]   tensor
          'mask_gt':    [H,W]   or 点级 mask
        """
        # TODO: 用 common.geometry 的工具生成点云和采样
        raise NotImplementedError
