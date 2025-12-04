"""
LGFF 多类别数据加载脚本，占位用于后续扩展。
规划用于类条件/多物体训练的数据集读取逻辑，当前 ``MultiObjectDataset``
仅提供接口骨架并抛出未实现异常，提醒在未来阶段补充具体实现。
"""

from torch.utils.data import Dataset


class MultiObjectDataset(Dataset):
    def __init__(self, *args, **kwargs):  # pragma: no cover - placeholder
        super().__init__()
        raise NotImplementedError("Class-conditional loader not yet implemented.")

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise NotImplementedError


__all__ = ["MultiObjectDataset"]
