"""
目标：多类混合 Loader
要求 Codex：

在 FFB6D 项目中找现有的多物体 Dataset（如果有），或自己用 Python os.walk 遍历。

只要遵守 __getitem__ 返回的字段名，后面 engines 就可以统一处理

"""


class MultiObjectDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split, class_map, config):
        """
        root_dir: data/
        class_map: {'helmet':0, 'handwheel':1, ...}
        """
        # TODO: 遍历每个子目录，对每个类别构建索引，记录 (img_path, depth_path, obj_id, cls_id, ...)
        ...

    def __getitem__(self, idx):
        # TODO: 类似 SingleObjectDataset，额外加 'cls_id'
        sample = {...}
        sample["cls_id"] = torch.tensor(cls_id)
        return sample
