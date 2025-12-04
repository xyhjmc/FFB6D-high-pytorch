"""TODO: Multi-class loader will be implemented in a later phase."""

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
