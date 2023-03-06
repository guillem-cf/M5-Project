from week1.datamodules.datasets.base_dataset import BaseDataset


class MITDataset(BaseDataset):
    def __init__(self, data, targets, transform=None, target_transform=None):
        super().__init__(data, targets, transform, target_transform)

