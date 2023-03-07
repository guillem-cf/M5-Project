from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, data, targets, transform=None, target_transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.targets[index]
        if self.transform:
            image = self.transform(self.data[index])
        if self.target_transform:
            label = self.transform(self.targets[index])
        return image, label