from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler


class CustomDataLoader:
    def __init__(self, dataset, batch_size, num_workers, shuffle, drop_last, distributed):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.distributed = distributed

    def __call__(self):
        if self.distributed:
            sampler = DistributedSampler(self.dataset)
            shuffle = False
        elif self.shuffle:
            sampler = RandomSampler(self.dataset)
            shuffle = False
        else:
            sampler = SequentialSampler(self.dataset)
            shuffle = False

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=shuffle,
            drop_last=self.drop_last,
        )
