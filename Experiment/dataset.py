import random

from torch.utils.data import Dataset
import torch
class RandomClsDataset(Dataset):
    def __init__(self, n=1000, in_dim=20, n_classes=3, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, in_dim, generator=g)
        self.y = torch.randint(low=0, high=n_classes, size=(n,), generator=g)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        self.case_name = f'image_{random.randint(0, 100)}'
        return self.case_name, self.x[idx], self.y[idx]