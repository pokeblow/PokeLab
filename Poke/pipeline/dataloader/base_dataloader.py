from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ..configure import PokeConfig


class PokeBaseDataloader:
    def __init__(self, config: PokeConfig, train_dataset=None, valid_dataset=None, test_dataset=None):
        super().__init__()
        self.config = config
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

        self.train_loader = self.set_train_dataloader()
        self.valid_loader = self.set_valid_dataloader()
        self.test_loader = self.set_test_dataloader() if self.test_dataset is not None else None

    def _batch_size(self) -> int:
        return int(self.config.run.get("batch_size", 1) or 1)

    def _num_workers(self) -> int:
        return int(self.config.dataloader.get("num_workers", 0) or 0)

    def _pin_memory(self) -> bool:
        default_pin = torch.cuda.is_available()
        return bool(self.config.dataloader.get("pin_memory", default_pin))

    def _build_loader(self, dataset, *, shuffle: bool, drop_last: bool) -> DataLoader | None:
        if dataset is None:
            return None

        return DataLoader(
            dataset,
            batch_size=self._batch_size(),
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self._num_workers(),
            pin_memory=self._pin_memory(),
        )

    def set_train_dataloader(self) -> DataLoader | None:
        return self._build_loader(self.train_dataset, shuffle=True, drop_last=True)

    def set_valid_dataloader(self) -> DataLoader | None:
        return self._build_loader(self.valid_dataset, shuffle=False, drop_last=False)

    def set_test_dataloader(self) -> DataLoader | None:
        return self._build_loader(self.test_dataset, shuffle=False, drop_last=False)
