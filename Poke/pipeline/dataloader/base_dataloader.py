from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from ..configure import PokeConfig


class PokeBaseDataloader:
    """
    Base dataloader for train/valid/test splits.

    Extension points:
    - `_loader_kwargs(split)`: inject split-specific DataLoader kwargs (e.g. collate_fn)
    - `set_*_dataloader()`: override when sampler/build logic differs (e.g. DDP)
    """

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

    def _persistent_workers(self) -> bool | None:
        value = self.config.dataloader.get("persistent_workers", None)
        return None if value is None else bool(value)

    def _prefetch_factor(self) -> int | None:
        value = self.config.dataloader.get("prefetch_factor", None)
        return None if value is None else int(value)

    def _loader_kwargs(self, split: str) -> dict[str, Any]:
        return {}

    def _build_loader(self, dataset, *, shuffle: bool, drop_last: bool, split: str) -> DataLoader | None:
        if dataset is None:
            return None

        num_workers = self._num_workers()
        kwargs: dict[str, Any] = dict(
            batch_size=self._batch_size(),
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=self._pin_memory(),
        )

        if num_workers > 0:
            persistent_workers = self._persistent_workers()
            prefetch_factor = self._prefetch_factor()
            if persistent_workers is not None:
                kwargs["persistent_workers"] = persistent_workers
            if prefetch_factor is not None:
                kwargs["prefetch_factor"] = prefetch_factor

        kwargs.update(self._loader_kwargs(split))
        return DataLoader(dataset, **kwargs)

    def set_train_dataloader(self) -> DataLoader | None:
        return self._build_loader(self.train_dataset, shuffle=True, drop_last=True, split="train")

    def set_valid_dataloader(self) -> DataLoader | None:
        return self._build_loader(self.valid_dataset, shuffle=False, drop_last=False, split="valid")

    def set_test_dataloader(self) -> DataLoader | None:
        return self._build_loader(self.test_dataset, shuffle=False, drop_last=False, split="test")
