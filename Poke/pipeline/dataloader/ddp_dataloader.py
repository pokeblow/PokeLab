from __future__ import annotations

from typing import Optional

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .base_dataloader import PokeBaseDataloader
from ..configure import PokeConfig


class PokeDDPDataloader(PokeBaseDataloader):
    """
    DDP-friendly dataloader:
    - train/valid/test 使用 DistributedSampler
    - train sampler: shuffle=True
    - valid/test sampler: shuffle=False（保证评估稳定）
    - 需要每个 epoch 调用 set_epoch(epoch) 来更新 sampler 的随机种子
    """

    def __init__(
        self,
        config: PokeConfig,
        train_dataset=None,
        valid_dataset=None,
        test_dataset=None,
        *,
        train_drop_last: bool = True,
        eval_drop_last: bool = False,
        pin_memory: bool = True,
        persistent_workers: Optional[bool] = None,
        prefetch_factor: Optional[int] = None,
    ):
        self.train_drop_last = train_drop_last
        self.eval_drop_last = eval_drop_last
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor

        self.train_sampler: Optional[DistributedSampler] = None
        self.valid_sampler: Optional[DistributedSampler] = None
        self.test_sampler: Optional[DistributedSampler] = None

        super().__init__(
            config=config,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
        )

    def _is_dist(self) -> bool:
        return dist.is_available() and dist.is_initialized()

    def _sampler(self, dataset, *, shuffle: bool, drop_last: bool) -> Optional[DistributedSampler]:
        if dataset is None or not self._is_dist():
            return None
        return DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)

    def _make_loader(
        self,
        dataset,
        sampler: Optional[DistributedSampler],
        *,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader | None:
        if dataset is None:
            return None

        num_workers = self._num_workers()
        loader_shuffle = (sampler is None) and shuffle

        kwargs = dict(
            batch_size=self._batch_size(),
            shuffle=loader_shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
        )

        if num_workers > 0:
            if self.persistent_workers is not None:
                kwargs["persistent_workers"] = bool(self.persistent_workers)
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = int(self.prefetch_factor)

        return DataLoader(dataset, **kwargs)

    def set_train_dataloader(self) -> DataLoader | None:
        self.train_sampler = self._sampler(self.train_dataset, shuffle=True, drop_last=self.train_drop_last)
        return self._make_loader(
            self.train_dataset,
            self.train_sampler,
            shuffle=True,
            drop_last=self.train_drop_last,
        )

    def set_valid_dataloader(self) -> DataLoader | None:
        self.valid_sampler = self._sampler(self.valid_dataset, shuffle=False, drop_last=self.eval_drop_last)
        return self._make_loader(
            self.valid_dataset,
            self.valid_sampler,
            shuffle=False,
            drop_last=self.eval_drop_last,
        )

    def set_test_dataloader(self) -> DataLoader | None:
        self.test_sampler = self._sampler(self.test_dataset, shuffle=False, drop_last=self.eval_drop_last)
        return self._make_loader(
            self.test_dataset,
            self.test_sampler,
            shuffle=False,
            drop_last=self.eval_drop_last,
        )

    def set_epoch(self, epoch: int) -> None:
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        if self.valid_sampler is not None:
            self.valid_sampler.set_epoch(epoch)
        if self.test_sampler is not None:
            self.test_sampler.set_epoch(epoch)
