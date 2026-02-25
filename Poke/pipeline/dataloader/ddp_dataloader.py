from typing import Optional
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
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
        drop_last: bool = True,
        pin_memory: bool = True,
        persistent_workers: Optional[bool] = None,
        prefetch_factor: Optional[int] = None,
    ):
        self.drop_last = drop_last
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

    # --------- helpers ---------
    def _is_dist(self) -> bool:
        return dist.is_available() and dist.is_initialized()

    def _sampler(self, dataset, shuffle: bool) -> Optional[DistributedSampler]:
        if dataset is None:
            return None
        if not self._is_dist():
            return None
        # 注意：DistributedSampler 会把数据分给每个 rank
        return DistributedSampler(dataset, shuffle=shuffle, drop_last=self.drop_last)

    def _num_workers(self) -> int:
        return int(self.config.dataloader.get("num_workers", 0) or 0)

    def _batch_size(self) -> int:
        # DDP下通常“每卡batch_size”，总batch_size = per_gpu * world_size
        return int(self.config.run.get("batch_size"))

    def _make_loader(
        self,
        dataset,
        sampler: Optional[DistributedSampler],
        shuffle: bool,
    ) -> DataLoader:
        num_workers = self._num_workers()

        # DDP 关键点：如果传了 sampler，shuffle 必须为 False
        loader_shuffle = (sampler is None) and shuffle

        kwargs = dict(
            batch_size=self._batch_size(),
            shuffle=loader_shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

        # 可选：persistent_workers / prefetch_factor（仅当 num_workers>0 才有意义）
        if num_workers > 0:
            if self.persistent_workers is not None:
                kwargs["persistent_workers"] = bool(self.persistent_workers)
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = int(self.prefetch_factor)

        return DataLoader(dataset, **kwargs)

    # --------- overrides ---------
    def set_train_dataloader(self) -> DataLoader:
        self.train_sampler = self._sampler(self.train_dataset, shuffle=True)
        return self._make_loader(self.train_dataset, self.train_sampler, shuffle=True)

    def set_valid_dataloader(self) -> DataLoader:
        self.valid_sampler = self._sampler(self.valid_dataset, shuffle=False)
        return self._make_loader(self.valid_dataset, self.valid_sampler, shuffle=False)

    def set_test_dataloader(self) -> DataLoader:
        self.test_sampler = self._sampler(self.test_dataset, shuffle=False)
        return self._make_loader(self.test_dataset, self.test_sampler, shuffle=False)

    # --------- ddp API ---------
    def set_epoch(self, epoch: int) -> None:
        """
        DDP 必须：每个 epoch 开始前调用，使 DistributedSampler 在各 epoch 产生不同 shuffle 顺序
        """
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        # valid/test 通常不 shuffle，不设也行；设了也不影响正确性
        if self.valid_sampler is not None:
            self.valid_sampler.set_epoch(epoch)
        if self.test_sampler is not None:
            self.test_sampler.set_epoch(epoch)