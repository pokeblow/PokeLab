from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from .base_dataloader import PokeBaseDataloader
from ..configure import PokeConfig


def _normalize_hw(value: int | tuple[int, int] | list[int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")
        return value, value

    if isinstance(value, (tuple, list)) and len(value) == 2:
        h, w = int(value[0]), int(value[1])
        if h <= 0 or w <= 0:
            raise ValueError(f"{name} must be positive, got {(h, w)}")
        return h, w

    raise TypeError(f"{name} must be int or 2-item tuple/list, got {type(value)}")


class PatchCollateFn:
    """
    Convert one sample into multiple sliding-window patches during collation.

    - Finds the first tensor/ndarray (>=2 dims) as patch anchor.
    - Applies the same crop window to all tensor/ndarray fields with matching H,W.
    - Keeps non-spatial fields unchanged (repeated per patch).
    """

    def __init__(
        self,
        patch_size: int | tuple[int, int],
        patch_stride: int | tuple[int, int] | None = None,
    ):
        self.patch_size = _normalize_hw(patch_size, "patch_size")
        self.patch_stride = _normalize_hw(patch_stride if patch_stride is not None else patch_size, "patch_stride")

    def __call__(self, batch: list[Any]):
        expanded_batch: list[Any] = []
        for sample in batch:
            anchor_hw = self._find_anchor_hw(sample)
            if anchor_hw is None:
                expanded_batch.append(sample)
                continue

            h, w = anchor_hw
            ph, pw = self.patch_size
            if h < ph or w < pw:
                raise ValueError(
                    f"Patch size {self.patch_size} is larger than sample spatial size {(h, w)}. "
                    f"Please reduce patch_size or resize input."
                )

            for top, left in self._iter_positions(h, w):
                expanded_batch.append(self._slice_sample(sample, top, left, anchor_hw))

        return default_collate(expanded_batch)

    def _iter_positions(self, h: int, w: int) -> Iterable[tuple[int, int]]:
        ph, pw = self.patch_size
        sh, sw = self.patch_stride

        ys = list(range(0, h - ph + 1, sh))
        xs = list(range(0, w - pw + 1, sw))

        # cover boundary to avoid dropping tail regions
        if ys[-1] != h - ph:
            ys.append(h - ph)
        if xs[-1] != w - pw:
            xs.append(w - pw)

        for y in ys:
            for x in xs:
                yield y, x

    def _find_anchor_hw(self, sample: Any) -> tuple[int, int] | None:
        if torch.is_tensor(sample) and sample.ndim >= 2:
            return int(sample.shape[-2]), int(sample.shape[-1])

        if isinstance(sample, np.ndarray) and sample.ndim >= 2:
            return int(sample.shape[-2]), int(sample.shape[-1])

        if isinstance(sample, dict):
            for value in sample.values():
                hw = self._find_anchor_hw(value)
                if hw is not None:
                    return hw
            return None

        if isinstance(sample, (list, tuple)):
            for value in sample:
                hw = self._find_anchor_hw(value)
                if hw is not None:
                    return hw
            return None

        return None

    def _slice_sample(self, sample: Any, top: int, left: int, anchor_hw: tuple[int, int]):
        ph, pw = self.patch_size

        if torch.is_tensor(sample) and sample.ndim >= 2 and tuple(sample.shape[-2:]) == anchor_hw:
            return sample[..., top : top + ph, left : left + pw]

        if isinstance(sample, np.ndarray) and sample.ndim >= 2 and tuple(sample.shape[-2:]) == anchor_hw:
            return sample[..., top : top + ph, left : left + pw]

        if isinstance(sample, dict):
            return {k: self._slice_sample(v, top, left, anchor_hw) for k, v in sample.items()}

        if isinstance(sample, list):
            return [self._slice_sample(v, top, left, anchor_hw) for v in sample]

        if isinstance(sample, tuple):
            return tuple(self._slice_sample(v, top, left, anchor_hw) for v in sample)

        return sample


class PokePatchDataloader(PokeBaseDataloader):
    """
    Base dataloader + patch extraction via collate_fn.
    """

    def __init__(
        self,
        config: PokeConfig,
        train_dataset=None,
        valid_dataset=None,
        test_dataset=None,
        *,
        patch_size: int | tuple[int, int] = (128, 128),
        patch_stride: int | tuple[int, int] | None = None,
        patch_on_train: bool = True,
        patch_on_valid: bool = True,
        patch_on_test: bool = False,
    ):
        self.patch_collate_fn = PatchCollateFn(patch_size=patch_size, patch_stride=patch_stride)
        self.patch_on_train = patch_on_train
        self.patch_on_valid = patch_on_valid
        self.patch_on_test = patch_on_test

        super().__init__(
            config=config,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
        )

    def _loader_kwargs(self, split: str) -> dict[str, Any]:
        use_patch = (
            (split == "train" and self.patch_on_train)
            or (split == "valid" and self.patch_on_valid)
            or (split == "test" and self.patch_on_test)
        )
        if not use_patch:
            return {}
        return {"collate_fn": self.patch_collate_fn}
