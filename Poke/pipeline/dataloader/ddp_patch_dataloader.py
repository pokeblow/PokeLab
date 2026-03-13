from __future__ import annotations

from typing import Any, Optional

from .ddp_dataloader import PokeDDPDataloader
from .patch_dataloader import PatchCollateFn
from ..configure import PokeConfig


class PokeDDPPatchDataloader(PokeDDPDataloader):
    """
    DDP dataloader + patch extraction via collate_fn.
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
        train_drop_last: bool = True,
        eval_drop_last: bool = False,
        pin_memory: bool = True,
        persistent_workers: Optional[bool] = None,
        prefetch_factor: Optional[int] = None,
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
            train_drop_last=train_drop_last,
            eval_drop_last=eval_drop_last,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
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
