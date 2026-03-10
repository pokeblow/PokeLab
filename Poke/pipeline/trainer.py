from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import shutil

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .configure import PokeConfig
from .module import PokeBaseModule
from .utils import get_caller_file, get_run_id

BASE_DIR = Path(__file__).resolve().parent
BUFFER_FILE = BASE_DIR / "buffer.tmp"


def _move_to_device(item: Any, device: torch.device) -> Any:
    if torch.is_tensor(item):
        if item.is_floating_point():
            return item.to(device=device, dtype=torch.float32, non_blocking=True)
        return item.to(device=device, non_blocking=True)

    if isinstance(item, list):
        return [_move_to_device(v, device) for v in item]

    if isinstance(item, tuple):
        return tuple(_move_to_device(v, device) for v in item)

    if isinstance(item, dict):
        return {k: _move_to_device(v, device) for k, v in item.items()}

    return item


def _strip_leading_metadata(batch: Any) -> Any:
    if not isinstance(batch, (list, tuple)) or len(batch) == 0:
        return batch

    first = batch[0]
    # 常见数据格式：(case_id, x, y)；若首元素不是 Tensor，视作元信息剥离。
    if torch.is_tensor(first):
        return batch

    payload = batch[1:]
    if len(payload) == 1:
        return payload[0]
    return payload


def write_buffer(content: str) -> None:
    with open(BUFFER_FILE, "w", encoding="utf-8") as f:
        f.write(content)


class PokeTrainer:
    def __init__(
        self,
        config: PokeConfig,
        train_module: PokeBaseModule,
        train_loader: Optional[DataLoader] = None,
        valid_loader: Optional[DataLoader] = None,
    ):
        self.config = config
        self.train_module = train_module
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.device = self.train_module.device
        self.epochs = int(self.config.run.get("epochs", 10))

        # 从 checkpoint 恢复时，默认从下一个 epoch 开始，避免重复训练。
        start_epoch = int(getattr(self.train_module, "epoch", 0))
        if getattr(self.train_module, "global_step", 0) > 0:
            start_epoch += 1
        self.epoch_idx = min(start_epoch, self.epochs)

        log_root = f"{get_caller_file().parent}/logs/{get_run_id()}"
        write_buffer(log_root)
        self.log_root = Path(log_root)

        if self.log_root.exists():
            shutil.rmtree(self.log_root)
        self.log_root.mkdir(parents=True, exist_ok=True)

        self.summary_path = self.log_root / "summary.log"
        if not self.summary_path.exists():
            with open(self.summary_path, "a", encoding="utf-8") as f:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"# summary.log created at {now}\n")

        # 避免重复注册参数（某些模块会在外部预先调用 configure_parameters）。
        if not getattr(self.train_module, "parameters_config", []):
            self.train_module.configure_parameters()

    @staticmethod
    def leaf(log_item: Any, split: str):
        aliases = {
            "val": "valid",
            "validation": "valid",
        }
        split = aliases.get(split, split)
        return getattr(log_item, split, None)

    def commit(self, split: str) -> None:
        for log in self.train_module.logs:
            leaf_log = self.leaf(log, split)
            if leaf_log is not None:
                leaf_log.commit_epoch()

    def log_line(self, split: str) -> str:
        parts = []
        for log in self.train_module.logs:
            if getattr(log, "log_type", "") == "image":
                continue
            if getattr(log, "log_type", "") == "lr" and split == "valid":
                continue

            leaf_log = self.leaf(log, split)
            if leaf_log is None:
                continue

            part = leaf_log.last_epoch_summary(log=True)
            if part:
                parts.append(part.rstrip(", "))
        return ", ".join(parts)

    def append_summary(self, text: str) -> None:
        with open(self.summary_path, "a", encoding="utf-8") as f:
            f.write(text.rstrip("\n") + "\n")

    def save_images(self, split: str, epoch: int) -> None:
        for log in self.train_module.logs:
            if getattr(log, "log_type", "") != "image":
                continue

            leaf_log = self.leaf(log, split)
            if leaf_log is None:
                continue

            images = leaf_log.get_buffer_image()
            for i, image_box in enumerate(images):
                fig = self.train_module.visualization(image_box)
                path = self.log_root / f"{log.log_name}_epoch_{epoch + 1}_{i + 1}.png"
                fig.savefig(str(path), dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"Visualization results saved to {path}")

    def _set_dataloader_epoch(self, epoch: int) -> None:
        sampler = getattr(self.train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        valid_sampler = getattr(self.valid_loader, "sampler", None)
        if valid_sampler is not None and hasattr(valid_sampler, "set_epoch"):
            valid_sampler.set_epoch(epoch)

    def fit(self) -> None:
        if self.train_loader is None:
            raise RuntimeError("train_loader must be provided.")

        has_valid = self.valid_loader is not None

        for epoch in range(self.epoch_idx, self.epochs):
            self.train_module.epoch = epoch
            self._set_dataloader_epoch(epoch)

            if hasattr(self.train_module, "train"):
                self.train_module.train()

            for idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} • train")):
                batch = _strip_leading_metadata(batch)
                batch = _move_to_device(batch, self.device)
                self.train_module.train_step(idx, batch)
                self.train_module.global_step += 1

            self.commit("train")

            if has_valid:
                if hasattr(self.train_module, "eval"):
                    self.train_module.eval()

                with torch.no_grad():
                    for idx, batch in enumerate(tqdm(self.valid_loader, desc=f"Epoch {epoch + 1}/{self.epochs} • valid")):
                        batch = _strip_leading_metadata(batch)
                        batch = _move_to_device(batch, self.device)
                        self.train_module.valid_step(idx, batch)

                self.save_images("valid", epoch)
                self.commit("valid")

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sections = [f"train: {self.log_line('train')}"]
            if has_valid:
                sections.append(f"valid: {self.log_line('valid')}")
            msg = f"[{now}] Epoch {epoch + 1}/{self.epochs}: " + " | ".join(sections)

            print(msg)
            self.append_summary(msg)

            self.train_module.save_parameters()
            self.train_module.set_scheduler()
