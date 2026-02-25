# trainer_2.py  —— simplified container-log compatible trainer
from __future__ import annotations

from typing import Optional, Any
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
import inspect

from .module import PokeBaseModule
from .configure import PokeConfig
from .utils import *

LOG_ROOT = "/Users/wanghaolin/GitHub/PokeLab/Method/logs/test"

from pathlib import Path

# 当前文件目录
BASE_DIR = Path(__file__).resolve().parent

# buffer 文件路径
BUFFER_FILE = BASE_DIR / "buffer.tmp"

def to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, (list, tuple)):
        return [t.to(device=device, dtype=torch.float32) for t in batch]
    return batch.to(device=device, dtype=torch.float32)


def write_buffer(content: str):
    with open(BUFFER_FILE, "w", encoding="utf-8") as f:
        f.write(content)




class PokeTrainer:
    def __init__(
        self,
        config: PokeConfig,
        train_module: PokeConfig,
        train_loader: Optional[DataLoader] = None,
        valid_loader: Optional[DataLoader] = None,
    ):
        self.train_module = train_module
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = train_module.device

        self.config = config
        self.epochs = int(self.config.run.get("epochs", 10))

        self.epoch_idx = self.train_module.epoch
        self.step_idx = self.train_module.global_step

        LOG_ROOT = f'{get_caller_file().parent}/logs/{get_run_id()}'
        write_buffer(f'{LOG_ROOT}')

        self.log_root = Path(LOG_ROOT)

        if self.log_root.exists():
            shutil.rmtree(self.log_root)  # 删除整个目录（包含所有内容）

        self.log_root.mkdir(parents=True, exist_ok=True)

        # --------- summary log ---------
        self.summary_path = self.log_root / "summary.log"
        # 写一行 header（如果文件不存在）
        if not self.summary_path.exists():
            with open(self.summary_path, "a", encoding="utf-8") as f:
                f.write(f"# summary.log created at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        self.train_module.configure_parameters()

    # ---------- helpers ----------
    @staticmethod
    def leaf(log_item, split: str):
        """Return split-leaf if exists, else None. Works for container logs."""
        return getattr(log_item, split, None)

    def commit(self, split: str):
        for log in self.train_module.logs:
            l = self.leaf(log, split)
            if l is not None:
                l.commit_epoch()

    def log_line(self, split: str) -> str:
        s = ""
        for log in self.train_module.logs:
            if getattr(log, "log_type", "") == "image":
                continue
            if getattr(log, "log_type", "") == "lr" and split == 'valid':
                continue
            l = self.leaf(log, split)
            if l is not None:
                part = l.last_epoch_summary(log=True)
                if part:
                    s += part
        return s

    def append_summary(self, text: str) -> None:
        """Append a single line to LOG_ROOT/summary.log."""
        with open(self.summary_path, "a", encoding="utf-8") as f:
            f.write(text.rstrip("\n") + "\n")

    def save_images(self, split: str, epoch: int):
        for log in self.train_module.logs:
            if getattr(log, "log_type", "") != "image":
                continue
            l = self.leaf(log, split)
            if l is None:
                continue
            imgs = l.get_buffer_image()
            for i, box in enumerate(imgs):
                fig = self.train_module.visualization(box)
                path = self.log_root / f"{log.log_name}_epoch_{epoch+1}_{i+1}.png"
                fig.savefig(str(path), dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"Visualization results saved to {path}")

    # ---------- main ----------
    def fit(self) -> None:
        if self.train_loader is None or self.valid_loader is None:
            raise RuntimeError("train_loader and valid_loader must be provided.")


        for epoch in range(self.epoch_idx, self.epochs):
            self.train_module.epoch = epoch
            if hasattr(self.train_module, "train"):
                self.train_module.train()

            # ---- train ----
            for idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} • train")):
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    batch = batch[1:]
                batch = to_device(batch, self.device)
                self.train_module.train_step(idx, batch)
                self.train_module.global_step += 1

            self.commit("train")

            # ---- valid ----
            with torch.no_grad():
                for idx, batch in enumerate(tqdm(self.valid_loader, desc=f"Epoch {epoch+1}/{self.epochs} • valid")):
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        batch = batch[1:]
                    batch = to_device(batch, self.device)
                    self.train_module.valid_step(idx, batch)

            self.save_images("valid", epoch)
            self.commit("valid")

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # 注意这里用 'valid'（你原来是 'val'，会导致取不到 leaf）
            msg = (
                f"[{now}] Epoch {epoch+1}/{self.epochs}: "
                f"train: {self.log_line('train')} | valid: {self.log_line('valid')}"
            )
            print(msg)
            self.append_summary(msg)

            if hasattr(self.train_module, "save_parameters"):
                self.train_module.save_parameters()
            if hasattr(self.train_module, "set_scheduler"):
                self.train_module.set_scheduler()