# poke_train.py  —— drop-in replacement
from __future__ import annotations
from typing import Dict, Tuple, Iterable, Optional
import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 如为同目录文件，请确保正确导入
from .poke_log import PokeLog  # <-- 在 poke_train 中使用 poke_log


# ========== 基础训练模块 ==========
class BaseTrainModule:
    """
    使用说明：
    1) 在子类中实现：train_step / valid_step，返回形如 {"loss": float, ...} 的字典；
    2) 在子类中重写 configure_logs() 来自定义要跟踪的指标；
    3) 如需模型保存，在子类中重写 configure_parameters_save() 与内部保存逻辑。
    """

    def __init__(self, config_path: str = ""):
        if config_path and os.path.isfile(config_path):
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logs = self._init_default_logs()

    # --------- 日志相关 ----------
    def _init_default_logs(self) -> None:
        if not isinstance(self.configure_logs(), tuple):
            return (self.configure_logs(),)
        else:
            return self.configure_logs()

    # --------- 需在子类实现 ----------
    def train_step(self, batch_idx: int, batch) -> Dict[str, float]:
        raise NotImplementedError

    def valid_step(self, batch_idx: int, batch) -> Dict[str, float]:
        raise NotImplementedError

    def configure_logs(self):
        """
        用户可重写，返回需要跟踪的日志集合。
        参考 _init_default_logs() 的文档字符串。
        """
        return None

    def configure_parameters_save(self):
        """
        用户可重写，配置模型保存逻辑（最佳/最后等）。
        在本实现中仅提供占位。
        """
        pass


# ========== 训练器 ==========
class PokeTrainer:
    def __init__(self, train_module: BaseTrainModule = None,
                 train_dataset=None, valid_dataset=None):
        self.train_module = train_module
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        # 训练基本配置（可被 configure() 覆盖）
        self.version = ""
        self.check_point = False
        self.epochs = 1
        self.batch_size = 1
        self.num_workers = 0

        self.train_loader: Optional[DataLoader] = None
        self.valid_loader: Optional[DataLoader] = None

    # --------- DataLoader ----------
    def _build_loaders(self) -> None:
        if self.train_dataset is not None:
            self.train_loader = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=self.num_workers,
            )
        if self.valid_dataset is not None:
            self.valid_loader = DataLoader(
                dataset=self.valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
            )

    # --------- 配置 ----------
    def configure(self) -> None:
        """
        从 train_module.config 读取训练超参并构建 DataLoader。
        兼容缺失字段的情况。
        """
        if self.train_module is None:
            raise ValueError("train_module 未提供。")

        cfg = self.train_module.config or {}

        self.version = cfg.get("version", self.version)
        self.check_point = cfg.get("check_point", self.check_point)
        self.epochs = int(cfg.get("epochs", self.epochs))
        self.batch_size = int(cfg.get("batch_size", self.batch_size))

        self.train_module.configure_parameters_save()

        self._build_loaders()

    # --------- 训练入口 ----------
    def fit(self) -> None:
        if self.train_loader is None and self.valid_loader is None:
            self._build_loaders()

        device = self.train_module.device  # 与模块保持一致

        # ---------------- Train ----------------
        for epoch in range(self.epochs):
            self.train_module.train() if hasattr(self.train_module, "train") else None

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} • train")
            for idx, batch in enumerate(pbar):
                # 允许 batch 是张量或张量序列
                if isinstance(batch, (list, tuple)):
                    batch = [t.to(device=device, dtype=torch.float32) for t in batch]
                else:
                    batch = batch.to(device=device, dtype=torch.float32)

                self.train_module.train_step(idx, batch)  # <- 要求返回 dict

            # ---------------- Valid（同 epoch 内） ----------------
            with torch.no_grad():
                pbar_v = tqdm(self.valid_loader, desc=f"Epoch {epoch + 1}/{self.epochs} • valid")
                for idx, batch in enumerate(pbar_v):
                    if isinstance(batch, (list, tuple)):
                        batch = [t.to(device=device, dtype=torch.float32) for t in batch]
                    else:
                        batch = batch.to(device=device, dtype=torch.float32)

                    self.train_module.valid_step(idx, batch)

            for item in self.train_module.logs:
                item.commit_epoch()
                print(item.export())

        for item in self.train_module.logs:
            item.plot_epoch_means()
