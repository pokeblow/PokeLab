# base_train_module.py
from __future__ import annotations

import os
import glob
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path

import torch

from .utils import *  # keep your project utils
from .log import PokeLog
from .configure import PokeConfig


# -----------------------
# utils
# -----------------------
def _unwrap_model(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if hasattr(m, "module") else m


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _atomic_torch_save(obj: Any, path: str) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    tmp = f"{path}.tmp_{int(time.time() * 1000)}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _cleanup_epoch_ckpts_by_name(ckpt_dir: str, name: str, keep_last_k: int) -> None:
    """
    删除旧的 {name}_checkpoint_epoch_*.ckpt，只保留最近 keep_last_k 个
    """
    if keep_last_k <= 0:
        return
    pattern = os.path.join(ckpt_dir, f"{name}_checkpoint_epoch_*.ckpt")
    paths = sorted(glob.glob(pattern))
    if len(paths) <= keep_last_k:
        return
    for p in paths[: len(paths) - keep_last_k]:
        try:
            os.remove(p)
        except OSError:
            pass


def _is_nan(x: float) -> bool:
    return x != x


# -----------------------
# ParameterConfig
# -----------------------
@dataclass
class ParameterConfig:
    name: str
    model: torch.nn.Module
    optimizer: Optional[torch.optim.Optimizer] = None
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

    indicator: Optional[PokeLog] = None

    ckpt_dir: str = ""
    save_checkpoint: bool = True

    save_every_epochs: int = 1
    keep_last_k: int = 5
    best_mode: str = "min"

    # 可选：允许你自定义 best 的后缀（默认 .pth）
    best_ext: str = ".pth"


# -----------------------
# BaseTrainModule
# -----------------------
class PokeBaseModule:
    """
    ✅ 文件名统一定位为：
    - {name}_latest.ckpt：每次 save_parameters() 覆盖写（断点续训用，完整训练态）
    - {name}_checkpoint_epoch_XXXXXXXX.ckpt：按 epoch 周期写（完整训练态）
    - {name}_best.pth：当 indicator 判定为 best 时，只保存权重（state_dict）

    多网络：每个 item[name] 独立写文件，不互相覆盖。
    """

    def __init__(self, config: Optional[PokeConfig] = None):
        self.epoch: int = 0
        self.global_step: int = 0
        self.config = config

        self.logs = self._init_default_logs()
        self.parameters_config: List[ParameterConfig] = []

        # 是否在 checkpoint 里保存 RNG 状态（使断点更可复现）
        self.save_rng_state: bool = False

    def _init_default_logs(self) -> None:
        if not isinstance(self.configure_logs(), tuple):
            return (self.configure_logs(),)
        return self.configure_logs()

    def configure_parameters(self):
        raise NotImplementedError

    # -----------------------
    # register
    # -----------------------
    def set_parameters(
        self,
        name: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        indicator: Optional[PokeLog] = None,
        ckpt_dir: str = "",
        save_checkpoint: bool = True,
        save_every_epochs: int = 1,
        keep_last_k: int = 5,
        best_mode: str = "min",
        best_ext: str = ".pth",
    ) -> None:
        self.parameters_config.append(
            ParameterConfig(
                name=name,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                indicator=indicator,
                ckpt_dir=ckpt_dir,
                save_checkpoint=save_checkpoint,
                save_every_epochs=int(save_every_epochs),
                keep_last_k=int(keep_last_k),
                best_mode=best_mode,
                best_ext=best_ext,
            )
        )

    # -----------------------
    # internal: best 判定
    # -----------------------
    def _is_best_now(self, item: ParameterConfig) -> bool:
        ind = item.indicator
        if ind is None:
            return False

        cur = getattr(ind, "current_epoch_means", None)
        history = getattr(ind, "epoch_means", None)
        if cur is None or history is None:
            return False
        if isinstance(cur, float) and _is_nan(cur):
            return False

        valid_history = [v for v in history if not (isinstance(v, float) and _is_nan(v))]
        if not valid_history:
            return False

        mode = (item.best_mode or "min").lower()
        if mode == "max":
            return cur == max(valid_history)
        return cur == min(valid_history)

    # -----------------------
    # internal: per-item ckpt paths (prefix with name)
    # -----------------------
    def _ckpt_paths(self, item: ParameterConfig) -> Dict[str, str]:
        if not item.ckpt_dir:
            return {"latest": "", "checkpoint": "", "best": ""}

        _ensure_dir(item.ckpt_dir)

        # ✅ name_latest, name_best, name_checkpoint...
        latest = os.path.join(item.ckpt_dir, f"{item.name}_latest.ckpt")
        checkpoint = os.path.join(item.ckpt_dir, f"{item.name}_checkpoint_epoch_{self.epoch:04d}.ckpt")
        best = os.path.join(item.ckpt_dir, f"{item.name}_best{item.best_ext}")

        return {"latest": latest, "checkpoint": checkpoint, "best": best}

    # -----------------------
    # internal: pack ckpt (single item)
    # -----------------------
    def _pack_checkpoint_one(self, item: ParameterConfig) -> Dict[str, Any]:
        m = _unwrap_model(item.model)

        ckpt: Dict[str, Any] = {
            "meta": {
                "epoch": int(self.epoch),
                "global_step": int(self.global_step),
                "name": item.name,
            },
            "item": {
                "model": m.state_dict(),
                "optimizer": item.optimizer.state_dict() if item.optimizer is not None else None,
                "scheduler": item.scheduler.state_dict() if item.scheduler is not None else None,
                "indicator": item.indicator.state_dict() if item.indicator is not None else None,
            },
        }

        if self.save_rng_state:
            ckpt["rng_state"] = {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }

        return ckpt

    # -----------------------
    # public: save (per item)
    # -----------------------
    def save_latest(self) -> None:
        for item in self.parameters_config:
            if not item.save_checkpoint or not item.ckpt_dir:
                continue
            paths = self._ckpt_paths(item)
            ckpt = self._pack_checkpoint_one(item)
            _atomic_torch_save(ckpt, paths["latest"])

    def save_epoch_checkpoint(self) -> None:
        for item in self.parameters_config:
            if not item.save_checkpoint or not item.ckpt_dir:
                continue

            save_every = int(item.save_every_epochs or 0)
            if save_every <= 0:
                continue
            if (self.epoch % save_every) != 0:
                continue

            paths = self._ckpt_paths(item)
            ckpt = self._pack_checkpoint_one(item)
            _atomic_torch_save(ckpt, paths["checkpoint"])

            _cleanup_epoch_ckpts_by_name(item.ckpt_dir, item.name, int(item.keep_last_k or 0))

    def save_best_weights(self) -> None:
        for item in self.parameters_config:
            if not item.save_checkpoint or not item.ckpt_dir:
                continue
            if not self._is_best_now(item):
                continue

            paths = self._ckpt_paths(item)
            m = _unwrap_model(item.model)

            # best 只存权重（state_dict）
            _atomic_torch_save(m.state_dict(), paths["best"])

    def save_parameters(self) -> None:
        """
        建议：仅在 epoch end（log.commit_epoch 之后）调用
        """
        self.save_latest()
        self.save_epoch_checkpoint()
        self.save_best_weights()

    # -----------------------
    # public: load latest
    # -----------------------
    def load_latest(
        self,
        ckpt_path: str,
        map_location: str | torch.device = "cpu",
        strict_model: bool = True,
        continue_epoch: bool = True,
    ) -> Dict[str, Any]:
        """
        ✅ 兼容两种 checkpoint：
        - 新版 per-item：{"meta":..., "item": {...}}
        - 旧版 all-in-one：{"meta":..., "items": {...}}
        """
        if not ckpt_path or (not os.path.isfile(ckpt_path)):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=map_location)

        meta = ckpt.get("meta", {})
        if not continue_epoch:
            self.epoch = 0
            self.global_step = 0
        else:
            self.epoch = int(meta.get("epoch", 0))
            self.global_step = int(meta.get("global_step", 0))

        # RNG（可选）
        if self.save_rng_state and "rng_state" in ckpt:
            rs = ckpt["rng_state"]
            if rs.get("torch") is not None:
                torch.set_rng_state(rs["torch"])
            if torch.cuda.is_available() and rs.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rs["cuda"])

        # ✅ per-item
        if "item" in ckpt:
            one = ckpt["item"]
            name_in_ckpt = meta.get("name", None)

            for item in self.parameters_config:
                if not item.save_checkpoint:
                    continue
                if name_in_ckpt is not None and item.name != name_in_ckpt:
                    continue

                m = _unwrap_model(item.model)
                if one.get("model") is not None:
                    m.load_state_dict(one["model"], strict=strict_model)

                if item.optimizer is not None and one.get("optimizer") is not None:
                    item.optimizer.load_state_dict(one["optimizer"])

                if item.scheduler is not None and one.get("scheduler") is not None:
                    item.scheduler.load_state_dict(one["scheduler"])

                if item.indicator is not None and one.get("indicator") is not None:
                    item.indicator.load_state_dict(one["indicator"])

            return ckpt

        # ✅ old all-in-one
        items_state: Dict[str, Any] = ckpt.get("items", {})
        for item in self.parameters_config:
            if not item.save_checkpoint:
                continue

            one = items_state.get(item.name)
            if one is None:
                continue

            m = _unwrap_model(item.model)
            if one.get("model") is not None:
                m.load_state_dict(one["model"], strict=strict_model)

            if item.optimizer is not None and one.get("optimizer") is not None:
                item.optimizer.load_state_dict(one["optimizer"])

            if item.scheduler is not None and one.get("scheduler") is not None:
                item.scheduler.load_state_dict(one["scheduler"])

            if item.indicator is not None and one.get("indicator") is not None:
                item.indicator.load_state_dict(one["indicator"])

        return ckpt

    # --------- 需在子类实现 ----------
    def train_step(self, batch_idx: int, batch) -> Dict[str, float]:
        raise NotImplementedError

    def valid_step(self, batch_idx: int, batch) -> Dict[str, float]:
        raise NotImplementedError

    def configure_logs(self):
        return None

    def set_scheduler(self):
        pass

    def visualization(self):
        pass


# ✅ 兼容你项目里可能引用的名字
BaseTrainModule = PokeBaseModule


# -----------------------
# minimal test
# -----------------------
if __name__ == "__main__":
    import shutil

    # -----------------------
    # 0. checkpoint 目录
    # -----------------------
    ckpt_dir = Path("./_ckpt_test")
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    class MyTrainModule(BaseTrainModule):
        def __init__(self):
            super().__init__(config=None)

            self.model = torch.nn.Linear(4, 1)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

            self.loss_log = self.configure_logs()
            self.configure_parameters()

        def configure_logs(self) -> PokeLog:
            return PokeLog(log_name="loss", log_type="loss", log_location="train")

        def configure_parameters(self):
            self.set_parameters(
                name="net",
                model=self.model,
                optimizer=self.optimizer,
                indicator=self.loss_log,
                ckpt_dir=str(ckpt_dir),
                save_every_epochs=1,
                keep_last_k=3,
                best_ext=".pth",
            )

        def train_step(self, batch_idx: int, batch):
            raise NotImplementedError

        def valid_step(self, batch_idx: int, batch):
            raise NotImplementedError

    module = MyTrainModule()

    print("Start training simulation ...")
    for epoch in range(5):
        module.epoch = epoch

        for _ in range(4):
            x = torch.randn(8, 4)
            y = module.model(x).mean()
            module.loss_log.set_step(float(y.detach()))

            y.backward()
            module.optimizer.step()
            module.optimizer.zero_grad()

        module.loss_log.commit_epoch()
        module.save_parameters()

        print(
            f"Epoch {epoch} | "
            f"epoch_mean={module.loss_log.current_epoch_means:.4f} | "
            f"best={module.loss_log.best_epoch_means:.4f}"
        )

    print("\nSaved files:")
    for p in sorted(ckpt_dir.iterdir()):
        if p.is_file():
            print(" ", p.name)

    # ✅ 注意：latest 文件名现在是 {name}_latest.ckpt
    print("\nReload from net_latest.ckpt ...")
    module2 = MyTrainModule()
    module2.load_latest(str(ckpt_dir / "net_latest.ckpt"))

    print(
        f"Restored epoch={module2.epoch}, "
        f"current_epoch_mean={module2.loss_log.current_epoch_means:.4f}, "
        f"best={module2.loss_log.best_epoch_means:.4f}"
    )
