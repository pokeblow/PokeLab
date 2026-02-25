# base_train_module.py
from __future__ import annotations

import os
import glob
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from .utils import *

import torch
from .log import PokeLog
from .configure import PokeConfig


# -----------------------
# utils
# -----------------------
def _unwrap_model(m: torch.nn.Module) -> torch.nn.Module:
    # DataParallel / DDP
    return m.module if hasattr(m, "module") else m


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _atomic_torch_save(obj: Any, path: str) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    tmp = f"{path}.tmp_{int(time.time() * 1000)}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _cleanup_epoch_ckpts(ckpt_dir: str, keep_last_k: int) -> None:
    """删除旧的 epoch_*.ckpt，只保留最近 keep_last_k 个"""
    if keep_last_k <= 0:
        return
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.ckpt")))
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

    # ✅ 明确就是 PokeLog 实例（或 None）
    indicator: Optional[PokeLog] = None

    best_path: str = ""
    ckpt_dir: str = ""
    save_checkpoint: bool = True

    save_every_epochs: int = 1  # 1=每个epoch都存；0=不存 epoch_*.ckpt
    keep_last_k: int = 5

    best_mode: str = "min"  # 当前实现里仍采用 “current == best” 判定


# -----------------------
# BaseTrainModule
# -----------------------
class PokeBaseModule:
    """
    epoch checkpoint 方案：
    - latest.ckpt：每次 save_parameters() 覆盖写（断点续训）
    - epoch_XXXXXXXX.ckpt：按 epoch 周期写（完整训练态）
    - best.pth：当 indicator 判定为 best 时，只保存权重

    多网络：items[name] 做隔离，不互相覆盖。
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
        else:
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
        indicator: Optional[Log] = None,   # ✅ 修正：默认 None
        best_path: str = "",
        ckpt_dir: str = "",
        save_checkpoint: bool = True,
        save_every_epochs: int = 1,
        keep_last_k: int = 5,
        best_mode: str = "min",
    ) -> None:
        self.parameters_config.append(
            ParameterConfig(
                name=name,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                indicator=indicator,
                best_path=best_path,
                ckpt_dir=ckpt_dir,
                save_checkpoint=save_checkpoint,
                save_every_epochs=int(save_every_epochs),
                keep_last_k=int(keep_last_k),
                best_mode=best_mode,
            )
        )

    # -----------------------
    # internal: best 判定
    # -----------------------
    def _is_best_now(self, item: ParameterConfig) -> bool:
        ind = item.indicator
        if ind is None:
            return False

        # NaN 防护（空 epoch 会产生 NaN）
        cur = getattr(ind, "current_epoch_means", None)
        best = getattr(ind, "best_epoch_means", None)
        if cur is None or best is None:
            return False
        if isinstance(cur, float) and _is_nan(cur):
            return False
        if isinstance(best, float) and _is_nan(best):
            return False

        return cur == best

    # -----------------------
    # internal: pack ckpt (multi-items)
    # -----------------------
    def _pack_checkpoint_all(self) -> Dict[str, Any]:
        items: Dict[str, Any] = {}
        for item in self.parameters_config:
            if not item.save_checkpoint:
                continue

            m = _unwrap_model(item.model)
            items[item.name] = {
                "model": m.state_dict(),
                "optimizer": item.optimizer.state_dict() if item.optimizer is not None else None,
                "scheduler": item.scheduler.state_dict() if item.scheduler is not None else None,
                "indicator": item.indicator.state_dict() if item.indicator is not None else None,
            }

        ckpt: Dict[str, Any] = {
            "meta": {
                "epoch": int(self.epoch),
                "global_step": int(self.global_step),
            },
            "items": items,
        }

        if self.save_rng_state:
            ckpt["rng_state"] = {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }

        return ckpt

    def _get_ckpt_dir(self) -> str:
        for item in self.parameters_config:
            folder_path = Path(item.ckpt_dir)
            folder_path.mkdir(parents=True, exist_ok=True)
            if item.save_checkpoint and item.ckpt_dir:
                return item.ckpt_dir
        return ""

    def _get_save_every_epochs(self) -> int:
        vals = [
            int(it.save_every_epochs)
            for it in self.parameters_config
            if it.save_checkpoint and it.save_every_epochs and it.save_every_epochs > 0
        ]
        return min(vals) if vals else 0

    def _get_keep_last_k(self) -> int:
        vals = [int(it.keep_last_k) for it in self.parameters_config if it.save_checkpoint and it.ckpt_dir]
        return min(vals) if vals else 0

    # -----------------------
    # public: save
    # -----------------------
    def save_latest(self) -> None:
        ckpt_dir = self._get_ckpt_dir()
        if not ckpt_dir:
            return
        _ensure_dir(ckpt_dir)
        ckpt = self._pack_checkpoint_all()
        _atomic_torch_save(ckpt, os.path.join(ckpt_dir, "latest.ckpt"))

    def save_epoch_checkpoint(self) -> None:
        ckpt_dir = self._get_ckpt_dir()
        if not ckpt_dir:
            return

        save_every = self._get_save_every_epochs()
        if not save_every:
            return

        if (self.epoch % save_every) != 0:
            return

        _ensure_dir(ckpt_dir)
        ckpt = self._pack_checkpoint_all()
        epoch_path = os.path.join(ckpt_dir, f"epoch_{self.epoch:08d}.ckpt")
        _atomic_torch_save(ckpt, epoch_path)
        _cleanup_epoch_ckpts(ckpt_dir, self._get_keep_last_k())

    def save_best_weights(self) -> None:
        for item in self.parameters_config:
            if not item.save_checkpoint:
                continue
            if item.best_path and self._is_best_now(item):
                m = _unwrap_model(item.model)
                print(f'{item.ckpt_dir}/{item.best_path}')
                _atomic_torch_save(m.state_dict(),  f'{item.ckpt_dir}/{item.best_path}')

    def save_parameters(self) -> None:
        """
        ✅ 建议：仅在 epoch end（log.commit_epoch 之后）调用它
        """
        self.save_latest()
        self.save_epoch_checkpoint()
        self.save_best_weights()

    # -----------------------
    # public: load latest
    # -----------------------
    def load_latest(self,
                    ckpt_path: str,
                    map_location: str | torch.device = "cpu",
                    strict_model: bool = True,
                    continue_epoch=True) -> Dict[str, Any]:
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

if __name__ == "__main__":
    import shutil
    from pathlib import Path
    import torch
    import yaml

    # -----------------------
    # 0. checkpoint 目录
    # -----------------------
    ckpt_dir = Path("/Users/wanghaolin/GitHub/PokeLab/Method/logs/_ckpt_test")
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # 1. 生成一个最小 yaml，避免 BaseTrainModule 强制要求 config_path
    #    （如果你已经允许 config_path=""，这里可以删掉）
    # -----------------------
    tmp_yaml = ckpt_dir / "tmp_config.yaml"
    tmp_yaml.write_text(yaml.safe_dump({"dummy": True}), encoding="utf-8")

    # -----------------------
    # 2. 实现 MyTrainModule（继承 BaseTrainModule）
    # -----------------------
    class MyTrainModule(BaseTrainModule):
        def __init__(self, config_path: str):
            super().__init__(config_path=config_path)

            # 你自己的模型/优化器
            self.model = torch.nn.Linear(4, 1)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

            # logs / params
            self.loss_log = self.configure_logs()
            self.configure_parameters()

        def configure_logs(self) -> PokeLog:
            # 这里“创建并返回”，同时也会赋给 self.loss_log
            return PokeLog(log_name="loss", log_type="loss", log_location="train")

        def configure_parameters(self):
            # 这里注册 checkpoint 需要保存的对象
            self.set_parameters(
                name="net",
                model=self.model,
                optimizer=self.optimizer,
                indicator=self.loss_log,
                ckpt_dir=str(ckpt_dir),
                best_path=str(ckpt_dir / "best.pth"),
                save_every_epochs=1,
                keep_last_k=3,
            )

        # 下面两个函数只是为了符合抽象接口；本测试不会用到
        def train_step(self, batch_idx: int, batch):
            raise NotImplementedError

        def valid_step(self, batch_idx: int, batch):
            raise NotImplementedError

    # -----------------------
    # 3. 训练模拟 + 保存
    # -----------------------
    module = MyTrainModule(config_path=str(tmp_yaml))

    print("Start training simulation with MyTrainModule + log_2.PokeLog ...")
    for epoch in range(5):
        module.epoch = epoch

        # fake train steps
        for _ in range(4):
            x = torch.randn(8, 4)
            y = module.model(x).mean()

            # 用 PokeLog 记录 step
            module.loss_log.set_step(float(y.detach()))

            y.backward()
            module.optimizer.step()
            module.optimizer.zero_grad()

        # epoch end：聚合并更新 best/current
        module.loss_log.commit_epoch()

        # epoch end 保存（latest + epoch_xxx + best）
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

    # -----------------------
    # 4. 测试断点恢复
    # -----------------------
    print("\nReload from latest.ckpt ...")
    module2 = MyTrainModule(config_path=str(tmp_yaml))

    module2.load_latest(str(ckpt_dir / "latest.ckpt"))

    print(
        f"Restored epoch={module2.epoch}, "
        f"current_epoch_mean={module2.loss_log.current_epoch_means:.4f}, "
        f"best={module2.loss_log.best_epoch_means:.4f}"
    )
