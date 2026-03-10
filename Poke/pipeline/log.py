# log.py
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class PokeLog:
    """
    统一接口：外层永远用 log.<location>.set_step(...)
    - 若 log_location == "all"：容器包含 train/val/test（或自定义 group_locations）
    - 若 log_location == "train"/"val"/"test"：容器只包含一个同名子日志
    - 禁止：log.set_step(...)（会报错）
    """

    log_name: str = ""
    log_type: str = ""        # e.g., loss, lr, acc, iou, image
    log_location: str = ""    # "all" or single location like "train"

    # group 支持：当 log_location == "all" 时生成这些 child
    group_locations: List[str] = field(default_factory=lambda: ["train", "valid", "test"])

    # ---- internal ----
    _leaf: bool = field(default=False, repr=False)  # True 表示这是子节点（真正记录数据的 leaf）
    _children: Dict[str, "PokeLog"] = field(default_factory=dict, init=False, repr=False)

    # ====== Leaf 数据（只有 leaf 真的用）======
    epoch_means: List[float] = field(default_factory=list)
    epoch_stds:  List[float] = field(default_factory=list)
    epoch_mins:  List[float] = field(default_factory=list)
    epoch_maxs:  List[float] = field(default_factory=list)

    best_epoch_means: float = float("inf")
    current_epoch_means: float = float("inf")

    _buffer: List[Any] = field(default_factory=list)

    current_step: int = 0
    current_epoch: int = 0

    # ---------------------------
    # init
    # ---------------------------
    def __post_init__(self) -> None:
        # leaf：不创建 children（leaf 才能 set_step）
        if self._leaf:
            return

        # container：创建 children
        if self.log_location == "all":
            locations = list(self.group_locations)
        else:
            # 单 location 也包装成 container，但只创建一个同名 child
            locations = [self.log_location]

        self._children = {
            loc: PokeLog(
                log_name=self.log_name,
                log_type=self.log_type,
                log_location=loc,
                group_locations=self.group_locations,
                _leaf=True,   # ⭐ 子节点是 leaf
            )
            for loc in locations
        }

    # ---------------------------
    # container/leaf 判定
    # ---------------------------
    @property
    def is_leaf(self) -> bool:
        return self._leaf

    @property
    def is_container(self) -> bool:
        return not self._leaf

    @property
    def child_names(self) -> List[str]:
        return list(self._children.keys())

    # ---------------------------
    # container 访问：log.train / log["train"]
    # ---------------------------
    def __getattr__(self, name: str) -> Any:
        if self.is_container and name in self._children:
            return self._children[name]
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")

    def __getitem__(self, key: str) -> "PokeLog":
        if not self.is_container:
            raise TypeError("Leaf log is not subscriptable.")
        return self._children[key]

    # =========================
    # 记录相关
    # =========================
    def set_step(self, value: Any) -> None:
        """只允许 leaf 调用；外层必须用 log.train.set_step(...)"""
        if self.is_container:
            raise RuntimeError(
                "Direct call to set_step() is not allowed on a container log. "
                "Use log.<location>.set_step(...), e.g., log.train.set_step(...)."
            )

        if self.log_type != "image":
            self._buffer.append(float(value))
        else:
            self._buffer.append(value)
        self.current_step += 1

    def get_buffer_image(self) -> List[Any]:
        if self.is_container:
            raise RuntimeError("Container log has no buffer. Use a child log, e.g., log.train.get_buffer_image().")
        return self._buffer

    def commit_epoch(self) -> None:
        """
        - container：对子日志逐个 commit（推荐每个 epoch 结束时调用一次 container.commit_epoch()）
        - leaf：聚合并清空 buffer
        """
        if self.is_container:
            for child in self._children.values():
                child.commit_epoch()
            # container 自身推进 epoch 计数（便于对齐）
            self.current_epoch += 1
            self.current_step = 0
            return

        # ---- leaf 聚合 ----
        if self.log_type != "image":
            if len(self._buffer) == 0:
                self.current_epoch_means = float("nan")
                self.epoch_means.append(np.nan)
                self.epoch_stds.append(np.nan)
                self.epoch_mins.append(np.nan)
                self.epoch_maxs.append(np.nan)
            else:
                arr = np.asarray(self._buffer, dtype=float)
                self.current_epoch_means = float(np.mean(arr))
                if self.current_epoch_means < self.best_epoch_means:
                    self.best_epoch_means = self.current_epoch_means

                self.epoch_means.append(self.current_epoch_means)
                self.epoch_stds.append(float(np.std(arr, ddof=0)))
                self.epoch_mins.append(float(np.min(arr)))
                self.epoch_maxs.append(float(np.max(arr)))

        self.current_epoch += 1
        self.current_step = 0
        self._buffer.clear()

    # =========================
    # 查询相关
    # =========================
    def current_value(self) -> float:
        """leaf：返回当前 buffer 均值；container：NaN"""
        if self.is_container or self.log_type == "image" or len(self._buffer) == 0:
            return float("nan")
        return float(np.mean(np.asarray(self._buffer, dtype=float)))

    def last_epoch_summary(self, log: bool = False) -> Optional[Dict[str, Any]]:
        """仅 leaf 有意义"""
        if self.is_container:
            raise RuntimeError("Container log has no single epoch summary. Use a child log, e.g., log.train.last_epoch_summary().")

        if len(self.epoch_means) == 0:
            return None
        i = -1

        if log:
            return f"{self.log_name} • {self.log_location} = {self.epoch_means[i]:.4f}, "

        return dict(
            log_name=self.log_name,
            log_type=self.log_type,
            log_location=self.log_location,
            mean=self.epoch_means[i],
            std=self.epoch_stds[i],
            min=self.epoch_mins[i],
            max=self.epoch_maxs[i],
        )

    def export(self) -> Dict[str, Any]:
        """导出为 dict（container 会递归 children）"""
        if self.is_container:
            return {
                "log_name": self.log_name,
                "log_type": self.log_type,
                "log_location": self.log_location,
                "children": {k: v.export() for k, v in self._children.items()},
            }

        return {
            "log_name": self.log_name,
            "log_type": self.log_type,
            "log_location": self.log_location,
            "epoch_means": list(self.epoch_means),
            "epoch_stds": list(self.epoch_stds),
            "epoch_mins": list(self.epoch_mins),
            "epoch_maxs": list(self.epoch_maxs),
        }

    # =========================
    # Checkpoint support
    # =========================
    def state_dict(self) -> Dict[str, Any]:
        if self.is_container:
            return {
                "log_name": self.log_name,
                "log_type": self.log_type,
                "log_location": self.log_location,
                "group_locations": list(self.group_locations),
                "current_epoch": int(self.current_epoch),
                "children": {k: v.state_dict() for k, v in self._children.items()},
            }

        return {
            "log_name": self.log_name,
            "log_type": self.log_type,
            "log_location": self.log_location,

            "epoch_means": list(self.epoch_means),
            "epoch_stds":  list(self.epoch_stds),
            "epoch_mins":  list(self.epoch_mins),
            "epoch_maxs":  list(self.epoch_maxs),

            "best_epoch_means": float(self.best_epoch_means),
            "current_epoch_means": float(self.current_epoch_means),
            "current_step": int(self.current_step),
            "current_epoch": int(self.current_epoch),
        }

    def load_state_dict(self, state: Dict[str, Any], strict: bool = False) -> None:
        # 判断目标是 container 还是 leaf（以 children 字段为准）
        if "children" in state:
            # container restore
            self.log_name = state.get("log_name", self.log_name)
            self.log_type = state.get("log_type", self.log_type)
            self.log_location = state.get("log_location", self.log_location)
            self.group_locations = state.get("group_locations", self.group_locations)
            self.current_epoch = int(state.get("current_epoch", 0))

            # 重建 children
            if self.log_location == "all":
                locations = list(self.group_locations)
            else:
                locations = [self.log_location]

            self._leaf = False
            self._children = {
                loc: PokeLog(
                    log_name=self.log_name,
                    log_type=self.log_type,
                    log_location=loc,
                    group_locations=self.group_locations,
                    _leaf=True,
                )
                for loc in locations
            }

            children_state = state.get("children", {})
            for k, v in children_state.items():
                if k in self._children:
                    self._children[k].load_state_dict(v, strict=strict)
                elif strict:
                    raise KeyError(f"Unknown child log '{k}' in state_dict")
            return

        # leaf restore
        self._leaf = True
        for k, v in state.items():
            if hasattr(self, k):
                setattr(self, k, v)
            elif strict:
                raise KeyError(f"PokeLog has no attribute '{k}'")
        self._buffer.clear()

    # =========================
    # 可视化（leaf）
    # =========================
    def plot_epoch_means(self) -> None:
        if self.is_container:
            raise RuntimeError("Container log cannot plot directly. Use a child log, e.g., log.train.plot_epoch_means().")

        y = np.asarray(self.epoch_means, dtype=float)
        x = np.arange(1, len(y) + 1, dtype=int)

        plt.figure()
        plt.plot(x, y, marker="o")
        plt.title(f"{self.log_name or 'log'} (type={self.log_type}) - {self.log_location} Epoch Means")
        plt.xlabel("Epoch")
        plt.ylabel("Mean")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # =========================
    # 魔术方法
    # =========================
    def __str__(self) -> str:
        if self.is_container:
            return (
                f"[ContainerLog] {self.log_name} (type={self.log_type}) "
                f"log_location={self.log_location} children={self.child_names} "
                f"current_epoch={self.current_epoch}"
            )

        cur_mean = self.current_value()
        return (
            f"[LeafLog] {self.log_name} (type={self.log_type}) location={self.log_location}\n"
            f"Current Epoch: {self.current_epoch}, Current Step: {self.current_step}\n"
            f"Current Buffer Mean: {cur_mean}\n"
            f"Current Epoch Mean: {self.current_epoch_means}\n"
            f"Best Epoch Mean: {self.best_epoch_means}"
        )


if __name__ == "__main__":
    # =========================
    # ✅ 你要求的行为：单 location 也必须用 .train
    # =========================
    loss_log = PokeLog(log_name="loss", log_type="loss", log_location="train")
    loss_log.train.set_step(1.0134)  # ✅ OK
    # loss_log.set_step(1.0134)      # ❌ 报错（符合你的要求）
    loss_log.commit_epoch()
    print(loss_log.train.last_epoch_summary())

    # =========================
    # ✅ all：多 location
    # =========================
    loss_all = PokeLog(log_name="loss", log_type="loss", log_location="all")
    loss_all.train.set_step(1.0)
    loss_all.val.set_step(0.8)
    loss_all.commit_epoch()
    print(loss_all.train.last_epoch_summary())
    print(loss_all.val.last_epoch_summary())

    # checkpoint round-trip
    sd = loss_all.state_dict()
    restored = PokeLog()
    restored.load_state_dict(sd)
    print(restored)
    print(restored.train.last_epoch_summary())
