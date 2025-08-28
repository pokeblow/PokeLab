from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt


@dataclass
class PokeLog:
    """
    - 使用 set_step(value) 追加一步的指标
    - 使用 commit_epoch() 把当前 epoch 的所有 step 聚合并入历史
    - 可视化：
        * plot_epoch_means(): 画每个 epoch 的均值变化
        * plot_current_epoch(): 画当前 epoch 内的逐步变化
    """
    log_name: str = ""
    log_type: str = "" # include loss, lr, acc, iou, image
    log_location: str = ""

    # 历史：按 epoch 存储聚合结果
    epoch_means: List[float] = field(default_factory=list)
    epoch_stds:  List[float] = field(default_factory=list)
    epoch_mins:  List[float] = field(default_factory=list)
    epoch_maxs:  List[float] = field(default_factory=list)

    best_epoch_means: float = float('inf')
    current_epoch_means: float = float('inf')

    # 临时：当前 epoch 的逐步值
    _buffer: List[float] = field(default_factory=list)

    current_step: int = 0
    current_epoch: int = 0

    # ====== 记录相关 ======
    def set_step(self, value: float) -> None:
        """记录一个 step 的数值。"""
        self._buffer.append(float(value))
        self.current_step += 1

    def commit_epoch(self) -> None:
        """将当前 epoch 的 step 值聚合并入历史，并清空缓冲。"""
        if len(self._buffer) == 0:
            # 空 epoch 时也推进计数，但写入 NaN，避免误用
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

    # ====== 查询相关 ======
    def current_value(self) -> float:
        """返回当前 epoch 的均值（若为空返回 NaN）。"""
        if len(self._buffer) == 0:
            return float("nan")
        return float(np.mean(self._buffer))

    def last_epoch_summary(self, log=False) -> Optional[Dict[str, float]]:
        """返回最近一个 epoch 的统计信息。若不存在则返回 None。"""
        if len(self.epoch_means) == 0:
            return None
        i = -1

        if log:
            return f"{self.log_name} • {self.log_location} = {self.epoch_means[i]:.4f}, "

        else:
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
        """导出为字典，便于 JSON 或持久化。"""
        return {
            "log_name": self.log_name,
            "log_type": self.log_type,
            "epoch_means": self.epoch_means,
            "epoch_stds": self.epoch_stds,
            "epoch_mins": self.epoch_mins,
            "epoch_maxs": self.epoch_maxs,
        }

    # ====== 可视化 ======
    def plot_epoch_means(self) -> None:
        """
        画每个 epoch 的均值变化。
        注意：只画一张图，不设置特定颜色（符合通用规范）。
        """
        y = np.asarray(self.epoch_means, dtype=float)
        x = np.arange(1, len(y) + 1, dtype=int)

        plt.figure()
        plt.plot(x, y, marker='o')
        plt.title(f"{self.log_name or 'log'} (type={self.log_type}) - Epoch Means")
        plt.xlabel("Epoch")
        plt.ylabel("Mean")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ====== 魔术方法 ======
    def __str__(self) -> str:
        cur_mean = self.current_value()
        return (
            f"Log Name: {self.log_name}, Log Type: {self.log_type}\n"
            f"Current Epoch: {self.current_epoch}, Current Step: {self.current_step}\n"
            f"Current Buffer Mean: {cur_mean}"
            f'Current Epoch Mean: {self.current_epoch_means}'
            f'Best Epoch Mean: {self.best_epoch_means}'
        )


if __name__ == "__main__":
    # 简单示例
    loss_log = PokeLog(log_name="loss", log_type="train")

    # 模拟 epoch 0
    loss_log.set_step(1.0134)
    loss_log.set_step(1.045)
    print(loss_log)               # 查看当前缓冲均值
    loss_log.commit_epoch()       # 聚合到历史

    # 模拟 epoch 1
    loss_log_2 = PokeLog(log_name="loss", log_type="train")
    for epoch in range(10):
        for v in [0.98, 0.95, 0.93, 0.92]:
            loss_log_2.set_step(v)
        loss_log_2.commit_epoch()
        print(loss_log_2.last_epoch_summary())  # 查看上一个 epoch 的统计
    print(loss_log_2.export())

    # 可视化
    loss_log_2.plot_epoch_means()