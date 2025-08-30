# -*- coding: utf-8 -*-
"""
集中管理全局变量 log_root（模块级“单例”）。
- 通过 set_log_root() 设置/修改
- 通过 get_log_root() 读取
- 通过 on_change(callback) 注册变更回调（可选）
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional, List

# 模块级存储（真正的“全局”）
_log_root: Optional[Path] = None
_callbacks: List[Callable[[Optional[Path]], None]] = []


def set_log_root(path: Optional[str | Path], *, create: bool = True) -> Optional[Path]:
    """
    设置/修改 log_root。
    - path=None 可用于清空
    - create=True 时若目录不存在会自动创建
    返回最终的 Path 或 None
    """
    global _log_root

    new_val: Optional[Path]
    if path is None:
        new_val = None
    else:
        p = Path(path).expanduser().resolve()
        if create:
            p.mkdir(parents=True, exist_ok=True)
        new_val = p

    _log_root = new_val
    # 通知订阅者
    for cb in list(_callbacks):
        try:
            cb(_log_root)
        except Exception:
            # 回调异常不影响主流程
            pass

    print(_log_root)


def get_log_root(default: Optional[str | Path] = None) -> Optional[Path]:
    """
    读取 log_root；若尚未设置且提供了 default，则返回 default（不落盘、不缓存）。
    """
    if _log_root is not None:
        return _log_root
    if default is None:
        return None
    return Path(default).expanduser().resolve()


def on_change(callback: Callable[[Optional[Path]], None]) -> None:
    """
    （可选）注册一个在 log_root 变化时触发的回调。
    """
    _callbacks.append(callback)
