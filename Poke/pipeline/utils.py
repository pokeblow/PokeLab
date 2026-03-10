import inspect
import platform
import random
import socket
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 保证确定性（会影响速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor_to_numpy(tensors: Any, batch_idx: int = 0):
    """
    支持:
    - 单个 Tensor -> np.ndarray
    - list/tuple[Tensor] -> list[np.ndarray]
    """
    if torch.is_tensor(tensors):
        return tensors.detach().cpu().numpy()[batch_idx]

    if isinstance(tensors, (list, tuple)):
        return [t.detach().cpu().numpy()[batch_idx] for t in tensors]

    raise TypeError(f"Unsupported type for tensor_to_numpy: {type(tensors)}")


def load_config(config_path: str) -> dict:
    path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format in config file: {path}") from e
    except OSError as e:
        raise RuntimeError(f"Failed to open config file: {path}") from e


def get_run_id():
    time_str = datetime.now().strftime("%Y%m%d_%H")
    host = socket.gethostname()

    cpu = platform.processor()
    if not cpu:
        cpu = platform.machine()

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        gpu = gpu.replace(" ", "").replace("/", "")
    else:
        gpu = "cpu"

    run_id = f"{time_str}_{host}_{cpu}_{gpu}"
    return run_id


def get_caller_file():
    frame = inspect.stack()[2]
    caller_path = Path(frame.filename).resolve()
    return caller_path
