import random
import numpy as np
import torch
from pathlib import Path
import yaml
from datetime import datetime
import platform
import socket
import torch
import inspect

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 保证确定性（会影响速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor_to_numpy(tensor: torch.Tensor, batch_idx=0) -> np.ndarray:
    return tensor.detach().cpu().numpy()[batch_idx]

def tensor_to_numpy(tensor_list, batch_idx=0):
    numpy_list = []
    for tensor in tensor_list:
        numpy_list.append(tensor.detach().cpu().numpy()[batch_idx])
    return numpy_list


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
    # ===== 时间（到小时）=====
    time_str = datetime.now().strftime("%Y%m%d_%H")

    # ===== 主机名 =====
    host = socket.gethostname()

    # ===== CPU 信息 =====
    cpu = platform.processor()
    if not cpu:
        cpu = platform.machine()

    # ===== GPU 信息 =====
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        gpu = gpu.replace(" ", "").replace("/", "")
    else:
        gpu = "cpu"

    # ===== 拼接 =====
    run_id = f"{time_str}_{host}_{cpu}_{gpu}"

    return run_id


def get_caller_file():
    # 当前帧 -> 上一层 -> 调用者
    frame = inspect.stack()[2]
    caller_path = Path(frame.filename).resolve()
    return caller_path
