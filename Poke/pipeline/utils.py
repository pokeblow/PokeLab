import random
import numpy as np
import torch

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