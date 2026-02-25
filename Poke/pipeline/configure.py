from pathlib import Path
from typing import Any
import yaml


class PokeConfig:
    def __init__(self, path: str):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f) or {}

        # 强制要求三大模块
        for key in ["dataloader", "run", "model"]:
            if key not in self._data:
                raise KeyError(f"Missing required key: '{key}'")

    # ===== 核心：点式访问 =====
    def get(self, key: str, default: Any = None) -> Any:
        value = self._data
        for k in key.split("."):
            if not isinstance(value, dict) or k not in value:
                return default
            value = value[k]
        return value

    # ===== 方便写法 =====
    def __getitem__(self, key: str):
        val = self.get(key)
        if val is None:
            raise KeyError(key)
        return val

    # ===== 三大模块快捷入口 =====
    @property
    def dataloader(self):
        return self._data["dataloader"]

    @property
    def run(self):
        return self._data["run"]

    @property
    def model(self):
        return self._data["model"]

if __name__ == "__main__":
    cfg = Config("/Users/wanghaolin/GitHub/PokeLab/Method/config/demo.yaml")

    version = cfg.get("version")
    # ===== run =====

    epochs = cfg.run.get("epochs")
    device = cfg.run.get("device")

    # ===== dataloader =====
    train_bs = cfg.dataloader.get("train.batch_size")

    # ===== model =====
    lr = cfg.model.get("optimizer.lr")

    # ===== 全局访问（可选）
    lr2 = cfg.get("model.optimizer.lr")