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

        for key in ["dataloader", "run", "model"]:
            if key not in self._data:
                raise KeyError(f"Missing required key: '{key}'")

    def get(self, key: str, default: Any = None) -> Any:
        value = self._data
        for k in key.split("."):
            if not isinstance(value, dict) or k not in value:
                return default
            value = value[k]
        return value

    def __getitem__(self, key: str):
        val = self.get(key)
        if val is None:
            raise KeyError(key)
        return val

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
    cfg = PokeConfig("/Users/wanghaolin/GitHub/PokeLab/Method/config/demo.yaml")

    version = cfg.get("version")
    epochs = cfg.run.get("epochs")
    device = cfg.run.get("device")
    train_bs = cfg.run.get("batch_size")
    lr = cfg.model.get("lr")

    print(version, epochs, device, train_bs, lr)
