# PokeLab

A lightweight PyTorch training and experiment pipeline with:
- modular training (`Poke.pipeline`)
- checkpoint management (`latest/epoch/best`)
- log aggregation + visualization
- simple prediction/evaluation demo

## 1. Project Layout

```text
PokeLab/
├── Poke/                  # core framework
│   ├── pipeline/          # trainer / module / config / dataloader / logs
│   └── evaluation/        # prediction & result saving tools
├── Method/                # training demo (model + dataset + config)
│   ├── train.py
│   └── config/demo.yaml
├── Experiment/            # prediction demo
│   └── exp_demo.py
└── Data/                  # optional data folder
```

## 2. Environment

Recommended:
- Python 3.10+
- PyTorch 2.x

Install dependencies:

```bash
pip install torch torchvision pyyaml numpy matplotlib tqdm pillow tabulate
```

## 3. Quick Start (Training Demo)

Run the demo trainer:

```bash
cd Method
python train.py
```

What this does:
- loads config from `Method/config/demo.yaml`
- builds random classification dataset (`RandomClsDataset`)
- trains via `PokeTrainer`
- writes checkpoints to `Method/parameters/<version>/`
- writes logs to `Method/logs/<run_id>/summary.log`

## 4. Configuration

`PokeConfig` requires three top-level keys:
- `dataloader`
- `run`
- `model`

Example (`Method/config/demo.yaml`):

```yaml
version: "demo"

dataloader:
  num_workers: 0

run:
  batch_size: 50
  epochs: 50
  seed: 42

model:
  lr: 0.001
  in_channels: 20
  out_channels: 3
  parameter_dir: "/abs/path/to/parameters"
```

Note:
- `parameter_dir` in the demo is an absolute path. Update it for your machine.

## 5. Checkpoint Rules

For each registered parameter group (`set_parameters(name=...)`), files are saved as:

- `{name}_latest.ckpt`
- `{name}_checkpoint_epoch_XXXX.ckpt`
- `{name}_best.pth`

Behavior:
- `latest`: overwritten every epoch end
- `checkpoint_epoch`: saved by `save_every_epochs`, old files cleaned by `keep_last_k`
- `best`: saved when metric reaches best value (`best_mode=min/max`)

Resume training:

```python
module.load_latest("/your/path/net_latest.ckpt", continue_epoch=True)
```

## 6. Logging and Curves

Training writes one summary line per epoch, e.g.:

```text
[2026-03-10 17:50:12] Epoch 1/1: train: loss • train = 1.2632 | valid: loss • valid = 1.2927
```

Generate metric plots from a `summary.log`:

```python
from Poke.pipeline.log_viz import plot_training_log

plot_training_log(log_path="/abs/path/to/summary.log", show=False)
```

If `log_path` is omitted, it uses the latest path stored in `Poke/pipeline/buffer.tmp`.

## 7. Prediction Demo

Run prediction + save outputs:

```bash
cd Experiment
python exp_demo.py
```

Outputs:
- image results: `Experiment/results/image_*/`
- value results: `Experiment/results/result_values.txt`

## 8. Extend the Framework

### 8.1 Custom Train Module

Inherit `PokeBaseModule` and implement:
- `configure_logs()`
- `configure_parameters()`
- `train_step()`
- `valid_step()`

Minimal skeleton:

```python
from Poke.pipeline import PokeBaseModule, PokeLog

class MyModule(PokeBaseModule):
    def configure_logs(self):
        self.log_loss = PokeLog(log_name="loss", log_type="loss", log_location="all")
        return (self.log_loss,)

    def configure_parameters(self):
        self.set_parameters(
            name="net",
            model=self.net,
            optimizer=self.optimizer,
            indicator=self.log_loss.valid,
            ckpt_dir="/tmp/ckpts",
            best_mode="min",
        )

    def train_step(self, batch_idx, batch):
        ...

    def valid_step(self, batch_idx, batch):
        ...
```

### 8.2 Custom DataLoader

- Single-device: `PokeBaseDataloader`
- DDP: `PokeDDPDataloader` (with sampler + `set_epoch`)

Defaults:
- train: `shuffle=True`, `drop_last=True`
- valid/test: `shuffle=False`, `drop_last=False`

## 9. Known Notes

- `Method/train.py` and `Experiment/exp_demo.py` are designed to run from their own folders (`cd Method`, `cd Experiment`).
- Logs/checkpoints/results are frequently overwritten by demo scripts.
- This repo currently includes generated artifacts (`logs/`, `parameters/`, result images). Clean them before packaging/release if needed.
