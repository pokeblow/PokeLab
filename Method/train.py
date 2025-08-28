# test_poke_train.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from Library.poke.poke_train import BaseTrainModule, PokeTrainer
from Library.poke.poke_log import PokeLog
import random

# ---- 一个极简的随机分类数据集（3类，20维特征） ----
class RandomClsDataset(Dataset):
    def __init__(self, n=1000, in_dim=20, n_classes=3, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, in_dim, generator=g)
        self.y = torch.randint(low=0, high=n_classes, size=(n,), generator=g)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ---- 继承与重写 BaseTrainModule ----
class SimpleClassifierModule(BaseTrainModule):
    def __init__(self, config_path: str = ""):
        super().__init__(config_path=config_path)
        # 超参（可从 config 中覆盖）
        in_dim = self.config.get("model", {}).get("in_dim", 20)
        n_classes = self.config.get("model", {}).get("n_classes", 3)
        hidden = self.config.get("model", {}).get("hidden", 64)
        lr = self.config.get("optim", {}).get("lr", 1e-3)

        # 一个很简单的两层感知机
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes)
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)

    # 可选：显式注册要记录的日志（也可以不重写，由返回的 metrics 自动创建）
    def configure_logs(self):
        self.loss_log = PokeLog(log_name="loss", log_type="loss", log_location='train')
        self.loss_log_val = PokeLog(log_name="loss_val", log_type="loss", log_location='train')
        self.acc_log = PokeLog(log_name="acc", log_type="acc", log_location='valid')
        self.acc_log_val = PokeLog(log_name="acc_val", log_type="acc", log_location='valid')
        return self.loss_log, self.loss_log_val, self.acc_log, self.acc_log_val

    # 必须：实现单步训练逻辑，返回一个 dict 的指标
    def train_step(self, batch_idx, batch):
        self.net.train()
        x, y = batch
        x = x.to(self.device, dtype=torch.float32)
        y = y.to(self.device, dtype=torch.long)

        logits = self.net(x)
        loss = self.criterion(logits, y)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean().item()
        self.loss_log.set_step(loss.item())
        self.acc_log.set_step(acc)


    # 必须：实现单步验证逻辑，返回一个 dict 的指标
    @torch.no_grad()
    def valid_step(self, batch_idx, batch):
        self.net.eval()
        x, y = batch
        x = x.to(self.device, dtype=torch.float32)
        y = y.to(self.device, dtype=torch.long)

        logits = self.net(x)
        loss = self.criterion(logits, y)
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean().item()

        self.loss_log_val.set_step(loss.item())
        self.acc_log_val.set_step(acc)

    def configure_parameters_save(self):
        self.set_parameters_save(model=self.net, indicator=self.loss_log,
                                 save_path='{}/net_{}.pth'.format(self.config['parameter_save_root'], self.config['version']))


if __name__ == "__main__":
    # 随机种子
    random.seed(42)
    torch.manual_seed(42)

    # 构造数据集
    train_set = RandomClsDataset(n=1200, seed=0)
    valid_set = RandomClsDataset(n=300,  seed=1)

    # 构造模块并给定（或直接从文件加载）配置
    module = SimpleClassifierModule(config_path='config/demo.yaml')

    # 训练器
    trainer = PokeTrainer(train_module=module, train_dataset=train_set, valid_dataset=valid_set)
    trainer.configure()
    trainer.fit()
