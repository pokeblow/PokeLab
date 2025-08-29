import random
import torch
import torch.nn as nn

from Poke.pipeline.poke_trainer import BaseTrainModule, PokeTrainer
from Poke.pipeline.poke_log import PokeLog
from model.model_demo import DemoNet
from dataset import RandomClsDataset


# ---- 继承与重写 BaseTrainModule ----
class SimpleClassifierModule(BaseTrainModule):
    def __init__(self, config_path: str = ""):
        super().__init__(config_path=config_path)
        lr = self.config['model']['lr']

        self.net = DemoNet(self.config['model']['in_channels'], self.config['model']['out_channels'])

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=2, gamma=0.5)

    def set_scheduler(self):
        self.scheduler.step()
    def configure_logs(self):
        self.lr_log = PokeLog(log_name="lr", log_type="lr", log_location='all')
        self.loss_log = PokeLog(log_name="loss", log_type="loss", log_location='train')
        self.loss_log_val = PokeLog(log_name="loss", log_type="loss", log_location='valid')
        self.acc_log = PokeLog(log_name="acc", log_type="acc", log_location='train')
        self.acc_log_val = PokeLog(log_name="acc", log_type="acc", log_location='valid')
        return self.lr_log, self.loss_log, self.loss_log_val, self.acc_log, self.acc_log_val

    # 必须：实现单步训练逻辑，返回一个 dict 的指标
    def train_step(self, batch_idx, batch):
        self.net.train()

        self.lr_log.set_step(self.optimizer.state_dict()['param_groups'][0]['lr'])
        x, y = batch
        x = x.to(self.device, dtype=torch.float32)
        y = y.to(self.device, dtype=torch.long)

        logits = self.net(x)
        loss = self.criterion(logits, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean().item()
        self.loss_log.set_step(loss.item())
        self.acc_log.set_step(acc)

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
    # 构造数据集
    train_set = RandomClsDataset(n=1200, seed=0)
    valid_set = RandomClsDataset(n=300,  seed=1)

    # 构造模块并给定（或直接从文件加载）配置
    module = SimpleClassifierModule(config_path='config/demo.yaml')

    # 训练器
    trainer = PokeTrainer(train_module=module, train_dataset=train_set, valid_dataset=valid_set)
    trainer.configure()
    trainer.fit()
