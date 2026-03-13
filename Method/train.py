import torch
import torch.nn as nn

from Poke.pipeline import PokeBaseModule, PokeLog, PokeBaseDataloader, PokeTrainer, PokeConfig
from model.model_demo import DemoNet
from dataset import RandomClsDataset
from visualization import visualization

# ---- 继承与重写 PokeBaseModule ----
class SimpleClassifierModule(PokeBaseModule):
    def __init__(self, config):
        super().__init__(config=config)
        lr = self.config['model']['lr']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = DemoNet(self.config.model.get('in_channels'), self.config.model.get('out_channels'))

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=2, gamma=0.5)

    def set_scheduler(self):
        self.scheduler.step()

    def visualization(self, image_box):
        return visualization(image_box)

    def configure_logs(self):
        self.log_lr = PokeLog(log_name="lr", log_type="lr", log_location='all')
        self.log_loss = PokeLog(log_name="loss", log_type="loss", log_location='all')
        self.log_acc = PokeLog(log_name="acc", log_type="acc", log_location='all')
        self.log_image = PokeLog(log_name="result_image", log_type="image", log_location='valid')
        return self.log_lr, self.log_loss, self.log_acc, self.log_image

    # 必须：实现单步训练逻辑，返回一个 dict 的指标
    def train_step(self, batch_idx, batch):
        self.net.train()

        self.log_lr.train.set_step(self.optimizer.state_dict()['param_groups'][0]['lr'])
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
        self.log_loss.train.set_step(loss.item())
        self.log_acc.train.set_step(acc)

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

        self.log_loss.valid.set_step(loss.item())
        self.log_acc.valid.set_step(acc)
        if batch_idx % 300 == 0:
            self.log_image.valid.set_step([torch.randn(50, 3, 224, 224).to(self.device)])

    def configure_parameters(self):
        self.set_parameters(
            name="net",
            model=self.net,
            optimizer=self.optimizer,
            indicator=self.log_loss.valid,
            ckpt_dir=f"{self.config.model.get('parameter_dir')}/{self.config['version']}",
            save_every_epochs=5,
            keep_last_k=3,
        )


if __name__ == "__main__":
    # ===== Build datasets =====
    train_set = RandomClsDataset(n=1200, seed=0)
    valid_set = RandomClsDataset(n=300, seed=1)

    # ===== Load configuration =====
    config = PokeConfig(path='config/demo.yaml')

    # ===== Build dataloader =====
    dataloader = PokeBaseDataloader(config, train_dataset=train_set, valid_dataset=valid_set)

    # ===== Build model module =====
    module = SimpleClassifierModule(config)
    # Uncomment to resume training from the latest checkpoint
    # module.load_latest('/Users/wanghaolin/GitHub/PokeLab/Method/parameters/demo/net_latest.ckpt', continue_epoch=True)

    # ===== Build trainer =====
    trainer = PokeTrainer(config=config, train_module=module, train_loader=dataloader.train_loader, valid_loader=dataloader.valid_loader)

    # ===== Start training =====
    trainer.fit()
