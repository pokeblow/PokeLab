from torch.utils.data import DataLoader
from ..configure import PokeConfig

class PokeBaseDataloader:
    def __init__(self, config: PokeConfig, train_dataset=None, valid_dataset=None, test_dataset=None):
        super().__init__()
        self.config = config
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

        self.train_loader = self.set_train_dataloader()
        self.valid_loader = self.set_valid_dataloader()
        if self.test_dataset != None:
            self.test_loader = self.set_test_dataloader()

    def set_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.run.get('batch_size'),
            shuffle=True,
            num_workers=self.config.dataloader.get('num_workers', 0),
        )

    def set_valid_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.config.run.get('batch_size'),
            shuffle=True,
            num_workers=self.config.dataloader.get('num_workers', 0),
        )

    def set_test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.run.get('batch_size'),
            shuffle=True,
            num_workers=self.config.dataloader.get('num_workers', 0),
        )