import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32):
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str = None):
        self.cifar10_test = CIFAR10(self.data_dir, train=False)
        self.cifar10_full = CIFAR10(self.data_dir, train=True)
        self.cifar10_train, self.cifar10_val = random_split(
            self.cifar10_full, [0.7, 0.3], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=self.batch_size)


cifar10 = CIFAR10DataModule("./data")

print("ok")
