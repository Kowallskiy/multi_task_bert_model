import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from typing import Dict


class DataModule(pl.LightningDataModule):
    """
    Lightning DataModule for handling training and validation datasets.

    Args:
        training_set (torch.utils.data.Dataset): Training dataset.
        validation_set (torch.utils.data.Dataset): Validation dataset.

    Attributes:
        training_set (torch.utils.data.Dataset): Training dataset.
        validation_set (torch.utils.data.Dataset): Validation dataset.
        train_ds (torch.utils.data.Dataset): Alias for the training dataset during setup.
        val_ds (torch.utils.data.Dataset): Alias for the validation dataset during setup.

    Methods:
        setup(self, stage: Optional[str] = None):
            Setup method to load and preprocess datasets.

        train_dataloader(self) -> DataLoader:
            Return a DataLoader for the training dataset.

        val_dataloader(self) -> DataLoader:
            Return a DataLoader for the validation dataset.
    """
    def __init__(self, training_set, validation_set):
        super().__init__()
        self.training_set = training_set
        self.validation_set = validation_set

    def setup(self, stage: str):
        self.train_ds = self.training_set
        self.val_ds = self.validation_set

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=1, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, shuffle=False)