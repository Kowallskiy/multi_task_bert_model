import pytest
import torch
from src.data_module import DataModule
from torch.utils.data import TensorDataset

# Example test data
train_data = TensorDataset(torch.randn(10, 3), torch.randint(0, 2, (10,)))
val_data = TensorDataset(torch.randn(5, 3), torch.randint(0, 2, (5,)))

def test_data_module_setup():
    data_module = DataModule(train_data, val_data)
    data_module.setup("fit")

    assert data_module.train_ds is not None
    assert data_module.val_ds is not None

def test_data_module_train_dataloader():
    data_module = DataModule(train_data, val_data)
    data_module.setup("fit")
    data_loader = data_module.train_dataloader()

    assert isinstance(data_loader, torch.utils.data.DataLoader)

def test_data_module_val_dataloader():
    data_module = DataModule(train_data, val_data)
    data_module.setup("fit")
    data_loader = data_module.val_dataloader()

    assert isinstance(data_loader, torch.utils.data.DataLoader)

if __name__ == "__main__":
    pytest.main()