import wandb
import pytorch_lightning as pl
import sys
import os

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

from src.model import MultiTaskBertModel
from src.utils import bert_config, tokenizer
from src.data_loader import load_dataset
from src.data_processing import tokenized_dataset
from src.data_module import DataModule

wandb.login()

config = bert_config()

training_data_raw = load_dataset('training_dataset')
validation_data_raw = load_dataset('validation_dataset')

tokenized_training_dataset = tokenized_dataset(training_data_raw, tokenizer)
tokenized_validation_dataset = tokenized_dataset(validation_data_raw, tokenizer)

wandb_logger = pl.loggers.WandbLogger(project='lit-wandb')

model = MultiTaskBertModel(config, training_data_raw)

dm = DataModule(tokenized_training_dataset, tokenized_validation_dataset)

trainer = pl.Trainer(
    logger = wandb_logger,
    log_every_n_steps=40,
    max_epochs=3,
    deterministic=True,
    profiler='simple')

trainer.fit(model, dm)

wandb.finish()