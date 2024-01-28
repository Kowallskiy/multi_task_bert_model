import wandb
import pytorch_lightning as pl

from model import MultiTaskBertModel
from utils import bert_config, tokenizer
from data_loader import load_dataset
from data_processing import tokenized_dataset
from data_module import DataModule

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
    max_epochs=2,
    deterministic=True)
# a808ea3766056b84413558f8145c7abf7c2e9e76
trainer.fit(model, dm)

wandb.finish()