# Multi-Task BERT Model for Task Scheduling

## Description
This model is designed for a scheduler application, capable of handling various tasks such as setting timers, scheduling meetings, appointments, 
and alarms. It provides _Named Entity Recognition (NER)_ labels to identify specific entities within the input text, along with an _Intent_ label to 
determine the overall task intention. The model's outputs facilitate efficient task management and organization, enabling seamless interaction with the scheduler application.
* __Named Entity Recognition (NER) Labeling:__ NER labeling enables the model to identify and categorize specific entities within the input text. These entities could
  include dates, times, locations, participants, and other relevant information essential for scheduling tasks. By extracting and labeling these entities, the model can
  accurately understand the details provided in the user's input, thus improving the precision and effectiveness of task scheduling.
* __Intent Labeling:__ Intent labeling allows the model to discern the overall intention or purpose behind the user's input. By classifying the user's request into
  distinct intent categories such as setting timers, scheduling meetings, appointments, or alarms, the model can determine the appropriate action to take in response
  to the user's query. This categorization streamlines the decision-making process, ensuring that the scheduler application responds promptly and accurately to user commands.

## Deployed Demo Model
Experience the model in action through an interactive demonstration available on [HuggingFace Spaces](https://huggingface.co/spaces/kowalsky/multi_task_bert). Dive into the capabilities of the Multi-Task BERT Model and witness firsthand its potential to streamline your scheduling tasks.

## Model Architecture

At the heart of the solution is the MultiTaskBertModel, a class extending __pl.LightningModule__, tailored for dual tasks of _NER_ and _Intent Classification_. It leverages a BertModel for encoding input text, followed by specialized linear layers for each task. This architecture ensures that the model not only captures the essence of each task but also optimizes for performance and accuracy.

```Python
class MultiTaskBertModel(pl.LightningModule):

    """
    Multi-task Bert model for Named Entity Recognition (NER) and Intent Classification

    Args:
        config (BertConfig): Bert model configuration.
        dataset (Dict[str, Union[str, List[str]]]): A dictionary containing keys 'text', 'ner', and 'intent'.
    """

    def __init__(self, config, dataset):
        super().__init__()

        self.num_ner_labels, self.num_intent_labels = num_unique_labels(dataset)

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        self.model = BertModel(config=config)

        self.ner_classifier = torch.nn.Linear(config.hidden_size, self.num_ner_labels)
        self.intent_classifier = torch.nn.Linear(config.hidden_size, self.num_intent_labels)

        # log hyperparameters
        self.save_hyperparameters()

        self.accuracy = MyAccuracy()

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Perform a forward pass through Multi-task Bert model.

        Args:
            input_ids (torch.Tensor, torch.shape: (batch, length_of_tokenized_sequences)): Input token IDs.
            attention_mask (Optional[torch.Tensor]): Attention mask for input tokens.

        Returns:
            Tuple[torch.Tensor,torch.Tensor]: NER logits, Intent logits.
        """

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        ner_logits = self.ner_classifier(sequence_output)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)

        return ner_logits, intent_logits

    def training_step(self: pl.LightningModule, batch, batch_idx: int) -> torch.Tensor:
        """
        Perform a training step for the Multi-task BERT model.

        Args:
            batch: Input batch.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value
        """
        loss, ner_logits, intent_logits, ner_labels, intent_labels = self._common_step(batch, batch_idx)
        accuracy_ner = self.accuracy(ner_logits, ner_labels, self.num_ner_labels)
        accuracy_intent = self.accuracy(intent_logits, intent_labels, self.num_intent_labels)
        self.log_dict({'training_loss': loss, 'ner_accuracy': accuracy_ner, 'intent_accuracy': accuracy_intent},
                      on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.validation_step_outputs_ner = []
        self.validation_step_outputs_intent = []

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Perform a validation step for the Multi-task BERT model.

        Args:
            batch: Input batch.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value.
        """
        loss, ner_logits, intent_logits, ner_labels, intent_labels = self._common_step(batch, batch_idx)
        # self.log('val_loss', loss)
        accuracy_ner = self.accuracy(ner_logits, ner_labels, self.num_ner_labels)
        accuracy_intent = self.accuracy(intent_logits, intent_labels, self.num_intent_labels)
        self.log_dict({'validation_loss': loss, 'val_ner_accuracy': accuracy_ner, 'val_intent_accuracy': accuracy_intent},
                      on_step=False, on_epoch=True, prog_bar=True)

        self.validation_step_outputs_ner.append(ner_logits)
        self.validation_step_outputs_intent.append(intent_logits)
        return loss

    def on_validation_epoch_end(self):
        """
        Perform actions at the end of validation epoch to track the training process in WandB.
        """
        validation_step_outputs_ner = self.validation_step_outputs_ner
        validation_step_outputs_intent = self.validation_step_outputs_intent

        dummy_input = torch.zeros((1, 128), device=self.device, dtype=torch.long)
        model_filename = f"model_{str(self.global_step).zfill(5)}.onnx"
        torch.onnx.export(self, dummy_input, model_filename)
        artifact = wandb.Artifact(name="model.ckpt", type="model")
        artifact.add_file(model_filename)
        self.logger.experiment.log_artifact(artifact)

        flattened_logits_ner = torch.flatten(torch.cat(validation_step_outputs_ner))
        flattened_logits_intent = torch.flatten(torch.cat(validation_step_outputs_intent))
        self.logger.experiment.log(
            {"valid/ner_logits": wandb.Histogram(flattened_logits_ner.to('cpu')),
             "valid/intent_logits": wandb.Histogram(flattened_logits_intent.to('cpu')),
             "global_step": self.global_step}
        )

    def _common_step(self, batch, batch_idx):
        """
        Common steps for both training and validation. Calculate loss for both NER and intent layer.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            Combiner loss value, NER logits, intent logits, NER labels, intent labels.
        """
        ids = batch['input_ids']
        mask = batch['attention_mask']
        ner_labels = batch['ner_labels']
        intent_labels = batch['intent_labels']

        ner_logits, intent_logits = self.forward(input_ids=ids, attention_mask=mask)

        criterion = torch.nn.CrossEntropyLoss()

        ner_loss = criterion(ner_logits.view(-1, self.num_ner_labels), ner_labels.view(-1).long())
        intent_loss = criterion(intent_logits.view(-1, self.num_intent_labels), intent_labels.view(-1).long())

        loss = ner_loss + intent_loss
        return loss, ner_logits, intent_logits, ner_labels, intent_labels

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer
```

## Training

The model training process is streamlined with the integration of __WandB__ for real-time tracking of training metrics and outputs. Training and validation steps are meticulously designed to evaluate the model's performance accurately, ensuring that each iteration improves upon the last.

```Python
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
```

