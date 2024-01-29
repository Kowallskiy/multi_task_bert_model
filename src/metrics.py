import torch
from torchmetrics import Metric

class MyAccuracy(Metric):
    """
    Accuracy metric costomized for handling sequences with padding.

    Methods:
        update(self, logits, labels, num_labels): Update the accuracy based on 
        model predictions and ground truth labels.

        compute(self): Compute the accuracy.

    Attributes:
        total (torch.Tensor): Total number of non-padding elements.
        correct (torch.Tensor): Number of correctly predicted non-padding elements.
    """
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, logits: torch.Tensor, labels: torch.Tensor, num_labels: int) -> None:
        """
        Args:
            logits (torch.Tensor): Model predictions.
            labels (torch.Tensor): Ground truth labels.
            num_labels (int): Number of unique labels.
        """
        flattened_targets = labels.view(-1) # shape (batch_size, sequence_len)
        active_logits = logits.view(-1, num_labels) # shape (batch_size * sequence_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * sequence_len)

        # compute accuracy only at active labels
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, sequnce_len)
        ac_labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        self.correct += torch.sum(ac_labels == predictions)
        self.total += torch.numel(ac_labels)

    def compute(self) -> torch.Tensor:
        """
        Calculate the accuracy.
        """
        return self.correct.float() / self.total.float()