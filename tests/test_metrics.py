import torch
import pytest
from src.metrics import MyAccuracy

def test_my_accuracy_metric():
    # Create an instance of the MyAccuracy metric
    accuracy_metric = MyAccuracy()

    # Mock data for testing
    num_labels = 2
    logits = torch.randn(1, 128, num_labels)
    labels = torch.randn(1, 128)  # -100 represents padded values
    
    # Update the accuracy metric
    accuracy_metric.update(logits, labels, num_labels)

    # Compute accuracy
    accuracy = accuracy_metric.compute()

    # Assert the correctness of the computed accuracy
    assert 0.0 <= accuracy <= 1.0  # You should replace this with the expected accuracy based on your input

if __name__ == "__main__":
    pytest.main()