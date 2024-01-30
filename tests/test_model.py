import pytest
import torch
from src.model import MultiTaskBertModel
from src.utils import bert_config

def test_multi_task_bert_model():
    # Step 1: Mock Data
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 0, 0]])
    ner_labels = torch.tensor([[0, 1, 2], [2, 0, 1]])
    intent_labels = torch.tensor([[0], [0]])

    # Step 2: Instantiate Model
    config = bert_config()
    dataset = {'text': ['what is love'], 'entities': ['A', 'B', 'C'], 'intent': 'love'}
    num_ner_labels, num_intent_labels = 3, 1
    model = MultiTaskBertModel(config, dataset)

    # Step 3: Forward Pass
    ner_logits, intent_logits = model.forward(input_ids, attention_mask)

    # Step 4: Loss Calculation
    loss, _, _, _, _ = model._common_step({'input_ids': input_ids, 'attention_mask': attention_mask,
                                           'ner_labels': ner_labels, 'intent_labels': intent_labels}, 0)

    # Step 5: Training Step
    training_loss = model.training_step({'input_ids': input_ids, 'attention_mask': attention_mask,
                                         'ner_labels': ner_labels, 'intent_labels': intent_labels}, 0)

    # Step 6: Assertions
    assert isinstance(loss, torch.Tensor)
    assert 0 <= training_loss


if __name__ == "__main__":
    pytest.main()