import torch
import os
import sys
from transformers import BertTokenizerFast
import numpy as np

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

from src.model import MultiTaskBertModel
from src.data_loader import load_dataset
from src.utils import bert_config, tokenizer

def load_model(model_path):
    config = bert_config()
    dataset = load_dataset("training_dataset")

    model = MultiTaskBertModel(config, dataset)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    return model

def preprocess_input(input_data):
    tok = tokenizer()
    preprocessed_input = tok(input_data,
                                   return_offsets_mapping=True,
                                   padding='max_length',
                                   truncation=True,
                                   max_length=128)

    input_ids = torch.tensor([preprocessed_input['input_ids']])
    attention_mask = torch.tensor([preprocessed_input['attention_mask']])
    offset_mapping = torch.tensor(preprocessed_input['offset_mapping'])
    return input_ids, attention_mask, offset_mapping

def perform_inference(model, input_ids, attention_mask):

    with torch.no_grad():

        ner_logits, intent_logits = model.forward(input_ids, attention_mask)

    return ner_logits, intent_logits
    
def align_predictions_with_input(predictions, offset_mapping, input_text):
    aligned_predictions = []
    current_word_idx = 0

    # Iterate through each prediction and its offset mapping
    for prediction, (start, end) in zip(predictions, offset_mapping):
        # Find the corresponding word in the input text
        word = input_text[start:end]

        # Check if the current word is a special token or part of padding
        if not word.strip():
            continue

        # Assign the prediction to the word
        aligned_predictions.append((word, prediction))
    
    return aligned_predictions

def main():
    model_path = "C:/Users/Userpc/Desktop/model/lit-wandb/a5yyvgve/checkpoints/epoch=0-step=35.ckpt"
    model = load_model(model_path)

    input_data = "Set a timer for 15 minutes"
    input_ids, attention_mask, offset_mapping = preprocess_input(input_data)

    ner_logits, intent_logits = perform_inference(model, input_ids, attention_mask)

    # print(f"Ner logits: {ner_logits.view(-1, 9).shape}")
    # print(f"Intent logits: {intent_logits}")

    ner_logits = torch.argmax(ner_logits.view(-1, 9), dim=1)
    intent_logits = torch.argmax(intent_logits)

    print(offset_mapping)

    ner_logits = align_predictions_with_input(ner_logits, offset_mapping, input_data)

    print(f"Ner logits: {ner_logits}")
    print(f"Intent logits: {intent_logits}")

if __name__ == "__main__":
    main()
