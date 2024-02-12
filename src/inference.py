import torch
import os
import sys
from transformers import BertTokenizerFast

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
    return input_ids, attention_mask

def perform_inference(model, input_ids, attention_mask):

    with torch.no_grad():

        ner_logits, intent_logits = model.forward(input_ids, attention_mask)

    return ner_logits, intent_logits
    
def postprocess_output(ner_logits, intent_logits):
    pass

def main():
    model_path = "C:/Users/Userpc/Desktop/model/lit-wandb/a5yyvgve/checkpoints/epoch=0-step=35.ckpt"
    model = load_model(model_path)

    input_data = "Set a timer for 15 minutes"
    input_ids, attention_mask = preprocess_input(input_data)

    ner_logits, intent_logits = perform_inference(model, input_ids, attention_mask)

    print(f"Ner logits: {ner_logits.view(-1, 9).shape}")
    print(f"Intent logits: {intent_logits}")

    ner_logits = torch.argmax(ner_logits.view(-1, 9), dim=1)
    intent_logits = torch.argmax(intent_logits)

    print(f"Ner logits: {ner_logits}")
    print(f"Intent logits: {intent_logits}")

if __name__ == "__main__":
    main()
