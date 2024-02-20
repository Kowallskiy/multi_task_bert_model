import torch
import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

from src.model import MultiTaskBertModel
from src.data_loader import load_dataset
from src.utils import bert_config, tokenizer, intent_ids_to_labels, intent_labels_to_ids


def load_model(model_path):
    """
    Load the pre-trained model weights from the specified path.

    Args:
        model_path (str): Path to the pre-trained model weights.

    Returns:
        model (MultiTaskBertModel): Loaded model with pre-trained weights.
    """
    # Initialize model with configuration and dataset information
    config = bert_config()
    dataset = load_dataset("training_dataset")
    model = MultiTaskBertModel(config, dataset)

    # Load the model weights from the specified path
    model.load_state_dict(torch.load(model_path))

    model.eval()

    return model

def preprocess_input(input_data):
    """
    Preprocess the input text data for inference.

    Args:
        input_data (str): Input text data to be preprocessed.

    Returns:
        input_ids (torch.Tensor): Tensor of input IDs after tokenization.
        attention_mask (torch.Tensor): Tensor of attention mask indicating input tokens.
        offset_mapping (torch.Tensor): Tensor of offset mappings for input tokens.
    """
    # Tokenize the input text and get offset mappings
    tok = tokenizer()
    preprocessed_input = tok(input_data,
                                   return_offsets_mapping=True,
                                   padding='max_length',
                                   truncation=True,
                                   max_length=128)

    # Convert preprocessed inputs to PyTorch tensors
    input_ids = torch.tensor([preprocessed_input['input_ids']])
    attention_mask = torch.tensor([preprocessed_input['attention_mask']])
    offset_mapping = torch.tensor(preprocessed_input['offset_mapping'])
    return input_ids, attention_mask, offset_mapping

def perform_inference(model, input_ids, attention_mask):

    with torch.no_grad():

        ner_logits, intent_logits = model.forward(input_ids, attention_mask)

    return ner_logits, intent_logits
    
def align_ner_predictions_with_input(predictions, offset_mapping, input_text):
    aligned_predictions = []
    current_word_idx = 0

    # Iterate through each prediction and its offset mapping
    for prediction, (start, end) in zip(predictions, offset_mapping):
        if start == end:
            continue
        # Find the corresponding word in the input text
        word = input_text[start:end]

        # Check if the current word is a special token or part of padding
        if not word.strip():
            continue

        # Assign the prediction to the word
        aligned_predictions.append((word, prediction))
    
    return aligned_predictions

def convert_intent_to_label(intent_logit):
    labels = intent_labels_to_ids()
    intent_labels = intent_ids_to_labels(labels)
    return intent_labels[int(intent_logit)]


def main(input_data):
    """
    Main function to perform inference using the pre-trained model.
    """
    # Load the pre-trained model
    model_path = "artifacts/trained_models/pytorch_model.bin"
    model = load_model(model_path)

    # Preprocess the input text
    input_ids, attention_mask, offset_mapping = preprocess_input(input_data)

    # Perform inference using the pre-trained model
    ner_logits, intent_logits = perform_inference(model, input_ids, attention_mask)

    # Post-process the model outputs and print the results
    ner_logits = torch.argmax(ner_logits.view(-1, 9), dim=1)
    intent_logits = torch.argmax(intent_logits)

    ner_logits = align_ner_predictions_with_input(ner_logits, offset_mapping, input_data)
    intent_label = convert_intent_to_label(intent_logits)

    return ner_logits, intent_label
    

if __name__ == "__main__":

    input_data = "I want to schedule a meeting for the 15th of this month at 2:30 PM."
    ner_logits, intent_label = main(input_data)

    print(f"Ner logits: {ner_logits}")
    print(f"Intent logits: {intent_label}")