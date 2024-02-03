from torch.utils.data import DataLoader, Dataset
import torch
from transformers import BertTokenizerFast, BertModel
from transformers import BertConfig, BertPreTrainedModel
import numpy as np
from typing import Dict, List, Union, Tuple
from src.utils import ner_labels_to_ids, intent_labels_to_ids, structure_data

class tokenized_dataset(Dataset):
    """
    A Pytorch Dataset for tokenizing and encoding text data for a BERT-based model.

    Args:
        dataset (dict): A dictionary containing 'text', 'ner', and 'intent' keys.
        tokenizer (BertTokenizerFast): A tokenizer for processing text input.
        max_len (int, optionl): Maximum length of tokenized sequences (default: 128).

    Attributes:
        len (int): Number of samples in the dataset.

    Methods:
        __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
            Retrieve and preprocess a single sample from the dataset.

        __len__(self) -> int:
            Get the total number of samples int the dataset.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing tokenized and encoded text, NER and intent labels.
    """
    def __init__(self, dataset: Dict[str, List[str]], tokenizer: BertTokenizerFast, max_len: int = 128):
        self.len = len(dataset['text'])
        self.ner_labels_to_ids = ner_labels_to_ids()
        self.intent_labels_to_ids = intent_labels_to_ids()
        self.text = dataset['text']
        self.intent = dataset['intent']
        self.ner = dataset['entities']
        self.tokenizer = tokenizer()
        self.max_len = max_len

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # step 1: get the sentence, ner label, and intent_label
        sentence = self.text[index].strip()
        intent_label = self.intent[index].strip()
        ner_labels = self.ner[index]

        # step 2: use tokenizer to encode a sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" which highlights where each token starts and ends
        encoding = self.tokenizer(
            sentence,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )

        # step 3: create ner token labels only for first word pieces of each tokenized word
        tokenized_ner_labels = [self.ner_labels_to_ids[label] for label in ner_labels]
        # create an empty array of -100 of length max_length
        encoded_ner_labels = np.ones(len(encoding['offset_mapping']), dtype=int) * -100

        # set only labels whose first offset position is 0 and the second is not 0
        i = 0
        prev = -1
        for idx, mapping in enumerate(encoding['offset_mapping']):
            if mapping[0] == mapping[1] == 0:
                continue
            if mapping[0] != prev:
                # overwrite label
                encoded_ner_labels[idx] = tokenized_ner_labels[i]
                prev = mapping[1]
                i += 1
            else:
                prev = mapping[1]

        # create intent token labels
        tokenized_intent_label = self.intent_labels_to_ids[intent_label]

        # step 4: turn everything into Pytorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['ner_labels'] = torch.as_tensor(encoded_ner_labels)
        item['intent_labels'] = torch.as_tensor(tokenized_intent_label)

        return item

    def __len__(self) -> int:
        return self.len
    
