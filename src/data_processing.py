from torch.utils.data import DataLoader, Dataset
import torch
from transformers import BertTokenizerFast, BertModel
from transformers import BertConfig, BertPreTrainedModel
import numpy as np


def num_unique_unique_labels(dataset):
    one_dimensional_ner = [tag for subset in dataset['ner'] for tag in subset]
    return len(set(one_dimensional_ner)), len(set(dataset['intent']))

def ner_labels_to_ids():
    labels_to_ids_ner = {
    'O': 0,
    'B-DATE': 1,
    'I-DATE': 2,
    'B-TIME': 3,
    'I-TIME': 4,
    'B-TASK': 5,
    'I-TASK': 6,
    'B-DUR': 7,
    'I-DUR': 8
    }
    return labels_to_ids_ner

def ner_ids_to_labels(ner_labels_to_ids):
    ner_ids_to_labels = {v: k for k, v in ner_labels_to_ids.items()}
    return ner_ids_to_labels

def intent_labels_to_ids():
    intent_labels_to_ids = {
    "'Schedule Appointment'": 0,
    "'Schedule Meeting'": 1,
    "'Set Alarm'": 2,
    "'Set Reminder'": 3,
    "'Set Timer'": 4
    }
    return intent_labels_to_ids

def intent_ids_to_labels(intent_labels_to_ids):
    intent_ids_to_labels = {v: k for k, v in intent_labels_to_ids.items()}
    return intent_ids_to_labels

def tokenizer():
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    return tokenizer

class tokenized_dataset(Dataset):
    def __init__(self, dataset, tokenizer, ner_labels_to_ids, max_len=128):
        self.len = len(dataset['text'])
        self.text = dataset['text']
        self.intent = dataset['intent']
        self.ner = dataset['ner']
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
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
        tokenized_ner_labels = [ner_labels_to_ids[label] for label in ner_labels]
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
        tokenized_intent_label = intent_labels_to_ids[intent_label]

        # step 4: turn everything into Pytorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['ner_labels'] = torch.as_tensor(encoded_ner_labels)
        item['intent_labels'] = torch.as_tensor(tokenized_intent_label)

        return item

    def __len__(self):
        return self.len
    
