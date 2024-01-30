from transformers import BertTokenizerFast, BertConfig
from typing import Dict, List, Union, Tuple


def num_unique_labels(dataset: Dict[str, Union[str, List[str]]]) -> Tuple[int, int]:
    """
    Calculate the number of NER labels and INTENT labels in the dataset.

    Args:
        dataset (dict): A dictionary containing 'text', 'entities' and 'intent' keys.

    Returns:
        Tuple: Number of unique NER and INTENT lables.
    """
    one_dimensional_ner = [tag for subset in dataset['entities'] for tag in subset]
    return len(set(one_dimensional_ner)), len(set(dataset['intent']))

def ner_labels_to_ids() -> Dict[str, int]:
    """
    Map NER labels to corresponding numeric IDs.

    Returns:
        Dict[str, int]: A dictionary where keys are NER labels, and values are their corresponding IDs.
    """
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

def ner_ids_to_labels(ner_labels_to_ids) -> Dict[int, str]:
    """
    Map numeric IDs to corresponding NER labels.

    Returns:
        Dict[int, str]: A dictionary where keys are numeric IDs, and values are their corresponding NER labels.
    """
    ner_ids_to_labels = {v: k for k, v in ner_labels_to_ids.items()}
    return ner_ids_to_labels

def intent_labels_to_ids() -> Dict[str, int]:
    """
    Map intent labels to corresponding numeric values.

    Returns:
        Dict[str, int]: A dictionary where keys are intent labels, and values are their corresponding numeric IDs.
    """
    intent_labels_to_ids = {
    "'Schedule Appointment'": 0,
    "'Schedule Meeting'": 1,
    "'Set Alarm'": 2,
    "'Set Reminder'": 3,
    "'Set Timer'": 4
    }
    return intent_labels_to_ids

def intent_ids_to_labels(intent_labels_to_ids) -> Dict[int, str]:
    """
    Map numeric values to corresponding intent labels.

    Returns:
        Dict[int, str]: A dictionary where keys are numeric IDs, and values are their corresponding intent labels.
    """
    intent_ids_to_labels = {v: k for k, v in intent_labels_to_ids.items()}
    return intent_ids_to_labels

def tokenizer() -> BertTokenizerFast:
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    return tokenizer

def bert_config() -> BertConfig:
    config = BertConfig.from_pretrained('bert-base-uncased')
    return config

def structure_data(dataset):
    structured_data = {'text': [], 'entities': [], 'intent': []}
    for sample in dataset:
        structured_data['text'].append(sample['text'])
        structured_data['entities'].append(sample['entities'].split())
        structured_data['intent'].append(sample['intent'])
    return structured_data