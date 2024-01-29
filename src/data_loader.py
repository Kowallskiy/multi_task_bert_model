import json
import os
from typing import Dict, List, Union
from src.utils import structure_data

def load_dataset(dataset_name: str) -> Dict[str, Union[str, List[str]]]:
    """
    Load training dataset or validation dataset.

    Args:
        dataset_name (str): The name of the dataset. Should be either 'training_dataset' or 'validation_dataset'.

    Returns:
        dataset (Dict[str, Union[str. List[str]]]): A dictionary representing the 
        loaded dataset with keys 'text', 'ner', and 'intent'.

    Raises:
        ValueError: If the provided dataset_name is not one of the valid_names.
        FileNotFoundError: If the dataset file is not found in the specified path.
    """

    valid_names = ["training_dataset", "validation_dataset"]

    if dataset_name not in valid_names:
        raise ValueError(f"Invalid dataset name. Expected one of {valid_names}, got {dataset_name}")
    
    path = f"C:/Users/Userpc/Desktop/model/data/raw/{dataset_name}.json"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at {path}")
    
    with open(path, 'r') as f:
        dataset = json.load(f)

    dataset = structure_data(dataset)

    return dataset