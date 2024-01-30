from src.utils import num_unique_labels, ner_ids_to_labels, structure_data
import pytest

def test_num_unique_labels():
    # Test case 1: No entities, empty intent
    dataset = {'entities': [], 'intent': []}
    result = num_unique_labels(dataset)
    assert result == (0, 0)

    # Test case 2: Multiple entities, empty intent
    dataset = {'entities': [['B-DATE', 'I-DATE'], ['B-TASK']], 'intent': []}
    result = num_unique_labels(dataset)
    assert result == (3, 0)

    # Test case 3: No entities, non-empty intent
    dataset = {'entities': [], 'intent': ['intent_1', 'intent_2']}
    result = num_unique_labels(dataset)
    assert result == (0, 2)

    # Test case 4: Multiple entities, non-empty intent
    dataset = {'entities': [['B-DATE', 'I-DATE'], ['B-TASK']], 'intent': ['intent_1', 'intent_2']}
    result = num_unique_labels(dataset)
    assert result == (3, 2)

def test_ner_ids_to_labels():
    # Test case 1: Empty dictionary
    input_dict = {}
    result = ner_ids_to_labels(input_dict)
    assert result == {}

    # Test case 2: Non-empty dictionary
    input_dict = {1: 'B-DATE', 2: 'I-DATE', 3: 'B-TASK'}
    result = ner_ids_to_labels(input_dict)
    expected_result = {'B-DATE': 1, 'I-DATE': 2, 'B-TASK': 3}
    assert result == expected_result

def test_structure_data():
    # Test case 1: Empty dataset
    input_dataset = []
    result = structure_data(input_dataset)
    expected_result = {'text': [], 'entities': [], 'intent': []}
    assert result == expected_result

    # Test case 2: Non-empty dataset with one sample
    input_dataset = [{'text': 'Hello world!', 'entities': 'O O', 'intent': 'Greeting'}]
    result = structure_data(input_dataset)
    expected_result = {'text': ['Hello world!'], 'entities': [['O', 'O']], 'intent': ['Greeting']}
    assert result == expected_result

    # Test case 3: Non-empty dataset with multiple samples
    input_dataset = [
        {'text': 'Sample 1', 'entities': 'B-DATE O', 'intent': 'Intent1'},
        {'text': 'Sample 2', 'entities': 'B-TASK I-TASK', 'intent': 'Intent2'}
    ]
    result = structure_data(input_dataset)
    expected_result = {'text': ['Sample 1', 'Sample 2'], 'entities': [['B-DATE', 'O'], ['B-TASK', 'I-TASK']],
                       'intent': ['Intent1', 'Intent2']}
    assert result == expected_result

if __name__ == "__main__":
    pytest.main()
