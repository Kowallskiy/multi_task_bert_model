from src.data_loader import load_dataset
import pytest
import regex as re


def test_load_dataset_invalid_name():
    # Test loading a dataset with an invalid name
    valid_names = ['training_dataset', 'validation_dataset']
    dataset_name = 'invalid_dataset'
    matched = re.escape(f"Invalid dataset name. Expected one of {valid_names}, got {dataset_name}")
    
    with pytest.raises(ValueError, match=matched):
        try:
            load_dataset(dataset_name)
        except ValueError as e:
            print(f"Actual error message: {str(e)}")
            raise

if __name__ == "__main__":
    pytest.main()