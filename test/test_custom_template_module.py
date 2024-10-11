import os
import pytest
from torchtune.datasets import instruct_dataset
from modules.custom_template_module import CustomTemplate  # Import your custom template here
from torchtune.models.llama3 import Llama3Tokenizer
import dotenv

# Load environment variables
dotenv.load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME")

def test_environment_variable():
    # Test if the environment variable is loaded correctly
    assert MODEL_NAME is not None, "MODEL_NAME environment variable should be set"
    print(f"MODEL_NAME: {MODEL_NAME}")

def test_custom_template_formatting():
    # Test data
    sample_data = [{
        "input": "This is a test input.",
        "output": "This is the expected output."
    },
    {
        "input": "This is a test input.",
        "output": "This is the expected output."
    }]
    
    
    # Expected format
    expected_format = "Input: This is a test input.\nOutput:This is the expected output."
    
    # Format using CustomTemplate
    formatted_string = CustomTemplate.format(sample_data)
    
    # Check if the formatted string matches the expected format
    assert formatted_string == expected_format, f"Expected: {expected_format}, but got: {formatted_string}"
    print(f"Formatted text: {formatted_string}")

def test_dataset_loading():
    # Load the tokenizer
    tokenizer = Llama3Tokenizer('/home/mehmet/codebase/llm_project/models/Llama-3.2-1B/original/tokenizer.model')

    # File path for the dataset (make sure this is correct)
    data_files = "/home/mehmet/codebase/llm_project/data/CustomeDataNoInstructions.json"

    # Test if the dataset file exists
    assert os.path.exists(data_files), f"Dataset file {data_files} does not exist."

    # Load the dataset using instruct_dataset
    dataset = instruct_dataset(
        tokenizer=tokenizer,
        source="json",
        split="train",
        data_files=data_files,
        column_map={
            "input": "input",
            "output": "output",
        },
        train_on_input=False,
        packed=False,
    )

    # Check if the dataset is not empty
    assert len(dataset) > 0, "Dataset should not be empty"

    # Check tokenization on the first sample
    tokenized = dataset[0]['tokens']
    
    # Assert that 'input_ids' is in the tokenized output
    assert "input_ids" in tokenized, "Tokenization output should contain 'input_ids'"
    print(f"First tokenized sample: {tokenized}")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__])
