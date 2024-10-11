from torchtune.datasets import instruct_dataset
from torchtune.data import InstructTemplate
from transformers import AutoTokenizer
from typing import Mapping, Optional, Dict, Any
import dotenv
import os

dotenv.load_dotenv()
MODEL_NAME= os.getenv("MODEL_NAME")

class CustomTemplate(InstructTemplate):
    # Define the template as string with {} as placeholders for data columns
    template = "Input: {input}\nOutput:{output}"

    # Implement this method
    @classmethod
    def format(
        cls, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
    ) -> str:
        # Use the column_map to get the correct keys from the sample
        input_field = column_map.get("input", "input") if column_map else "input"
        output_field = column_map.get("output", "output") if column_map else "output"

        # Extract the values using the column_map
        input_text = sample.get(input_field, "")
        output_text = sample.get(output_field, "")

        # Format the string using the template
        formatted_string = cls.template.format(input=input_text, output=output_text)
        return formatted_string
    
# Load in tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
dataset = instruct_dataset(
    tokenizer=tokenizer,
    source="json",
    data_files="/home/mehmet/codebase/llm_project/data/CustomeDataNoInstructions.json",
    train_on_input=False,
    packed=False,
    split="train",
)