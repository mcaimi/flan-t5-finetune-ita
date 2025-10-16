#!/usr/bin/env python

try:
    import torch
    import json
except ImportError as e:
    raise e

# create a new dataset class that holds training data information
class CustomPIIDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str) -> None:
        from pathlib import Path

        # path that contains json datasets
        self.dataset_path: Path = Path(dataset_path)
        
        # find datafiles
        self.datasets: list = [f for f in self.dataset_path.glob("**/*.json")]

        # load dataset
        self.dataset: list = []
        for fname in self.datasets:
            print(f"Loading {fname}...")
            with open(fname, "r") as json_dataset:
                self.dataset.extend(json.load(json_dataset))
            
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> dict:
        # return datapoint
        return self.dataset[idx]

# preprocessing stuff
class DataPreprocessor():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    # preprocessing function to prepare dataset for training
    def data_preprocess(self, examples):
        """
        Convert PII anonymization data to T5 text-to-text format
        Input: "anonymize: [original text]"
        Output: "[anonymized text]"
        """
        inputs = []
        targets = []
        
        for original, anonymized in zip(examples['original'], examples['anonymized']):
            # Create input with task prefix
            input_text = f"anonymize: {original}"
            inputs.append(input_text)
            
            # Target is the anonymized text
            targets.append(anonymized)
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            inputs,
            max_length=256,  # Increased for longer Italian sentences
            truncation=True,
            padding=False
        )
        
        # Tokenize targets
        labels = self.tokenizer(
            targets,
            max_length=256,
            truncation=True,
            padding=False
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

# test function
def anonymize_text(text, model, tokenizer):
    """
    Anonymize PII in Italian text using the fine-tuned model
    """
    # Prepare input
    input_text = f"anonymize: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True)
    
    # Move to device if using GPU
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        model = model.to("cuda")
    
    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )
    
    # Decode output
    anonymized = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return anonymized