# FLAN-T5 Fine-tuning for PII Anonymization (Italian)

This project provides a complete Jupyter notebook for fine-tuning Google's FLAN-T5-small model for anonymizing Personally Identifiable Information (PII) in Italian text.

## Overview

- **Model**: FLAN-T5-small (80M parameters)
- **Task**: PII Anonymization
- **Language**: Italian
- **Approach**: Text-to-text format (original text → anonymized text with placeholders)

## Features

- ✅ Complete training pipeline
- ✅ Synthetic Italian PII dataset with 30+ examples
- ✅ Text-to-text anonymization format
- ✅ Training and evaluation
- ✅ Inference examples
- ✅ Model saving and loading
- ✅ GPU support (automatically detected)

## PII Types Supported

The model learns to replace the following types of PII with generic placeholders:
- **[NOME]**: Person names (e.g., Mario Rossi → [NOME])
- **[INDIRIZZO]**: Street addresses (e.g., Via Roma 25 → [INDIRIZZO])
- **[TELEFONO]**: Phone numbers (e.g., 339-1234567 → [TELEFONO])
- **[EMAIL]**: Email addresses (e.g., mario@email.it → [EMAIL])
- **[CARTA_CREDITO]**: Credit card numbers (e.g., 4532-1234-5678-9010 → [CARTA_CREDITO])
- **[CODICE_FISCALE]**: Italian fiscal codes/tax IDs (e.g., RSSMRA85M01H501X → [CODICE_FISCALE])
- **[DATA_NASCITA]**: Dates of birth (e.g., 15/03/1985 → [DATA_NASCITA])

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
uv init && uv sync
```

### Hardware Requirements

Training dataset is really (**really**) small and has been generated using AI. Add more datapoints as needed.

- **Minimum**: CPU with 8GB RAM (slower training)
- **Recommended**: GPU with 8GB+ VRAM (CUDA-enabled)

## Model Input/Output Format

### Input Format
```
anonymize: [Italian text with PII]
```

### Output Format
```
[Same text with PII replaced by placeholders]
```

### Example
**Input**: 
```
anonymize: Mi chiamo Mario Rossi, il mio numero è 339-1234567 e abito in Via Roma 25.
```

**Output**: 
```
Mi chiamo [NOME], il mio numero è [TELEFONO] e abito in [INDIRIZZO].
```

### More Examples

**Input**: 
```
anonymize: Per contattare Giulia Rossi chiamare il 339-8765432 o scrivere a giulia.rossi@email.it
```
**Output**: 
```
Per contattare [NOME] chiamare il [TELEFONO] o scrivere a [EMAIL]
```

**Input**: 
```
anonymize: Il paziente Marco Esposito, nato il 25/08/1982, codice fiscale SPSMRC82M25H501Z.
```
**Output**: 
```
Il paziente [NOME], nato il [DATA_NASCITA], codice fiscale [CODICE_FISCALE].
```

## Using the Fine-tuned Model

After training, load and use the model:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the model
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR)

# Anonymize text
text = "Il signor Paolo Conti abita in Via Garibaldi 45."
input_text = f"anonymize: {text}"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=256)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
# Output: Il signor [NOME] abita in [INDIRIZZO].
```

## Customization

### Expanding the Dataset

The notebook uses synthetic data. To improve performance, you can:

1. Add more examples to the `dataset/ita_dataset.json` list in the notebook
2. Include more variations of PII patterns
3. Add domain-specific examples (medical, legal, financial)
4. Use real (properly consented) data if available

### Using a Larger Model

For better performance, use a larger model:
```python
model_name = "google/flan-t5-base"  # 250M parameters
# or
model_name = "google/flan-t5-large"  # 780M parameters
```

Note: Larger models require more memory and training time.

## References

- [FLAN-T5 Model](https://huggingface.co/google/flan-t5-small)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Italian Privacy Law (GDPR)](https://gdpr-info.eu/)

