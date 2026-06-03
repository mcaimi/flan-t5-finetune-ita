# FLAN-T5 Fine-tuning for PII Anonymization (Italian)

Fine-tune Google's [FLAN-T5](https://huggingface.co/google/flan-t5-small) seq2seq model to detect and replace Personally Identifiable Information (PII) in Italian text with standardized placeholders. The project is aimed at GDPR-aligned document anonymization workflows: train locally in Jupyter, run inference from Python, export to ONNX, or orchestrate training on Kubernetes with Kubeflow Pipelines.

## Purpose

Italian documents often contain names, addresses, fiscal codes, and other sensitive fields that must be redacted before sharing or archiving. Rule-based anonymizers struggle with natural language variation. This repository treats anonymization as a **text-to-text** task:

- **Input**: Italian sentence containing PII, prefixed with `anonymize: `
- **Output**: Same sentence with PII replaced by tags such as `[NOME]`, `[EMAIL]`, or `[CODICE_FISCALE]`

The base model is FLAN-T5-small (~80M parameters), fine-tuned on a synthetic Italian dataset. The training data is intentionally small and AI-generated; expand it before relying on the model in production.

## Features

- End-to-end fine-tuning pipeline in Jupyter (download base model, train, evaluate, save checkpoint)
- Synthetic Italian PII dataset with 30+ examples (`dataset/`)
- Reusable Python library (`jupyter/libs/`) for dataset loading, preprocessing, and inference
- Demo CLI script (`main.py`) with batch examples and interactive mode
- ONNX export notebook for deployment outside PyTorch
- Kubeflow Pipeline for automated training on OpenShift / OCP AI (S3 datasets, Hugging Face, MLflow)
- Automatic GPU / MPS / CPU detection
- MLflow experiment tracking integration

## PII Types Supported

The model learns to replace the following types of PII with generic placeholders:

| Placeholder | Description | Example |
|---|---|---|
| `[NOME]` | Person names | Mario Rossi → `[NOME]` |
| `[INDIRIZZO]` | Street addresses | Via Roma 25 → `[INDIRIZZO]` |
| `[TELEFONO]` | Phone numbers | 339-1234567 → `[TELEFONO]` |
| `[EMAIL]` | Email addresses | mario@email.it → `[EMAIL]` |
| `[CARTA_CREDITO]` | Credit card numbers | 4532-1234-5678-9010 → `[CARTA_CREDITO]` |
| `[CODICE_FISCALE]` | Italian fiscal codes | RSSMRA85M01H501X → `[CODICE_FISCALE]` |
| `[DATA_NASCITA]` | Dates of birth | 15/03/1985 → `[DATA_NASCITA]` |

## Project Structure

```
flan-t5-finetune-ita/
├── main.py                          # Demo inference script (batch + interactive)
├── pyproject.toml                   # Python dependencies (uv / pip)
├── dataset/
│   ├── ita_example.json             # Sample dataset (source/target pairs)
│   └── ita_dataset/                 # Versioned training data
├── jupyter/
│   ├── localenv                     # Local paths and model settings (dotenv)
│   ├── parameters.yaml              # Hugging Face and MLflow configuration
│   ├── flan_t5_finetuning_for_anonymization.ipynb   # Main training notebook
│   ├── test_finetuned_model.ipynb   # Quick inference tests
│   ├── convert_to_onnx.ipynb        # Export checkpoint to ONNX
│   ├── tokenizer_and_data_prep.ipynb # Tokenizer and dataset exploration
│   ├── checkpoint_download.py       # Download base model from Hugging Face
│   └── libs/
│       ├── dataset.py               # CustomPIIDataset, DataPreprocessor, anonymize_text()
│       ├── utility.py               # Accelerator detection, HF download helpers
│       └── parameters.py            # YAML config loader
└── kubeflow/
    ├── main_pipeline.py             # Kubeflow pipeline definition
    ├── finetune-pipeline.yaml       # Compiled pipeline manifest
    ├── train_model.py               # Training component
    ├── convert_model.py             # ONNX conversion component
    ├── fetch_model.py               # Hugging Face model fetch
    ├── download_from_s3.py          # Dataset download from S3
    ├── upload_model.py              # Push model to registry
    └── persistent_data.py           # Dataset unpack utilities
```

Training produces a checkpoint directory (default: `jupyter/flan-finetuned-ita/`). ONNX export writes to `jupyter/flan-finetuned-ita-onnx/` by default. Both paths are gitignored.

## Installation

Requires Python 3.12+.

```bash
git clone <repository-url>
cd flan-t5-finetune-ita
uv sync
```

Or with pip:

```bash
pip install -e .
```

### Hardware Requirements

The bundled dataset is small. Expect limited generalization until you add more examples.

| Environment | Requirement |
|---|---|
| Minimum | CPU, 8 GB RAM (slower training) |
| Recommended | GPU with 8 GB+ VRAM (CUDA) or Apple Silicon (MPS) |

### Configuration

Copy or edit `jupyter/localenv` to point at your model checkpoint, dataset, and optional proxy settings:

```ini
PARAMETER_FILE = "parameters.yaml"
MODEL_REPO_ID = "google/flan-t5-small"
DATASET_FILE = "../dataset"
OUTPUT_DIR = "flan-finetuned-ita"
ONNX_OUTPUT_DIR = "flan-finetuned-ita-onnx"
```

Set your Hugging Face token in `jupyter/parameters.yaml` when downloading models from the Hub.

## Model Input/Output Format

### Input

```
anonymize: [Italian text with PII]
```

### Output

```
[Same text with PII replaced by placeholders]
```

### Examples

**Input**

```
anonymize: Mi chiamo Mario Rossi, il mio numero è 339-1234567 e abito in Via Roma 25.
```

**Output**

```
Mi chiamo [NOME], il mio numero è [TELEFONO] e abito in [INDIRIZZO].
```

**Input**

```
anonymize: Il paziente Marco Esposito, nato il 25/08/1982, codice fiscale SPSMRC82M25H501Z.
```

**Output**

```
Il paziente [NOME], nato il [DATA_NASCITA], codice fiscale [CODICE_FISCALE].
```

## Usage

### 1. Train the model (Jupyter)

From the `jupyter/` directory, open and run `flan_t5_finetuning_for_anonymization.ipynb`:

```bash
cd jupyter
uv run jupyter notebook flan_t5_finetuning_for_anonymization.ipynb
```

The notebook will:

1. Download `google/flan-t5-small` from Hugging Face
2. Load JSON datasets from `dataset/` (all `**/*.json` files)
3. Split into train/validation sets
4. Fine-tune with Hugging Face `Trainer`
5. Save the checkpoint to `OUTPUT_DIR`

Optional notebooks:

- `tokenizer_and_data_prep.ipynb` — explore tokenization before training
- `test_finetuned_model.ipynb` — smoke-test a saved checkpoint
- `convert_to_onnx.ipynb` — export for ONNX Runtime serving

### 2. Run the demo script

After training, run the interactive demo from the repository root:

```bash
uv run python main.py
```

The script loads the checkpoint from `OUTPUT_DIR` in `jupyter/localenv`, prints several anonymization examples, then accepts custom Italian text until you type `quit`.

### 3. Use the Python API

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from jupyter.libs.dataset import anonymize_text

OUTPUT_DIR = "jupyter/flan-finetuned-ita"

tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR)
model.eval()

text = "Il signor Paolo Conti abita in Via Garibaldi 45."
result = anonymize_text(text, model, tokenizer)
print(result)
# Il signor [NOME] abita in [INDIRIZZO].
```

Lower-level usage without the helper:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

text = "Contattare la dottoressa Anna Verdi all'email anna.verdi@ospedale.it"
inputs = tokenizer(f"anonymize: {text}", return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# Contattare la dottoressa [NOME] all'email [EMAIL]
```

### 4. Export to ONNX

Open `jupyter/convert_to_onnx.ipynb` and run all cells, or follow the same flow programmatically with [Optimum](https://huggingface.co/docs/optimum):

```python
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

OUTPUT_DIR = "jupyter/flan-finetuned-ita"
ONNX_DIR = "jupyter/flan-finetuned-ita-onnx"

model = ORTModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR, export=True)
model.save_pretrained(ONNX_DIR)
AutoTokenizer.from_pretrained(OUTPUT_DIR).save_pretrained(ONNX_DIR)
```

### 5. Kubeflow Pipeline (OpenShift / OCP AI)

For cluster-based training, compile and submit the pipeline defined in `kubeflow/main_pipeline.py`:

```bash
cd kubeflow
python main_pipeline.py   # compiles finetune-pipeline.yaml
```

The pipeline downloads datasets from S3, fetches the base model from Hugging Face, trains, optionally converts to ONNX, and uploads artifacts to a model registry. See [Kubernetes secrets](#kubernetes-secrets-and-s3-connections) below for required cluster configuration.

## Dataset Format

Training data is stored as JSON arrays. Each entry has two fields:

```json
{
  "source": "Mi chiamo Mario Rossi e vivo a Roma.",
  "target": "Mi chiamo [NOME] e vivo a Roma."
}
```

Add files under `dataset/` (any `**/*.json` path). The loader merges all JSON files it finds. To improve quality:

1. Add more `source`/`target` pairs with varied phrasing and PII combinations
2. Include domain-specific examples (medical, legal, financial)
3. Use real, properly consented data where available

## Customization

### Using a larger base model

In `jupyter/localenv` or the training notebook, change the model ID:

```python
model_name = "google/flan-t5-base"   # 250M parameters
# model_name = "google/flan-t5-large"  # 780M parameters
```

Larger models need more memory and training time.

### MLflow tracking

Configure the tracking server in `jupyter/parameters.yaml`:

```yaml
mlflow:
  endpoint: "http://localhost:5001"
  experiment: "flan-t5-finetune-ita"
```

## Kubernetes Secrets and S3 Connections

Notebooks and Kubeflow components expect secrets for Hugging Face and S3 storage.

### Hugging Face token

```bash
oc create secret generic huggingface-secret \
  --from-literal=HF_TOKEN=hf_api_token \
  --from-literal=HF_HOME=hf_home_path
```

### S3 connections (OCP AI)

Create dashboard-managed secrets for each bucket. Example for `s3-artifacts`:

```yaml
kind: Secret
apiVersion: v1
metadata:
  name: s3-artifacts
  namespace: flan-t5-finetune
  labels:
    opendatahub.io/dashboard: 'true'
    opendatahub.io/managed: 'true'
  annotations:
    opendatahub.io/connection-type: s3
    opendatahub.io/connection-type-ref: s3
    openshift.io/display-name: s3-artifacts
data:
  AWS_ACCESS_KEY_ID: <ACCESS_KEY_BASE64>
  AWS_DEFAULT_REGION: <REGION_BASE64>
  AWS_S3_BUCKET: <BUCKET_NAME_BASE64>
  AWS_S3_ENDPOINT: <S3_API_ENDPOINT_BASE64>
  AWS_SECRET_ACCESS_KEY: <SECRET_KEY_BASE64>
type: Opaque
```

Required connections:

| Secret | Purpose |
|---|---|
| `s3-artifacts` | Training artifacts and experiment outputs |
| `s3-models` | Model checkpoints |
| `s3-pipelines` | Kubeflow pipeline data |
| `s3-datasets` | Training datasets |

### HTTP proxy (optional)

```bash
oc create configmap network-settings \
  --from-literal=HTTP_PROXY="proxy address" \
  --from-literal=HTTPS_PROXY="proxy address" \
  --from-literal=NO_PROXY="localhost,127.0.0.1,.svc.cluster.local,kubernetes.default.svc"
```

### Notebook image dependencies

Some cluster notebook images may need extra packages at runtime:

```python
!pip install -q numpy transformers==4.57.3 python-dotenv torch datasets accelerate mlflow mlflow[kubernetes]
```

## References

- [FLAN-T5 Model](https://huggingface.co/google/flan-t5-small)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Optimum ONNX Runtime](https://huggingface.co/docs/optimum/onnxruntime/overview)
- [Italian Privacy Law (GDPR)](https://gdpr-info.eu/)
