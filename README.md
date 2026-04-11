# Banking-Support
training a model to classify requests to banking support based on the user's last message in the context of the dialogue

## Setup

```powershell
poetry env use python
poetry install
```

on a CUDA environment:

```powershell
poetry install --with cuda
```

## Synthetic data generation test

fast check:

```powershell
poetry run python -m scripts.generate_synthetic_data.run_generate
```

generation with the default model:

```powershell
poetry run python -m scripts.generate_synthetic_data.run_generate --transformers
```

default generation model:

```text
Qwen/Qwen3-4B-Instruct-2507
```

generate a small dataset for one label:

```powershell
poetry run python -m scripts.generate_synthetic_data.run_generate --transformers --dataset --labels transfer_status --train-per-class 2 --test-per-class 1
```

generate the full dataset:

```powershell
poetry run python -m scripts.generate_synthetic_data.run_generate --transformers --dataset --train-per-class 20 --test-per-class 10
```
files are saved to `data/generated/`