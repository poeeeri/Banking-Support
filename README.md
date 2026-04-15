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

## Train classifier

the classifier is trained with `scripts/train_banking_classifier.py`.
by default it uses the generated dataset from `data/generated/synthetic_dialogs/`
and saves artifacts to `artifacts/banking_intent_classifier/`.

The script logs training to ClearML. Configure ClearML before training:

```powershell
poetry run clearml-init
```

test on a small dataset:

```powershell
poetry run python scripts/train_banking_classifier.py --train-path data/generated/smoke_test/train.jsonl --test-path data/generated/smoke_test/test.jsonl --output-dir artifacts/smoke_classifier
```

Train on the full generated dataset:

```powershell
poetry run python scripts/train_banking_classifier.py --train-path data/generated/synthetic_dialogs/train.jsonl --test-path data/generated/synthetic_dialogs/test.jsonl --model-name DeepPavlov/rubert-base-cased --output-dir artifacts/banking_intent_classifier --num-train-epochs 4 --train-batch-size 8 --eval-batch-size 16
```

Classifier arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `--train-path` | `data/generated/synthetic_dialogs/train.jsonl` | Path to the training JSONL dataset. |
| `--test-path` | `data/generated/synthetic_dialogs/test.jsonl` | Path to the test JSONL dataset. |
| `--model-name` | `DeepPavlov/rubert-base-cased` | Hugging Face model name or local model path. |
| `--output-dir` | `artifacts/banking_intent_classifier` | Directory for checkpoints, the best model, metrics, and prediction files. |
| `--num-train-epochs` | `4.0` | Number of training epochs. |
| `--lr` | `2e-5` | Learning rate. |
| `--train-batch-size` | `8` | Per-device training batch size. |
| `--eval-batch-size` | `16` | Per-device evaluation batch size. |
| `--weight-decay` | `0.01` | Weight decay used by the trainer. |
| `--warmup-ratio` | `0.1` | Warmup ratio for the learning rate schedule. |
| `--max-length` | `256` | Maximum tokenized sequence length. Longer examples are truncated. |
| `--seed` | `42` | Random seed for reproducible training. |
| `--fp16` | disabled | Enable mixed precision training. Use only with supported CUDA hardware. |