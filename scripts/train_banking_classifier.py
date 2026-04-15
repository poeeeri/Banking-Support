from __future__ import annotations

import numpy as np
import pandas as pd
import argparse
import logging
import json
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from sklearn.metrics import accuracy_score, classification_report, f1_score
from pathlib import Path
from common import ID_TO_LABEL, LABEL_TO_ID, INTENT_SPECS
from datasets import Dataset
from clearml import Task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', default='data/generated/synthetic_dialogs/train.jsonl')
    parser.add_argument('--test-path', default='data/generated/synthetic_dialogs/test.jsonl')
    parser.add_argument('--model-name', default='DeepPavlov/rubert-base-cased')
    parser.add_argument('--output-dir', default='artifacts/banking_intent_classifier')
    parser.add_argument('--num-train-epochs', type=float, default=4.0)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--train-batch-size', type=int, default=8)
    parser.add_argument('--eval-batch-size', type=int, default=16)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-ratio', type=float, default=0.1)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')
    return parser.parse_args()


def load_jsonl(path):
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def extract_last_user_message(dialog):
    lines = [line.strip() for line in dialog.splitlines() if line.strip()]
    user_lines = [line for line in lines if line.lower().startswith("user:")]
    if not user_lines:
        return ""
    return user_lines[-1].split(":", 1)[1].strip()


def dialog_turns(dialog):
    turns: list[tuple[str, str]] = []
    for raw_line in dialog.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        speaker, text = line.split(":", 1)
        speaker = speaker.strip().lower()
        text = text.strip()
        if speaker in {"user", "assistant"} and text:
            turns.append((speaker, text))
    return turns


def build_last_message(dialog):
    return extract_last_user_message(dialog)


def normalize_dialog(dialog):
    turns = dialog_turns(dialog)
    return "\n".join(f"{speaker}: {text}" for speaker, text in turns)


def build_training_text(dialog):
    normalized = normalize_dialog(dialog)
    last_user_message = extract_last_user_message(normalized)
    return (
        "Ниже диалог клиента с банковской поддержкой.\n"
        "Определи интент по последнему сообщению пользователя с учетом контекста.\n\n"
        f"{normalized}\n\n"
        f"Последнее сообщение пользователя: {last_user_message}"
    )


def prepare_rows(records):
    prepared: list[dict] = []
    for row in records:
        text = build_training_text(row["dialog"])
        prepared.append(
            {
                "text": text,
                "labels": LABEL_TO_ID[row["label"]],
                "label": row["label"],
                "dialog": row["dialog"],
                "last_user_message": build_last_message(row["dialog"]),
            }
        )
    return prepared


def tokenize_batch(batch, tokenizer, max_length):
    tokenized = tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_length,
    )
    tokenized["labels"] = batch["labels"]
    return tokenized


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "weighted_f1": f1_score(labels, predictions, average="weighted"),
    }


def sanitize_metrics(metrics):
    sanitized = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, np.integer)):
            sanitized[key] = value.item()
        else:
            sanitized[key] = value
    return sanitized


def build_prediction_samples(prediction_rows):
    by_label: dict[str, list[dict]] = {}
    for row in prediction_rows:
        by_label.setdefault(row["gold_label"], []).append(row)

    samples: list[dict] = []
    for label, rows in sorted(by_label.items()):
        correct_row = next((item for item in rows if item["gold_label"] == item["predicted_label"]), None)
        wrong_row = next((item for item in rows if item["gold_label"] != item["predicted_label"]), None)
        if correct_row:
            samples.append({"sample_type": "correct", **correct_row})
        if wrong_row:
            samples.append({"sample_type": "error", **wrong_row})
    return samples


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    task = Task.init(
        project_name="Banking-Support",
        task_name="rubert-intent-classifier",
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=False,
    )
    task.connect(vars(args))

    set_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_train_records = load_jsonl(Path(args.train_path))
    raw_test_records = load_jsonl(Path(args.test_path))
    train_records = prepare_rows(raw_train_records)
    test_records = prepare_rows(raw_test_records)

    train_dataset = Dataset.from_list(train_records)
    test_dataset = Dataset.from_list(test_records)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(INTENT_SPECS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    tokenized_train = train_dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    tokenized_test = test_dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer, args.max_length),
        batched=True,
        remove_columns=test_dataset.column_names,
    )
    training_args = TrainingArguments(
        output_dir = args.output_dir,
        learning_rate = 2e-5,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="clearml",
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))

    prediction_output = trainer.predict(tokenized_test)
    predictions = np.argmax(prediction_output.predictions, axis=-1)
    references = np.array(test_dataset["labels"])

    report = classification_report(
        references,
        predictions,
        labels=list(range(len(INTENT_SPECS))),
        target_names=[spec.label for spec in INTENT_SPECS],
        output_dict=True,
        zero_division=0,
    )

    prediction_rows = []
    for row, pred_id in zip(test_records, predictions):
        prediction_rows.append(
            {
                "gold_label": row["label"],
                "predicted_label": ID_TO_LABEL[int(pred_id)],
                "last_user_message": row["last_user_message"],
                "dialog": row["dialog"],
            }
        )

    metrics_payload = {
        "train_runtime": train_result.metrics,
        "eval_metrics": sanitize_metrics(eval_metrics),
        "classification_report": report,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(prediction_rows).to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    sample_rows = build_prediction_samples(prediction_rows)
    pd.DataFrame(sample_rows).to_csv(output_dir / "prediction_samples.csv", index=False, encoding="utf-8-sig")

    logging.info("saved trained model to %s", output_dir / "best_model")
    logging.info("eval accuracy=%.4f macro_f1=%.4f", eval_metrics["eval_accuracy"], eval_metrics["eval_macro_f1"])



if __name__ == '__main__':
    main()