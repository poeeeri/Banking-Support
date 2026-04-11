from pathlib import Path
import sys
import argparse
import json


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.generate_synthetic_data.generation_config import GenerationConfig
from scripts.generate_synthetic_data.client import (
    DEFAULT_GENERATION_MODEL,
    DummyLLMClient,
    TransformersLLMClient,
)
from scripts.generate_synthetic_data.generator import SyntheticDatasetGenerator
from scripts.common import registry


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="transfer_status")
    parser.add_argument("--labels", nargs="*")
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument("--output-dir", default="data/generated/synthetic_dialogs")
    parser.add_argument("--train-per-class", type=int, default=20)
    parser.add_argument("--test-per-class", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--model-name", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--transformers", action="store_true")
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=350)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def build_llm_client(args):
    if not args.transformers:
        return DummyLLMClient()

    return TransformersLLMClient(
        model_name=args.model_name,
        use_4bit=args.use_4bit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )


def write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_dataset(generator, args):
    output_dir = Path(args.output_dir)
    labels = args.labels or list(registry.intents.keys())
    unknown_labels = [label for label in labels if label not in registry.intents]
    if unknown_labels:
        raise ValueError(f"Unknown labels: {unknown_labels}")

    train_records: list[dict] = []
    test_records: list[dict] = []

    for label in labels:
        print(f"Generating label={label} train={args.train_per_class} test={args.test_per_class}")
        train_part = generator.generate_n_for_label(
            label,
            args.train_per_class,
            max_attempts=args.max_attempts,
        )
        test_part = generator.generate_n_for_label(
            label,
            args.test_per_class,
            max_attempts=args.max_attempts,
        )
        print(f"  generated train={len(train_part)} test={len(test_part)}")
        train_records.extend(train_part)
        test_records.extend(test_part)

    examples = registry.sample_examples_by_label(
        train_records + test_records,
        seed=generator.config.seed,
    )
    metadata = {
        "model_name": args.model_name if args.transformers else "dummy",
        "labels": labels,
        "train_per_class": args.train_per_class,
        "test_per_class": args.test_per_class,
        "train_records": len(train_records),
        "test_records": len(test_records),
    }

    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "test.jsonl", test_records)
    write_json(output_dir / "examples_by_label.json", examples)
    write_json(output_dir / "metadata.json", metadata)

    print(f"Saved train: {output_dir / 'train.jsonl'}")
    print(f"Saved test: {output_dir / 'test.jsonl'}")
    print(f"Saved examples: {output_dir / 'examples_by_label.json'}")
    print(f"Saved metadata: {output_dir / 'metadata.json'}")


def main():
    args = parse_args()
    config = GenerationConfig(
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
    )
    generator = SyntheticDatasetGenerator(registry, config, build_llm_client(args))

    if args.dataset:
        generate_dataset(generator, args)
        return

    sample = generator.generate_one(args.label)
    print(sample)


if __name__ == "__main__":
    main()
