from scripts.generate_synthetic_data.prompt_builder import PromptBuilder
from scripts.generate_synthetic_data.parser import GenerationParser


class SyntheticDatasetGenerator:
    def __init__(self, registry, config, llm_client):
        self.registry = registry
        self.config = config
        self.prompt_builder = PromptBuilder(registry, config)
        self.llm_client = llm_client

    def generate_one(self, label):
        prompt = self.prompt_builder.build_generation_prompt(label)

        try:
            raw_output = self.llm_client.generate(prompt)
            record = GenerationParser.parse_record(raw_output)

            is_valid = self.registry.validate_record(record, expected_label=label)
            if not is_valid:
                return None

            return record

        except Exception:
            return None
        
    
    def generate_n_for_label(self, label, n, max_attempts = 5):
        results: list[dict] = []
        attempts = 0

        while len(results) < n and attempts < n * max_attempts:
            attempts += 1
            record = self.generate_one(label)
            if record is not None:
                results.append(record)

        return results
    

    def generate_dataset_split(self, labels = None, max_attempts = 5):
        train_records: list[dict] = []
        test_records: list[dict] = []

        labels = labels or self.registry.intents.keys()
        for label in labels:
            train_records.extend(
                self.generate_n_for_label(
                    label,
                    self.config.train_per_class,
                    max_attempts=max_attempts,
                )
            )
            test_records.extend(
                self.generate_n_for_label(
                    label,
                    self.config.test_per_class,
                    max_attempts=max_attempts,
                )
            )

        return train_records, test_records
