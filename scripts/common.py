from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import re
import random


# структура одного запроса
@dataclass(frozen=True)
class Intent:
    label: str
    title: str
    description: str
    response_hint: str
    examples: list[str]


# класс для импорта общих интентов из json файла в словарь и определяем методы для работы с ними
class IntentRegProc:
    def __init__(self, json_path):
        self.json_path = Path(json_path)
        self.intents: dict[str, Intent] = {}
        self._load()


    def _load(self):
        with open(self.json_path, "r", encoding='utf-8') as f:
            data = json.load(f)
        for intents_data in data['intents']:
            intent = Intent(**intents_data)
            self.intents[intent.label] = intent


    def get_intent_spec(self, label: str) -> Intent:
        return self.intents[label]
    

    def extract_last_user_message(self, dialog_text):
        lines = [line.strip() for line in dialog_text.splitlines() if line.strip()]
        user_lines = [line for line in lines if line.lower().startswith('user:')]
        if not user_lines:
            return ''
        return user_lines[-1].split(':', 1)[1].strip()
    

    def dialog_turns(self, dialog):
        turns: list[tuple[str, str]] = []
        for raw_line in dialog.splitlines():
            line = raw_line.strip()
            if not line or ':' not in line:
                continue
            role, text = line.split(":", 1)
            role = role.strip().lower()
            text = text.strip()
            if role in {"user", "assistant"} and text:
                turns.append((role, text))
        return turns


    def normalize_dialog(self, dialog):
        turns = self.dialog_turns(dialog)
        return "\n".join(f"{role}: {text}" for role, text in turns)


    def validate_record(self, record, expected_label):
        label = str(record.get("label", "")).strip()
        dialog = self.normalize_dialog(str(record.get("dialog", "")).strip())
        if expected_label and label != expected_label:
            return False
        if label not in self.intents:
            return False
        turns = self.dialog_turns(dialog)
        if len(turns) < 3:
            return False
        if turns[-1][0] != "user":
            return False
        last_user_message = self.extract_last_user_message(dialog)
        if len(last_user_message) < 8:
            return False
        record["dialog"] = dialog
        return True
    

    def build_training_text(self, dialog):
        normalized = self.normalize_dialog(dialog)
        last_user_message = self.extract_last_user_message(normalized)
        return (
            "Ниже диалог клиента с банковской поддержкой.\n"
            "Определи интент по последнему сообщению пользователя с учетом контекста.\n\n"
            f"{normalized}\n\n"
            f"Последнее сообщение пользователя: {last_user_message}"
        )


    def safe_slug(self, text):
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip())
        return slug.strip("_") or "artifact"


    def sample_examples_by_label(self, records, seed = 42):
        rng = random.Random(seed)
        grouped: dict[str, list[dict]] = {}
        for record in records:
            grouped.setdefault(record["label"], []).append(record)
        result: dict[str, dict] = {}
        for label, items in grouped.items():
            result[label] = rng.choice(items)
        return result


# определяем справочник интентов
registry = IntentRegProc("structures/intent.json")
INTENT_SPECS = registry.intents
LABEL_TO_ID = {label: idx for idx, label in enumerate(INTENT_SPECS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}