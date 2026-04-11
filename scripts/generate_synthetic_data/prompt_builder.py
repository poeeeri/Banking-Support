import random
from scripts.generate_synthetic_data.generation_config import GenerationConfig
from scripts.common import IntentRegProc


class PromptBuilder:
    def __init__(self, registry: IntentRegProc, config: GenerationConfig):
        self.registry = registry
        self.config = config
        self.rng = random.Random(config.seed)

    def build_generation_prompt(self, label: str) -> str:
        intent = self.registry.get_intent_spec(label)

        turns_count = self.rng.randint(self.config.turns_min, self.config.turns_max)
        assistant_style = self.config.assistant_style

        typo_rule = (
            "Допускаются редкие естественные опечатки или разговорные сокращения."
            if self.config.allow_typos
            else "Не используй опечатки."
        )

        smalltalk_rule = (
            "Можно добавить короткое вступление или уточнение перед основным вопросом."
            if self.config.allow_smalltalk
            else "Не добавляй лишний small talk."
        )

        context_shift_rule = (
            "Контекст должен помогать понять последний вопрос, но не дублировать его дословно."
            if self.config.allow_context_shift
            else "Контекст должен быть прямым и без отвлечений."
        )

        examples_text = "\n".join(f"- {ex}" for ex in intent.examples)

        return f"""
            Сгенерируй реалистичный диалог клиента с банковской поддержкой на русском языке.

            Целевой интент последнего сообщения пользователя:
            {intent.label}

            Описание интента:
            {intent.description}

            Какой тип ответа ожидается от поддержки:
            {intent.response_hint}

            Типичные формулировки:
            {examples_text}

            Требования:
            1. Диалог должен быть про банковские услуги.
            2. Последнее сообщение пользователя должно однозначно соответствовать интенту "{intent.label}".
            3. Контекст диалога должен помогать понять последний вопрос.
            4. Диалог должен быть естественным, не шаблонным.
            5. Используй стиль оператора: {assistant_style}.
            6. Количество реплик: примерно {turns_count}.
            7. {typo_rule}
            8. {smalltalk_rule}
            9. {context_shift_rule}

            Строго соблюдай формат:
            - реплики только в виде "user: ..." и "assistant: ..."
            - последняя реплика обязательно от user
            - верни только JSON
            - JSON должен содержать поля:
            - "label"
            - "dialog"

            Пример формата:
            {{
                "label": "{intent.label}",
                "dialog": "user: ...\\nassistant: ...\\nuser: ..."
            }}
            """.strip()