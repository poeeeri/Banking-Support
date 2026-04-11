import re
import json


class GenerationParser:
    @staticmethod
    def extract_json_block(text: str) -> str:
        text = text.strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text).strip()
            text = re.sub(r"\s*```$", "", text).strip()

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return match.group(0).strip()

        return text

    @staticmethod
    def parse_record(text: str) -> dict:
        cleaned = GenerationParser.extract_json_block(text)
        return json.loads(cleaned)