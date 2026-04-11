import json
import re


DEFAULT_GENERATION_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


class BaseLLMClient:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


# для теста
class DummyLLMClient(BaseLLMClient):
    def generate(self, prompt: str) -> str:
        return json.dumps(
            {
                "label": "transfer_status",
                "dialog": (
                    "user: Здравствуйте\n"
                    "assistant: Добрый день, чем могу помочь?\n"
                    "user: Мой перевод до сих пор не пришел, какой у него статус?"
                ),
            },
            ensure_ascii=False,
        )


class TransformersLLMClient(BaseLLMClient):
    def __init__(
        self,
        model_name: str = DEFAULT_GENERATION_MODEL,
        use_4bit: bool = False,
        max_new_tokens: int = 350,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        do_sample: bool = True,
        system_prompt: str | None = None,
        chat_template_kwargs: dict | None = None,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Install generation dependencies first: "
                "`poetry install` or `poetry install --with cuda`."
            ) from exc

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = do_sample
        self.system_prompt = system_prompt or (
            "Ты генерируешь только валидный JSON без пояснений, markdown "
            "и тройных кавычек. Ответ должен содержать только поля label и dialog."
        )
        self.chat_template_kwargs = chat_template_kwargs or {}

        quantization_config = None
        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ImportError(
                    "4-bit mode requires bitsandbytes: "
                    "`poetry install --with cuda`."
                ) from exc

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            quantization_config=quantization_config,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_inputs(self, prompt: str):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        if self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **self.chat_template_kwargs,
            )
        else:
            text = f"{self.system_prompt}\n\n{prompt}\n"

        return self.tokenizer([text], return_tensors="pt").to(self.model.device)

    @staticmethod
    def _strip_thinking(text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def generate(self, prompt: str) -> str:
        model_inputs = self._build_inputs(prompt)

        output_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = output_ids[0][model_inputs.input_ids.shape[-1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return self._strip_thinking(text)