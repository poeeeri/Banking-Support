from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class GenerationConfig:
    train_per_class: int = 40
    test_per_class: int = 10
    turns_min: int = 3
    turns_max: int = 6
    seed: int = 42
    assistant_style: str = 'вежливый'
    allow_typos: bool = True
    allow_smalltalk: bool = True
    allow_context_shift: bool = True
    require_json_output: bool = True