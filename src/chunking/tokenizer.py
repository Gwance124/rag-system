from typing import Protocol


class Tokenizer(Protocol):
    def count_tokens(self, text: str) -> int: ...


class FakeTokenizer:
    """Whitespace-based token counter for tests, independent of any real model."""

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class HFTokenizer:
    """Wraps a real HF tokenizer loaded from a local model directory (e.g. vLLM's
    model path). Never touches the network."""

    def __init__(self, model_path: str):
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    def count_tokens(self, text: str) -> int:
        return len(self._tok.encode(text))
