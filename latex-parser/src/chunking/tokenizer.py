from typing import Protocol


class Tokenizer(Protocol):
    def count_tokens(self, text: str) -> int: ...


class FakeTokenizer:
    """Whitespace-based token counter for tests, independent of any real model."""

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class HFTokenizer:
    """Wraps a real HF tokenizer loaded directly from a local tokenizer.json
    file (e.g. vLLM's model dir). Deliberately avoids AutoTokenizer/AutoConfig,
    which for some repos (e.g. ones with custom trust_remote_code model
    classes) executes unrelated model code that can require torch just to
    read config metadata - tokenizer.json alone is self-contained and needs
    none of that. Never touches the network."""

    def __init__(self, model_path: str):
        import os
        from transformers import PreTrainedTokenizerFast

        tokenizer_file = os.path.join(model_path, "tokenizer.json")
        self._tok = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)

    def count_tokens(self, text: str) -> int:
        return len(self._tok.encode(text))
