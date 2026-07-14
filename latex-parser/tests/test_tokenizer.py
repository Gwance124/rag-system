from chunking.tokenizer import FakeTokenizer, HFTokenizer


def test_fake_tokenizer_counts_words():
    tok = FakeTokenizer()
    assert tok.count_tokens("hello world foo") == 3


def test_hf_tokenizer_delegates_to_underlying_encode(monkeypatch):
    import os

    class StubTokenizer:
        def __init__(self, tokenizer_file=None):
            assert tokenizer_file == os.path.join("/fake/model/path", "tokenizer.json")

        def encode(self, text):
            return list(range(7))  # pretend 7 tokens

    monkeypatch.setattr("transformers.PreTrainedTokenizerFast", StubTokenizer)

    tok = HFTokenizer("/fake/model/path")
    assert tok.count_tokens("anything") == 7
