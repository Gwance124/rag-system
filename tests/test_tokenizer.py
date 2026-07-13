from chunking.tokenizer import FakeTokenizer, HFTokenizer


def test_fake_tokenizer_counts_words():
    tok = FakeTokenizer()
    assert tok.count_tokens("hello world foo") == 3


def test_hf_tokenizer_delegates_to_underlying_encode(monkeypatch):
    class StubTokenizer:
        def encode(self, text):
            return list(range(7))  # pretend 7 tokens

    def fake_from_pretrained(model_path, local_files_only=True):
        assert local_files_only is True
        return StubTokenizer()

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", fake_from_pretrained
    )

    tok = HFTokenizer("/fake/model/path")
    assert tok.count_tokens("anything") == 7
