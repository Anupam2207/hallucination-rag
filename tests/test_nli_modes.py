from src.detection.nli_verifier import NLIVerifier


def test_nli_disabled_mode_returns_not_run():
    verifier = NLIVerifier(enabled=False)
    result = verifier.verify("RAG combines retrieval and generation.", "RAG combines retrieval and generation.")
    assert result["label"] == "not_run"
    assert result["enabled"] is False


def test_nli_enabled_but_unavailable_falls_back_without_crashing(monkeypatch):
    def fake_load(self):
        self.available = False
        self.model = None
        self.error = "model unavailable for test"

    monkeypatch.setattr(NLIVerifier, "_load_model", fake_load)
    verifier = NLIVerifier(enabled=True)
    result = verifier.verify("RAG combines retrieval and generation.", "RAG combines retrieval and generation.")
    assert result["label"] == "unavailable"
    assert result["available"] is False
    assert "unavailable" in result["error"]
