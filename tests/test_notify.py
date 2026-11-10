from engine.notify import brain_message, notify, stage_message


def test_no_url_sends_nothing(monkeypatch):
    monkeypatch.delenv("NOTIFY_URL", raising=False)
    assert notify("t", "m") is False


def test_send_failure_never_raises(monkeypatch):
    monkeypatch.setenv("NOTIFY_URL", "http://127.0.0.1:9/nothing-listens-here")
    assert notify("t", "m") is False


def test_messages():
    t, b, p = stage_message("premarket", {"news_picks": ["AAPL"], "orders_submitted": 1}, "paper")
    assert "PAPER" in t and "AAPL" in b and p == "default"
    t, b, p = stage_message("close", {"equity": 99000, "pnl_day": -1000, "kill_switch_engaged": True}, "paper")
    assert t.startswith("🔴") and "KILL SWITCH" in b and p == "urgent"
    t, b, p = brain_message({"summary": "ok", "findings": [{"severity": "high", "title": "x"}]})
    assert "problems" in t and p == "high"
