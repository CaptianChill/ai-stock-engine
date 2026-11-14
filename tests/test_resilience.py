"""Things that went wrong on the Windows PC in November 2026."""
from __future__ import annotations

import subprocess
import sys

import pytest
import requests

from engine.alpaca import AlpacaError, _Base


class _Resp:
    status_code, content, text = 200, b"{}", "{}"

    def json(self):
        return {"ok": True}


def _client(monkeypatch, failures: int):
    monkeypatch.setattr("engine.alpaca.time.sleep", lambda s: None)
    c = _Base("k", "s")
    calls = {"n": 0}

    def request(*a, **k):
        calls["n"] += 1
        if calls["n"] <= failures:
            raise requests.ConnectionError("getaddrinfo failed")
        return _Resp()
    c.s.request = request
    return c, calls


def test_dns_failure_after_wake_is_retried(monkeypatch):
    c, calls = _client(monkeypatch, failures=3)
    assert c._req("GET", "https://paper-api.alpaca.markets/v2/calendar") == {"ok": True}
    assert calls["n"] == 4


def test_network_down_for_good_gives_a_clear_error(monkeypatch):
    c, _ = _client(monkeypatch, failures=99)
    with pytest.raises(AlpacaError, match="failed after retries"):
        c._req("GET", "https://paper-api.alpaca.markets/v2/calendar")


def test_report_prints_non_ascii_on_a_cp1252_console():
    code = ("import sys, io; sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding='cp1252');"
            "sys.argv=['engine','--help'];"
            "from engine import __main__ as m\n"
            "try: m.main(['status'])\n"
            "except SystemExit: pass\n"
            "except Exception as e:\n"
            "    if 'charmap' in str(e): raise\n"
            "print('spread t\\u2248 0.03')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert "UnicodeEncodeError" not in r.stderr, r.stderr
