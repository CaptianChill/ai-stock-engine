"""LLM scoring agent: one headline + one ticker in, good / bad / irrelevant + confidence out.

The model is never asked for a price target, a trade, or a size. It only reads news.
Model id and prompt version are pinned and stored with every score, so a new model
produces new rows and old replays stay reproducible.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

PROMPT_VERSION = "p1"

SYSTEM = (
    "You are a financial news classifier. You judge whether a single news item is good news, "
    "bad news, or irrelevant for the stock price of one named company over the next trading day. "
    "Judge only from the text given. Reply with JSON only."
)

USER_TEMPLATE = (
    "Company ticker: {symbol}\n"
    "Headline: {headline}\n"
    "Summary: {summary}\n\n"
    "Is this GOOD, BAD, or IRRELEVANT news for {symbol}'s stock price over the next trading day? "
    'Reply exactly as JSON: {{"label": "good" | "bad" | "irrelevant", "confidence": <number 0 to 1>}}'
)

_LABEL_SCORE = {"good": 1, "bad": -1, "irrelevant": 0}


@dataclass
class Verdict:
    label: str
    score: int
    confidence: float
    parse_ok: bool


def parse_reply(text: str) -> Verdict:
    """Strict-but-forgiving parser. Anything unparseable becomes 'irrelevant' with zero
    confidence, so a malformed reply can never create a position."""
    m = re.search(r"\{.*?\}", text or "", re.S)
    if m:
        try:
            j = json.loads(m.group(0))
            label = str(j.get("label", "")).strip().lower()
            conf = float(j.get("confidence"))
            if label in _LABEL_SCORE and 0.0 <= conf <= 1.0:
                return Verdict(label, _LABEL_SCORE[label], conf, True)
        except (ValueError, TypeError):
            pass
    return Verdict("irrelevant", 0, 0.0, False)


class Scorer:
    def __init__(self, api_key: str | None, model: str, prompt_version: str = PROMPT_VERSION):
        if prompt_version != PROMPT_VERSION:
            raise ValueError(f"Prompt version {prompt_version} requested but code implements {PROMPT_VERSION}")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.prompt_version = prompt_version

    def score(self, symbol: str, headline: str, summary: str | None) -> Verdict:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=60,
            temperature=0,
            system=SYSTEM,
            messages=[{"role": "user", "content": USER_TEMPLATE.format(
                symbol=symbol, headline=headline.strip()[:500], summary=(summary or "").strip()[:800])}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return parse_reply(text)

    def score_rows(self, rows: list[dict]) -> list[dict]:
        out = []
        for r in rows:
            v = self.score(r["symbol"], r["headline"], r.get("summary"))
            out.append({"news_id": r["news_id"], "symbol": r["symbol"], "model": self.model,
                        "prompt_version": self.prompt_version, "label": v.label, "score": v.score,
                        "confidence": v.confidence})
        return out
