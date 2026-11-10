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

    def params(self, symbol: str, headline: str, summary: str | None) -> dict:
        """The exact request for one headline. Shared by live scoring and the batch path, so a
        batch-scored headline is scored with the identical prompt."""
        return dict(
            model=self.model,
            max_tokens=60,
            # Newer models may "think" first by default; that would spend the whole 60-token
            # budget and cut the JSON off (parsed as irrelevant -> no trades). Classification
            # needs no thinking, so it is switched off explicitly.
            thinking={"type": "disabled"},
            system=SYSTEM,
            messages=[{"role": "user", "content": USER_TEMPLATE.format(
                symbol=symbol, headline=(headline or "").strip()[:500], summary=(summary or "").strip()[:800])}],
        )

    def score(self, symbol: str, headline: str, summary: str | None) -> Verdict:
        p = self.params(symbol, headline, summary)
        # Sent as extra_body so it works on any installed SDK version.
        msg = self.client.messages.create(**{k: v for k, v in p.items() if k != "thinking"},
                                          extra_body={"thinking": p["thinking"]})
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return parse_reply(text)

    def row_out(self, r: dict, v: Verdict) -> dict:
        return {"news_id": r["news_id"], "symbol": r["symbol"], "model": self.model,
                "prompt_version": self.prompt_version, "label": v.label, "score": v.score,
                "confidence": v.confidence}

    def score_rows(self, rows: list[dict]) -> list[dict]:
        return [self.row_out(r, self.score(r["symbol"], r["headline"], r.get("summary"))) for r in rows]


class BatchScorer:
    """Scores many headlines through Anthropic's Message Batches API: half the price of normal
    calls, no PC needing to stay awake, results usually within an hour (always within 24h).

    Resumable: every submitted batch and its row mapping is saved to a state folder, so a
    restart collects batches already in flight instead of paying for them again. Requests that
    error or expire are simply left unscored and get picked up by the next run.
    """

    CHUNK = 10_000

    def __init__(self, scorer: Scorer, state_dir: str, client=None, sleep=None):
        import os
        import time as _time
        self.scorer = scorer
        self.client = client or scorer.client
        self.state_dir = state_dir
        self.sleep = sleep or _time.sleep
        os.makedirs(state_dir, exist_ok=True)

    # ---- state files -------------------------------------------------------------------
    def _path(self, batch_id: str) -> str:
        import os
        return os.path.join(self.state_dir, f"{batch_id}.json")

    def pending(self) -> list[str]:
        import os
        out = []
        for f in sorted(os.listdir(self.state_dir)):
            if f.endswith(".json"):
                with open(os.path.join(self.state_dir, f)) as fh:
                    st = json.load(fh)
                if st.get("model") == self.scorer.model and st.get("prompt_version") == self.scorer.prompt_version \
                        and not st.get("collected"):
                    out.append(st["batch_id"])
        return out

    def in_flight_keys(self) -> set[tuple[str, str]]:
        keys = set()
        for bid in self.pending():
            with open(self._path(bid)) as fh:
                keys |= {(k[0], k[1]) for k in json.load(fh)["rows"].values()}
        return keys

    # ---- submit / collect --------------------------------------------------------------
    def submit(self, rows: list[dict]) -> list[str]:
        ids = []
        for i in range(0, len(rows), self.CHUNK):
            chunk = rows[i:i + self.CHUNK]
            requests, mapping = [], {}
            for j, r in enumerate(chunk):
                cid = f"r{j}"
                mapping[cid] = [str(r["news_id"]), r["symbol"]]
                requests.append({"custom_id": cid,
                                 "params": self.scorer.params(r["symbol"], r["headline"], r.get("summary"))})
            b = self.client.messages.batches.create(requests=requests)
            with open(self._path(b.id), "w") as fh:
                json.dump({"batch_id": b.id, "model": self.scorer.model,
                           "prompt_version": self.scorer.prompt_version, "rows": mapping,
                           "collected": False}, fh)
            ids.append(b.id)
            print(f"submitted batch {b.id} ({len(chunk)} headlines)")
        return ids

    def wait(self, batch_ids: list[str], poll_seconds: int = 60) -> None:
        left = set(batch_ids)
        while left:
            for bid in sorted(left):
                b = self.client.messages.batches.retrieve(bid)
                if b.processing_status == "ended":
                    left.discard(bid)
                else:
                    c = b.request_counts
                    print(f"batch {bid}: {c.succeeded} done, {c.processing} processing", flush=True)
            if left:
                self.sleep(poll_seconds)

    def collect(self, batch_id: str) -> list[dict]:
        with open(self._path(batch_id)) as fh:
            st = json.load(fh)
        out = []
        for res in self.client.messages.batches.results(batch_id):
            key = st["rows"].get(res.custom_id)
            if key is None or res.result.type != "succeeded":
                continue  # errored / expired / canceled: stays unscored, next run retries it
            text = "".join(b.text for b in res.result.message.content if getattr(b, "type", "") == "text")
            out.append(self.scorer.row_out({"news_id": key[0], "symbol": key[1]}, parse_reply(text)))
        return out

    def mark_collected(self, batch_id: str) -> None:
        with open(self._path(batch_id)) as fh:
            st = json.load(fh)
        st["collected"] = True
        with open(self._path(batch_id), "w") as fh:
            json.dump(st, fh)

    def run(self, rows: list[dict], insert, poll_seconds: int = 60) -> int:
        """Submit what isn't already in flight, wait, store results. Returns rows stored."""
        busy = self.in_flight_keys()
        fresh = [r for r in rows if (str(r["news_id"]), r["symbol"]) not in busy]
        if busy:
            print(f"resuming {len(self.pending())} batch(es) already submitted")
        ids = self.pending() + (self.submit(fresh) if fresh else [])
        self.wait(ids, poll_seconds)
        stored = 0
        for bid in ids:
            scored = self.collect(bid)
            insert(scored)
            self.mark_collected(bid)
            stored += len(scored)
            print(f"batch {bid}: stored {len(scored)} scores")
        return stored
