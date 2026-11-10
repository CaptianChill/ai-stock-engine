"""Batch scoring path, with a fake Anthropic client (no network, no cost)."""
from types import SimpleNamespace as NS

from engine.llm_scorer import BatchScorer, Scorer


class FakeBatches:
    def __init__(self):
        self.created, self.polls = {}, 0

    def create(self, requests):
        bid = f"b{len(self.created)}"
        self.created[bid] = requests
        return NS(id=bid)

    def retrieve(self, bid):
        self.polls += 1
        done = self.polls > 1
        return NS(processing_status="ended" if done else "in_progress",
                  request_counts=NS(succeeded=0, processing=len(self.created[bid])))

    def results(self, bid):
        for r in self.created[bid]:
            if "FAIL" in r["params"]["messages"][0]["content"]:
                yield NS(custom_id=r["custom_id"], result=NS(type="errored"))
                continue
            msg = NS(content=[NS(type="text", text='{"label": "good", "confidence": 0.8}')])
            yield NS(custom_id=r["custom_id"], result=NS(type="succeeded", message=msg))


def _scorer(fake):
    s = Scorer.__new__(Scorer)
    s.client, s.model, s.prompt_version = NS(messages=NS(batches=fake)), "m", "p1"
    return s


ROWS = [{"news_id": "1", "symbol": "AAPL", "headline": "beats", "summary": None},
        {"news_id": "2", "symbol": "BRK.B", "headline": "buyback", "summary": "x"},
        {"news_id": "3", "symbol": "MSFT", "headline": "FAIL", "summary": None}]


def test_batch_scores_and_skips_errors(tmp_path):
    fake = FakeBatches()
    stored = []
    bs = BatchScorer(_scorer(fake), str(tmp_path), sleep=lambda s: None)
    n = bs.run(ROWS, stored.extend)
    assert n == 2
    assert {(r["news_id"], r["symbol"]) for r in stored} == {("1", "AAPL"), ("2", "BRK.B")}
    assert all(r["label"] == "good" and r["score"] == 1 and r["model"] == "m" for r in stored)
    assert bs.pending() == []


def test_same_prompt_as_live_scoring():
    fake = FakeBatches()
    sc = _scorer(fake)
    BatchScorer(sc, "/tmp/_bs_prompt_test", sleep=lambda s: None).submit(ROWS[:1])
    sent = next(iter(fake.created.values()))[0]["params"]
    assert sent == sc.params("AAPL", "beats", None)


def test_resume_does_not_resubmit(tmp_path):
    fake = FakeBatches()
    bs = BatchScorer(_scorer(fake), str(tmp_path), sleep=lambda s: None)
    bs.submit(ROWS)                       # crash after submit, before collecting
    stored = []
    bs.run(ROWS, stored.extend)           # restart with the same unscored rows
    assert len(fake.created) == 1         # nothing paid for twice
    assert len(stored) == 2


def test_thinking_disabled_so_reply_fits():
    p = _scorer(FakeBatches()).params("AAPL", "x", None)
    assert p["thinking"] == {"type": "disabled"} and p["max_tokens"] == 60
