import ast
from datetime import date
from pathlib import Path

import pandas as pd

from engine import brain
from engine.config import ACTIVE_VERSION, StrategyVersion
from tests.synth import make_market, make_version
from tests.test_runner import FakeDB


class BrainDB(FakeDB):
    def __init__(self, bars, scores):
        super().__init__(bars, scores)
        self.reports, self.proposals, self.versions = {}, {}, {}

    def news_counts_by_day(self, start, end): return pd.Series([50.0] * 25)
    def recent_stages(self, since): return [{"stage": s, "status": v["status"], "detail": v["detail"]}
                                            for (d, s), v in self.state.items() if d >= since]
    def recent_alerts(self, since): return []
    def decision_summary(self, day): return {}
    def latest_gate_results(self, version): return {}

    def save_brain_report(self, day, facts, review, model):
        self.reports[day] = {"facts": facts, "review": review, "model": model}
        return len(self.reports)

    def insert_proposal(self, rid, p):
        i = len(self.proposals) + 1
        self.proposals[i] = {**p, "proposal_id": i, "status": "pending", "candidate_version": None}
        return i

    def get_proposal(self, i): return self.proposals.get(i)
    def decide_proposal(self, i, status, cand=None): self.proposals[i].update(status=status, candidate_version=cand)
    def register_version(self, v, engine, params): self.versions[v] = params


def test_brain_cannot_reach_order_placement():
    tree = ast.parse(Path(brain.__file__).read_text())
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | \
               {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    forbidden = {"execution", "alpaca", "runner", "risk"}
    assert not any(m.split(".")[-1] in forbidden for m in imported), imported


def test_proposal_validation():
    sv = ACTIVE_VERSION
    raw = [
        {"param": "news.min_signal", "new_value": 0.7, "rationale": "ok"},          # valid
        {"param": "news.min_signal", "new_value": 2.0, "rationale": "too high"},    # out of bounds
        {"param": "risk.max_position_pct", "new_value": 0.2, "rationale": "loosen"},  # loosening risk
        {"param": "risk.allow_shorts", "new_value": 1, "rationale": "not allowed"},  # not on the list
        {"param": "news.max_positions", "new_value": sv.news.max_positions, "rationale": "no-op"},
        {"param": "risk.daily_loss_stop_pct", "new_value": 0.02, "rationale": "tighten"},  # valid
    ]
    kept, dropped = brain.validate_proposals(raw, sv)
    assert [k["param"] for k in kept] == ["news.min_signal", "risk.daily_loss_stop_pct"]
    assert len(dropped) == 4


def test_proposals_capped_at_three():
    raw = [{"param": "news.min_signal", "new_value": v, "rationale": "x"} for v in (0.4, 0.5, 0.7, 0.8, 0.9)]
    kept, _ = brain.validate_proposals(raw, ACTIVE_VERSION)
    assert len(kept) == 3


def test_scorer_quality_detects_working_and_broken_scorer():
    bars, scores, _ = make_market(edge=0.01)
    sv = make_version()
    q = brain.scorer_quality(scores, bars, sv)
    assert q["working"] is True and q["spread_t"] > 3
    flipped = scores.assign(score=-scores["score"])
    assert brain.scorer_quality(flipped, bars, sv)["working"] is False


def test_watchdog_flags_failures_and_missing_data():
    bars, _, days = make_market(n_days=60)
    last = days[-1]
    thin = bars[~((bars["day"] == last) & (bars["symbol"] > "S10"))]
    issues = brain.data_watchdog(thin, pd.Series([50.0] * 20 + [5.0]),
                                 [{"stage": "premarket", "status": "failed", "detail": {"error": "x"}}],
                                 [], {"S30"}, last)
    titles = {i["title"] for i in issues}
    assert {"Bars missing", "No price for held stocks", "News feed thin", "Stage failed: premarket"} <= titles


def test_nightly_runs_end_to_end_and_queues_only_valid_proposals():
    bars, scores, days = make_market(n_days=120, edge=0.01)
    db = BrainDB(bars, scores)
    db.mark_stage(days[-1], "exits", "failed", {"error": "boom"})

    def fake_review(facts, sv, key, model):
        assert "scorer_quality_30d" in facts and "watchdog" in facts
        return {"summary": "ok", "findings": [], "llm": True, "proposals": [
            {"param": "news.min_signal", "new_value": 0.7, "rationale": "persistent weak picks"},
            {"param": "risk.max_position_pct", "new_value": 0.5, "rationale": "bad idea"}]}

    out = brain.nightly(db, make_version(), "paper", None, "m", "m", "p1", today=days[-1], review_fn=fake_review)
    assert out["proposals"] == 1 and out["urgent"] >= 1
    assert db.alerts and db.alerts[0][0] == "brain"
    assert db.reports[days[-1]]["review"]["dropped_proposals"]


def test_approve_creates_candidate_only():
    bars, scores, _ = make_market(n_days=60)
    db = BrainDB(bars, scores)
    pid = db.insert_proposal(1, {"param": "news.min_signal", "old_value": 0.6, "new_value": 0.75, "rationale": "x"})
    name = brain.approve(db, pid, ACTIVE_VERSION)
    assert name == f"{ACTIVE_VERSION.version}-p{pid}"
    cand = StrategyVersion.from_json(db.versions[name])
    assert cand.news.min_signal == 0.75 and cand.version == name
    assert ACTIVE_VERSION.news.min_signal == 0.6          # the live config is untouched
    assert db.proposals[pid]["status"] == "approved"


def test_review_parser_survives_garbage():
    assert brain.parse_review("not json")["llm"] is False
    j = brain.parse_review('Here: {"summary": "fine", "findings": []}')
    assert j["llm"] is True and j["proposals"] == []


def test_version_roundtrip():
    sv = ACTIVE_VERSION
    assert StrategyVersion.from_json(sv.to_json()) == sv
