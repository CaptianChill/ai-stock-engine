from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from engine import risk as rk
from engine import strategies as st
from engine.audit import stats
from engine.audit.gates import (ALL_GATES, gate_independent, gate_no_lookahead, gate_overfitting,
                                run_replay_gates)
from engine.audit.independent import reproduce
from engine.config import LIVE_CONFIRM_PHRASE, NewsParams, RiskParams, Settings
from engine.execution import client_order_id, live_blockers
from engine.llm_scorer import parse_reply
from engine.replay import metrics, run_replay
from tests.synth import make_market, make_version

NY = ZoneInfo("America/New_York")


@pytest.fixture(scope="module")
def market():
    return make_market()


# -- timing / look-ahead ---------------------------------------------------------
def test_news_assigned_to_correct_trade_day():
    days = [date(2026, 10, 29), date(2026, 10, 30), date(2026, 11, 2)]  # Thu, Fri, Mon
    ts = pd.Series(pd.to_datetime([
        datetime(2026, 10, 29, 8, 0, tzinfo=NY),    # premarket Thu -> Thu
        datetime(2026, 10, 29, 9, 20, tzinfo=NY),   # after 09:15 cutoff, market hours -> excluded
        datetime(2026, 10, 29, 12, 0, tzinfo=NY),   # intraday -> excluded
        datetime(2026, 10, 29, 17, 0, tzinfo=NY),   # after close Thu -> Fri
        datetime(2026, 10, 31, 10, 0, tzinfo=NY),   # Saturday -> Mon
        datetime(2026, 11, 2, 16, 30, tzinfo=NY),   # after last known day -> excluded
    ], utc=True))
    out = st.assign_trade_day(ts, days, NewsParams()).tolist()
    assert out == [date(2026, 10, 29), None, None, date(2026, 10, 30), date(2026, 11, 2), None]


def test_eligibility_uses_only_prior_days(market):
    bars, _, days = market
    elig = st.eligible_universe(bars, 40, 1e6, 1.0)
    assert elig[days[19]] == set()        # needs 20 prior days
    assert len(elig[days[20]]) == 40


def test_lookahead_gate_rejects_pre_cutoff_period(market):
    bars, scores, days = market
    sv = make_version()
    res = run_replay(bars, scores, sv)
    g = gate_no_lookahead(res, scores, sv, period_start=days[0], llm_cutoff=days[10])
    assert not g.passed
    g2 = gate_no_lookahead(res, scores, sv, period_start=days[0], llm_cutoff=date(2020, 1, 1))
    assert g2.passed and g2.detail["timing"]["violations"] == 0


# -- replay correctness ------------------------------------------------------------
def test_planted_signal_is_found_and_noise_is_not():
    bars, scores, _ = make_market(edge=0.01)
    sv = make_version(news_weight=1.0, momentum_weight=0.0)
    good = metrics(run_replay(bars, scores, sv).daily)
    bars0, scores0, _ = make_market(edge=0.0, seed=11)
    flat = metrics(run_replay(bars0, scores0, sv).daily)
    assert good["sharpe"] > 3
    assert abs(flat["sharpe"]) < good["sharpe"] / 3


def test_costs_reduce_returns_monotonically(market):
    bars, scores, _ = market
    sv = make_version(news_weight=1.0, momentum_weight=0.0)
    r = [metrics(run_replay(bars, scores, sv, cost_bps=c).daily)["total_return"] for c in (0, 5, 20, 50)]
    assert r == sorted(r, reverse=True)


def test_news_gross_exposure_never_exceeds_100pct(market):
    bars, scores, _ = market
    sv = make_version(news_weight=1.0, momentum_weight=0.0)
    res = run_replay(bars, scores, sv)
    out = reproduce(bars.to_dict("records"), [t for t in res.trades if t["engine"] == "news"],
                    sv.costs.cost_bps_per_side, sv.news.hold_days, sv.news.min_dollar_volume, sv.risk.min_price)
    assert out["n_problems"] == 0, out["problems"][:3]


def test_independent_reproduction_catches_tampering(market):
    bars, scores, _ = market
    sv = make_version()
    res = run_replay(bars, scores, sv)
    assert gate_independent(bars, res, sv).passed
    res.trades[0] = {**res.trades[0], "net": res.trades[0]["net"] + 0.01}
    assert not gate_independent(bars, res, sv).passed


def test_trimmed_window_only_counts_trades_inside(market):
    bars, scores, days = market
    sv = make_version()
    res = run_replay(bars, scores, sv, start=days[200], end=days[300])
    assert all(days[200] <= t["entry_day"] <= days[300] for t in res.trades)
    assert res.daily.index.min() == days[200] and res.daily.index.max() == days[300]


def test_momentum_picks_trending_names_and_veto_drops_bad_news():
    bars, scores, days = make_market(edge=0.0, trend_symbols=5, seed=3)
    sv = make_version()
    elig = st.eligible_universe(bars, 40, 1e6, 1.0)
    targets = st.momentum_targets(bars, None, elig, sv.momentum)
    last = targets[max(targets)]
    assert set(last) == {f"S{i:02d}" for i in range(5)}
    d = max(targets)
    sig = pd.DataFrame([{"trade_day": d, "symbol": "S00", "signal": -0.9, "n_items": 1, "last_published": None}])
    vetoed = st.momentum_targets(bars, sig, elig, sv.momentum)[d]
    assert "S00" not in vetoed


# -- full gate run -----------------------------------------------------------------
def test_gates_pass_on_strong_planted_signal_and_fail_on_noise():
    sv = make_version(news_weight=1.0, momentum_weight=0.0)
    bars, scores, days = make_market(edge=0.01)
    _, gates = run_replay_gates(bars, scores, sv, days[30], date(2020, 1, 1), n_trials=3, period_end=days[-1])
    by = {g.gate: g for g in gates}
    assert all(g.passed for g in gates), {k: v.detail for k, v in by.items() if not v.passed}

    bars0, scores0, days0 = make_market(edge=0.0, seed=21)
    _, gates0 = run_replay_gates(bars0, scores0, sv, days0[30], date(2020, 1, 1), n_trials=3, period_end=days0[-1])
    assert not all(g.passed for g in gates0)


# -- overfitting statistics ---------------------------------------------------------
def test_deflated_sharpe_penalizes_many_trials():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0008, 0.01, 500)
    one = stats.deflated_sharpe(r, 1)["dsr"]
    many = stats.deflated_sharpe(r, 1000)["dsr"]
    assert many < one


def test_pbo_high_for_noise_low_for_real_edge():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, (1000, 20))
    assert stats.pbo_cscv(noise)["pbo"] > 0.3
    real = noise.copy()
    real[:, 0] += 0.003
    assert stats.pbo_cscv(real)["pbo"] < 0.1


# -- risk agent -------------------------------------------------------------------
P = RiskParams()


def _buy(notional=5000, price=50.0):
    return rk.OrderIntent("news", "AAA", "buy", price, target_notional=notional)


def test_risk_blocks_entries_but_allows_exits_under_kill_switch():
    v = rk.check(_buy(), equity=100_000, gross_exposure=0, kill_switch_engaged=True, entries_halted=False,
                 held_symbols={}, p=P)
    assert not v.approved
    s = rk.OrderIntent("news", "AAA", "sell", 50, qty=10)
    v = rk.check(s, equity=100_000, gross_exposure=0, kill_switch_engaged=True, entries_halted=True,
                 held_symbols={"AAA": 10}, p=P)
    assert v.approved and v.qty == 10


def test_risk_caps_position_and_gross():
    v = rk.check(_buy(notional=50_000), equity=100_000, gross_exposure=0, kill_switch_engaged=False,
                 entries_halted=False, held_symbols={}, p=P)
    assert v.qty * 50 <= 10_000
    v = rk.check(_buy(), equity=100_000, gross_exposure=99_990, kill_switch_engaged=False,
                 entries_halted=False, held_symbols={}, p=P)
    assert not v.approved


def test_risk_rejects_shorts_and_penny_stocks():
    s = rk.OrderIntent("news", "AAA", "sell", 50, qty=10)
    assert not rk.check(s, equity=1e5, gross_exposure=0, kill_switch_engaged=False, entries_halted=False,
                        held_symbols={}, p=P).approved
    assert not rk.check(_buy(price=2.0), equity=1e5, gross_exposure=0, kill_switch_engaged=False,
                        entries_halted=False, held_symbols={}, p=P).approved


# -- live lock ----------------------------------------------------------------------
def _settings(mode="paper", confirm=None):
    return Settings(None, None, None, None, "m", "p1", date(2026, 7, 1), mode, confirm)


def test_live_lock():
    all_pass = {g: True for g in ALL_GATES}
    assert live_blockers(_settings("paper"), all_pass)
    assert live_blockers(_settings("live", "wrong"), all_pass)
    one_fail = {**all_pass, "paper_match": False}
    assert live_blockers(_settings("live", LIVE_CONFIRM_PHRASE), one_fail) == ["gate 'paper_match' has not passed"]
    assert live_blockers(_settings("live", LIVE_CONFIRM_PHRASE), all_pass) == []


def test_client_order_ids_are_deterministic():
    a = client_order_id(date(2026, 11, 2), "news", "AAPL", "buy", "paper")
    assert a == client_order_id(date(2026, 11, 2), "news", "AAPL", "buy", "paper")
    assert a != client_order_id(date(2026, 11, 2), "news", "AAPL", "buy", "live")


# -- LLM reply parsing --------------------------------------------------------------
@pytest.mark.parametrize("text,label,conf,ok", [
    ('{"label": "good", "confidence": 0.8}', "good", 0.8, True),
    ('Sure! {"label":"BAD","confidence":0.6} hope that helps', "bad", 0.6, True),
    ('{"label": "great", "confidence": 0.9}', "irrelevant", 0.0, False),
    ('{"label": "good", "confidence": 1.7}', "irrelevant", 0.0, False),
    ("no json here", "irrelevant", 0.0, False),
])
def test_parse_reply(text, label, conf, ok):
    v = parse_reply(text)
    assert (v.label, v.confidence, v.parse_ok) == (label, conf, ok)


def test_risk_cap_counts_shares_already_held_by_other_engine():
    # 10% of 100k = 10k cap; 150 shares @ $50 = 7.5k already held -> at most 2.5k more (50 shares)
    v = rk.check(_buy(notional=10_000), equity=100_000, gross_exposure=7_500, kill_switch_engaged=False,
                 entries_halted=False, held_symbols={"AAA": 150}, p=P)
    assert v.approved and v.qty == 50
    full = rk.check(_buy(notional=10_000), equity=100_000, gross_exposure=10_000, kill_switch_engaged=False,
                    entries_halted=False, held_symbols={"AAA": 200}, p=P)
    assert not full.approved


# -- long-history momentum gate ---------------------------------------------------------
from engine.audit.gates import gate_momentum_history, momentum_only


def test_momentum_history_gate_passes_real_trend_fails_noise_and_missing_data():
    sv = make_version()
    long_trend, _, _ = make_market(n_days=1700, trend_symbols=5, edge=0.0, news_per_day=1, seed=9)
    g = gate_momentum_history(long_trend, sv, n_trials=3)
    assert g.passed, g.detail["checks"]
    assert g.detail["rebalances"] >= 60
    long_noise, _, _ = make_market(n_days=1700, trend_symbols=0, edge=0.0, news_per_day=1, seed=10)
    assert not gate_momentum_history(long_noise, sv, n_trials=3).passed
    assert not gate_momentum_history(None, sv, n_trials=3).passed
    assert gate_momentum_history(None, make_version(momentum_weight=0.0, news_weight=1.0), 3).passed


def test_momentum_only_drops_news_and_veto():
    mv = momentum_only(make_version())
    assert mv.news_weight == 0 and mv.momentum_weight == 1 and mv.momentum.llm_veto is False
