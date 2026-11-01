"""Command line: python -m engine <command> ...

  init-db                         apply schema.sql
  fetch-history START END         daily bars (from ~13 months before START) + news for START..END
  score-news START END [--max-calls N]   LLM-score news for symbols in the tradeable universe
  replay START END                run the replay with costs, store it, print metrics
  audit RUN_ID                    run every replay gate on a stored run and record results
  paper-gates                     run the paper-trading gates and record results
  test-kill-switch                prove the kill switch and loss stop block entries
  tick [--stage S]                scheduler entry point (paper unless unlocked for live)
  status                          gates, kill switch, open positions, latest equity
  kill "reason" / release         engage or release the kill switch
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from .config import ACTIVE_VERSION, load_settings

NY = ZoneInfo("America/New_York")


def _db(settings):
    from .db import DB
    return DB(settings.database_url)


def _history_start(start: date) -> date:
    return start - timedelta(days=int(ACTIVE_VERSION.momentum.lookback_days * 1.6) + 30)


def cmd_init_db(a, s):
    _db(s).init_schema()
    print("schema applied")


def cmd_fetch_history(a, s):
    from .alpaca import DataClient, TradingClient
    from .runner import EXCHANGES
    db = _db(s)
    tc = TradingClient(s.alpaca_key, s.alpaca_secret, live=False)
    dc = DataClient(s.alpaca_key, s.alpaca_secret)
    syms = sorted(x["symbol"] for x in tc.assets()
                  if x.get("tradable") and x.get("exchange") in EXCHANGES and "." not in x["symbol"])
    print(f"{len(syms)} symbols; fetching bars from {_history_start(a.start)} to {a.end}")
    n = db.upsert_bars(dc.daily_bars(syms, _history_start(a.start), a.end))
    print(f"bars stored: {n}")
    since = datetime.combine(a.start - timedelta(days=14), time(0), NY)
    until = datetime.combine(a.end, time(23, 59), NY)
    n = 0
    batch = []
    for item in dc.news(since, until):
        batch.append(item)
        if len(batch) >= 1000:
            n += db.upsert_news(batch)
            batch = []
            print(f"news stored: {n}", end="\r")
    n += db.upsert_news(batch)
    print(f"\nnews stored: {n}")


def _universe_union(bars, sv, from_day: date) -> set[str]:
    """Every symbol eligible on any day from `from_day` on: only these get LLM calls."""
    from . import strategies as st
    u: set[str] = set()
    for p in (sv.news, sv.momentum):
        for d, names in st.eligible_universe(bars, p.universe_size, p.min_dollar_volume, sv.risk.min_price).items():
            if d >= from_day:
                u |= names
    return u


def cmd_score_news(a, s):
    from .llm_scorer import Scorer
    db = _db(s)
    start = max(a.start, s.llm_knowledge_cutoff + timedelta(days=1))
    bars = db.load_bars(_history_start(start), a.end)
    universe = _universe_union(bars, ACTIVE_VERSION, start - timedelta(days=14))
    since = datetime.combine(start - timedelta(days=14), time(0), NY)
    until = datetime.combine(a.end, time(23, 59), NY)
    rows = db.unscored_news(since, until, s.llm_model, s.llm_prompt_version, universe)
    print(f"{len(rows)} (headline, symbol) pairs to score with {s.llm_model}")
    if a.max_calls is not None:
        rows = rows[: a.max_calls]
    scorer = Scorer(s.anthropic_key, s.llm_model, s.llm_prompt_version)
    for i in range(0, len(rows), 100):
        db.insert_scores(scorer.score_rows(rows[i:i + 100]))
        print(f"scored {min(i + 100, len(rows))}/{len(rows)}", end="\r")
    print()


def _load_inputs(db, s, start: date, end: date):
    bars = db.load_bars(_history_start(start), end)
    since = datetime.combine(max(start, s.llm_knowledge_cutoff + timedelta(days=1)) - timedelta(days=5), time(0), NY)
    since = max(since, datetime.combine(s.llm_knowledge_cutoff + timedelta(days=1), time(0), NY))
    until = datetime.combine(end, time(23, 59), NY)
    scores = db.load_scores(since, until, s.llm_model, s.llm_prompt_version)
    return bars, scores


def cmd_replay(a, s):
    from .replay import metrics, run_replay
    db = _db(s)
    sv = ACTIVE_VERSION
    db.register_version(sv.version, "combined", sv.to_json())
    bars, scores = _load_inputs(db, s, a.start, a.end)
    res = run_replay(bars, scores, sv, start=a.start, end=a.end)
    m = metrics(res.daily, res.trades)
    m["news"] = metrics(res.news_daily, [t for t in res.trades if t["engine"] == "news"])
    m["momentum"] = metrics(res.momentum_daily, [t for t in res.trades if t["engine"] == "momentum"])
    run_id = db.save_replay(sv.version, a.start, a.end, sv.costs.cost_bps_per_side, m, res.daily, res.trades)
    print(json.dumps({"run_id": run_id, **m}, indent=2, default=str))


def cmd_audit(a, s):
    from .audit.gates import run_replay_gates
    db = _db(s)
    sv = ACTIVE_VERSION
    run = db.load_replay(a.run_id)
    if run is None:
        sys.exit(f"no replay run {a.run_id}")
    if run["version"] != sv.version:
        sys.exit(f"run {a.run_id} is version {run['version']}, active version is {sv.version}")
    start, end = run["period_start"], run["period_end"]
    bars, scores = _load_inputs(db, s, start, end)
    variants = None
    others = db.replays_for_period(start, end)
    if len({o["version"] for o in others}) >= 2:
        latest = {}
        for o in others:
            latest[o["version"]] = o["daily_returns"]
        variants = pd.DataFrame(latest).sort_index().fillna(0.0)
    _, gates = run_replay_gates(bars, scores, sv, start, s.llm_knowledge_cutoff,
                                n_trials=db.count_versions(), variants=variants, period_end=end)
    for g in gates:
        db.save_audit(sv.version, a.run_id, g.gate, g.passed, g.detail)
        print(f"{'PASS' if g.passed else 'FAIL'}  {g.gate}")
    print(f"\n{sum(g.passed for g in gates)}/{len(gates)} replay gates passed")


def cmd_paper_gates(a, s):
    from .audit.gates import run_paper_gates
    db = _db(s)
    for g in run_paper_gates(db, ACTIVE_VERSION):
        db.save_audit(ACTIVE_VERSION.version, None, g.gate, g.passed, g.detail)
        print(f"{'PASS' if g.passed else 'FAIL'}  {g.gate}  {json.dumps(g.detail, default=str)}")


def cmd_test_kill_switch(a, s):
    from . import risk as rk
    db = _db(s)
    p = ACTIVE_VERSION.risk
    prior = db.kill_switch()
    buy = rk.OrderIntent("news", "TEST", "buy", 100.0, target_notional=1000.0)
    sell = rk.OrderIntent("news", "TEST", "sell", 100.0, qty=5)
    results = {}
    try:
        db.set_kill_switch(True, "kill switch self-test")
        engaged = db.kill_switch()["engaged"]
        v1 = rk.check(buy, equity=10000, gross_exposure=0, kill_switch_engaged=engaged, entries_halted=False,
                      held_symbols={"TEST": 5}, p=p)
        v2 = rk.check(sell, equity=10000, gross_exposure=0, kill_switch_engaged=engaged, entries_halted=False,
                      held_symbols={"TEST": 5}, p=p)
        v3 = rk.check(buy, equity=10000, gross_exposure=0, kill_switch_engaged=False, entries_halted=True,
                      held_symbols={}, p=p)
        results = {"buy_blocked_by_kill_switch": not v1.approved, "exit_allowed_with_kill_switch": v2.approved,
                   "buy_blocked_by_daily_stop": not v3.approved, "switch_read_from_db": engaged}
    finally:
        db.set_kill_switch(prior["engaged"], prior["reason"] or "restored after self-test")
    ok = all(results.values())
    db.mark_stage(date.today(), "kill_switch_test", "done" if ok else "failed", results)
    print(("PASS " if ok else "FAIL ") + json.dumps(results))


def cmd_tick(a, s):
    from .runner import Runner
    print(json.dumps(Runner(s, _db(s)).tick(force_stage=a.stage), default=str))


def cmd_status(a, s):
    from .audit.gates import ALL_GATES
    from .execution import live_blockers
    db = _db(s)
    sv = ACTIVE_VERSION
    res = db.latest_gate_results(sv.version)
    print(f"strategy version {sv.version} | mode {s.trading_mode}")
    for g in ALL_GATES:
        print(f"  {g:28s} {'PASS' if res.get(g) else ('FAIL' if g in res else 'not run')}")
    print("kill switch:", db.kill_switch())
    blockers = live_blockers(s, res)
    print("live trading:", "UNLOCKED" if not blockers else "locked")
    mode = "live" if s.is_live else "paper"
    print("open positions:", len(db.open_positions(mode)))
    eq = db.equity_history(mode)
    if len(eq):
        print("latest equity:", eq.iloc[-1].to_dict())


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(prog="engine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db").set_defaults(fn=cmd_init_db)
    for name, fn in (("fetch-history", cmd_fetch_history), ("score-news", cmd_score_news), ("replay", cmd_replay)):
        p = sub.add_parser(name)
        p.add_argument("start", type=date.fromisoformat)
        p.add_argument("end", type=date.fromisoformat)
        if name == "score-news":
            p.add_argument("--max-calls", type=int)
        p.set_defaults(fn=fn)
    p = sub.add_parser("audit"); p.add_argument("run_id", type=int); p.set_defaults(fn=cmd_audit)
    sub.add_parser("paper-gates").set_defaults(fn=cmd_paper_gates)
    sub.add_parser("test-kill-switch").set_defaults(fn=cmd_test_kill_switch)
    p = sub.add_parser("tick"); p.add_argument("--stage", choices=["premarket", "exits", "close"]); p.set_defaults(fn=cmd_tick)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    p = sub.add_parser("kill"); p.add_argument("reason"); p.set_defaults(
        fn=lambda a, s: _db(s).set_kill_switch(True, a.reason))
    sub.add_parser("release").set_defaults(fn=lambda a, s: _db(s).set_kill_switch(False, "released by user"))
    a = ap.parse_args(argv)
    a.fn(a, load_settings())


if __name__ == "__main__":
    main()
