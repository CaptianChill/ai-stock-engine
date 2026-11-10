"""Command line: python -m engine <command> ...

  init-db                         apply schema.sql
  fetch-history START END         daily bars (from ~13 months before START) + news for START..END
  score-news START END [--max-calls N] [--batch]   LLM-score news (--batch: half price, via the Batches API)
  replay START END                run the replay with costs, store it, print metrics
  fetch-bars START END            price history only (for the long momentum check, e.g. from 2016)
  audit RUN_ID [--momentum-history-start 2016-01-01]   run every replay gate and record results
  paper-gates                     run the paper-trading gates and record results
  test-kill-switch                prove the kill switch and loss stop block entries
  tick [--stage S]                scheduler entry point (paper unless unlocked for live)
  research START END [--max-calls N] [--skip-fetch] [--batch]   the whole validation run in one go
  status                          gates, kill switch, open positions, latest equity
  kill "reason" / release         engage or release the kill switch
  brain                           nightly review: watchdog, scorer check, written review, proposals
  report                          print the latest brain report
  proposals [--all] / approve ID / reject ID   the brain's proposal queue (approve = test only)
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


def _last_weekday() -> date:
    d = date.today() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _history_start(start: date) -> date:
    return start - timedelta(days=int(ACTIVE_VERSION.momentum.lookback_days * 1.6) + 30)


def _version(db, name: str | None):
    """The active version, or a registered candidate (e.g. one the brain proposed and you approved)."""
    from .config import StrategyVersion
    if not name or name == ACTIVE_VERSION.version:
        return ACTIVE_VERSION
    params = db.version_params(name)
    if params is None:
        sys.exit(f"strategy version {name} is not registered")
    return StrategyVersion.from_json(params)


def cmd_init_db(a, s):
    _db(s).init_schema()
    print("schema applied")


def cmd_fetch_history(a, s):
    """Fill in recent prices for the stocks already stored, then download news. Resumable:
    re-running skips what is already in the database."""
    from .alpaca import DataClient
    db = _db(s)
    dc = DataClient(s.alpaca_key, s.alpaca_secret)
    cov = db.bars_coverage()
    if not cov:
        sys.exit("no price history yet: run fetch-bars first")
    want_from = _history_start(a.start)
    # Group symbols by where their stored history ends, and fetch only the missing tail.
    gaps: dict[date, list[str]] = {}
    for sym, (lo, hi) in cov.items():
        if hi < a.end - timedelta(days=4):
            gaps.setdefault(max(hi + timedelta(days=1), want_from), []).append(sym)
    n = 0
    for frm, syms in sorted(gaps.items()):
        n += db.upsert_bars(dc.daily_bars(sorted(syms), frm, a.end))
    print(f"price gaps filled: {n} bars for {sum(len(v) for v in gaps.values())} of {len(cov)} stocks")
    since = datetime.combine(a.start - timedelta(days=14), time(0), NY)
    until = datetime.combine(a.end, time(23, 59), NY)
    last = db.last_news_time(since, until)
    if last is not None and last > since:
        print(f"resuming news download after {last}")
        since = last
    n = 0
    batch = []
    for item in dc.news(since, until):
        batch.append(item)
        if len(batch) >= 1000:
            n += db.upsert_news(batch)
            batch = []
            print(f"news stored: {n}", flush=True)
    n += db.upsert_news(batch)
    print(f"news stored: {n}")


def cmd_fetch_bars(a, s):
    """Price history only (no news): for the long-history momentum check."""
    from .alpaca import DataClient, TradingClient
    from .runner import EXCHANGES
    db = _db(s)
    tc = TradingClient(s.alpaca_key, s.alpaca_secret, live=False)
    dc = DataClient(s.alpaca_key, s.alpaca_secret)
    syms = sorted(x["symbol"] for x in tc.assets()
                  if x.get("tradable") and x.get("exchange") in EXCHANGES and "." not in x["symbol"])
    # Rank by recent dollar volume and keep the most-traded names: the engines only ever trade
    # the top 500, and the full market since 2016 would not fit a free 1 GB database.
    # The ranking is saved, so a restart resumes instead of starting over.
    from pathlib import Path
    rank_file = Path("logs") / f"top{a.top}_{a.end}.txt"
    if rank_file.exists():
        keep = rank_file.read_text().split()
        print(f"resuming with the saved list of {len(keep)} most-traded stocks")
    else:
        recent = pd.DataFrame(list(dc.daily_bars(syms, a.end - timedelta(days=45), a.end)))
        if recent.empty:
            sys.exit("no recent bars returned; check the Alpaca keys and data feed")
        dv = (recent["close"] * recent["volume"]).groupby(recent["symbol"]).mean()
        keep = sorted(dv.sort_values(ascending=False).index[: a.top])
        rank_file.parent.mkdir(exist_ok=True)
        rank_file.write_text("\n".join(keep))
        print(f"keeping the {len(keep)} most-traded of {len(syms)} symbols")
    cov = db.bars_coverage()
    # A stock is done when its stored history reaches the end date (stocks listed after the
    # start date simply begin later; Alpaca pages each stock in date order, so a cut-off
    # download shows up as an early last day).
    done = {sym for sym, (lo, hi) in cov.items() if hi >= a.end - timedelta(days=4)}
    todo = [sym for sym in keep if sym not in done]
    if len(todo) < len(keep):
        print(f"{len(keep) - len(todo)} stocks already downloaded; fetching the other {len(todo)}")
    total = 0
    for i in range(0, len(todo), 100):
        total += db.upsert_bars(dc.daily_bars(todo[i:i + 100], a.start, a.end))
        print(f"bars stored: {total} ({min(i + 100, len(todo))}/{len(todo)} stocks)", flush=True)


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
    if getattr(a, "batch", False):
        from .llm_scorer import BatchScorer
        stored = BatchScorer(scorer, "logs/batches").run(rows, db.insert_scores)
        print(f"batch scoring finished: {stored} scores stored")
        return
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
    sv = _version(db, a.version)
    db.register_version(sv.version, "combined", sv.to_json())
    bars, scores = _load_inputs(db, s, a.start, a.end)
    res = run_replay(bars, scores, sv, start=a.start, end=a.end)
    m = metrics(res.daily, res.trades)
    m["news"] = metrics(res.news_daily, [t for t in res.trades if t["engine"] == "news"])
    m["momentum"] = metrics(res.momentum_daily, [t for t in res.trades if t["engine"] == "momentum"])
    run_id = db.save_replay(sv.version, a.start, a.end, sv.costs.cost_bps_per_side, m, res.daily, res.trades)
    print(json.dumps({"run_id": run_id, **m}, indent=2, default=str))
    return run_id


def cmd_research(a, s):
    """The whole validation run in one command (what the replay-and-audit workflow does):
    price history, news, LLM scoring, replay, every audit gate, kill-switch self-test, status."""
    from argparse import Namespace as N
    print("== init-db"); cmd_init_db(N(), s)
    if not a.skip_fetch:
        print(f"== fetch-bars {a.momentum_history_start} {a.end}")
        cmd_fetch_bars(N(start=a.momentum_history_start, end=a.end, top=1500), s)
        print(f"== fetch-history {a.start} {a.end}")
        cmd_fetch_history(N(start=a.start, end=a.end), s)
    print("== score-news"); cmd_score_news(N(start=a.start, end=a.end, max_calls=a.max_calls, batch=a.batch), s)
    print("== replay"); run_id = cmd_replay(N(start=a.start, end=a.end, version=None), s)
    print(f"== audit {run_id}"); cmd_audit(N(run_id=run_id, momentum_history_start=a.momentum_history_start), s)
    print("== test-kill-switch"); cmd_test_kill_switch(N(), s)
    print("== status"); cmd_status(N(), s)


def cmd_audit(a, s):
    from .audit.gates import run_replay_gates
    db = _db(s)
    run = db.load_replay(a.run_id)
    if run is None:
        sys.exit(f"no replay run {a.run_id}")
    sv = _version(db, run["version"])
    start, end = run["period_start"], run["period_end"]
    bars, scores = _load_inputs(db, s, start, end)
    variants = None
    others = db.replays_for_period(start, end)
    if len({o["version"] for o in others}) >= 2:
        latest = {}
        for o in others:
            latest[o["version"]] = o["daily_returns"]
        variants = pd.DataFrame(latest).sort_index().fillna(0.0)
    long_bars = db.load_bars(a.momentum_history_start, end) if sv.momentum_weight > 0 else None
    _, gates = run_replay_gates(bars, scores, sv, start, s.llm_knowledge_cutoff,
                                n_trials=db.count_versions(), variants=variants, period_end=end,
                                long_bars=long_bars)
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


def cmd_brain(a, s):
    from . import brain
    import os
    mode = "live" if s.is_live else "paper"
    out = brain.nightly(_db(s), ACTIVE_VERSION, mode, s.anthropic_key,
                        os.environ.get("BRAIN_MODEL", "claude-sonnet-5-5"), s.llm_model, s.llm_prompt_version)
    print(json.dumps(out, indent=2, default=str))


def cmd_report(a, s):
    r = _db(s).latest_brain_report()
    if not r:
        print("no brain report yet")
        return
    rv = r["review"]
    print(f"Brain report for {r['report_day']} ({r['model'] or 'deterministic checks only'})\n")
    print(rv.get("summary", ""), "\n")
    for f in rv.get("findings", []):
        print(f"[{f.get('severity', '?').upper()}] {f.get('title')}: {f.get('detail')}")
    q = r["facts"].get("scorer_quality_30d", {})
    print(f"\nLLM scorer, last 30 days: {json.dumps(q, default=str)}")


def cmd_proposals(a, s):
    rows = _db(s).list_proposals(None if a.all else "pending")
    if not rows:
        print("no proposals" + ("" if a.all else " pending"))
    for p in rows:
        print(f"#{p['proposal_id']} [{p['status']}] {p['param']}: {p['old_value']} -> {p['new_value']}"
              f"{'  candidate ' + p['candidate_version'] if p['candidate_version'] else ''}\n    {p['rationale']}")


def cmd_approve(a, s):
    from . import brain
    name = brain.approve(_db(s), a.proposal_id, ACTIVE_VERSION)
    print(f"Approved for TESTING as candidate version {name}. It is not trading.\n"
          f"Next: python -m engine replay START END --version {name}, then audit the run.")


def cmd_reject(a, s):
    db = _db(s)
    p = db.get_proposal(a.proposal_id)
    if p is None or p["status"] != "pending":
        sys.exit(f"proposal {a.proposal_id} is not pending")
    db.decide_proposal(a.proposal_id, "rejected")
    print("rejected")


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
            p.add_argument("--batch", action="store_true", help="half-price scoring via the Batches API")
        if name == "replay":
            p.add_argument("--version", help="replay a registered candidate version instead of the active one")
        p.set_defaults(fn=fn)
    p = sub.add_parser("audit"); p.add_argument("run_id", type=int)
    p.add_argument("--momentum-history-start", type=date.fromisoformat, default=date(2016, 1, 1))
    p.set_defaults(fn=cmd_audit)
    p = sub.add_parser("fetch-bars"); p.add_argument("start", type=date.fromisoformat)
    p.add_argument("end", type=date.fromisoformat)
    p.add_argument("--top", type=int, default=1500, help="most-traded symbols to keep (default 1500)")
    p.set_defaults(fn=cmd_fetch_bars)
    sub.add_parser("paper-gates").set_defaults(fn=cmd_paper_gates)
    sub.add_parser("test-kill-switch").set_defaults(fn=cmd_test_kill_switch)
    p = sub.add_parser("tick"); p.add_argument("--stage", choices=["premarket", "exits", "close"]); p.set_defaults(fn=cmd_tick)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    p = sub.add_parser("research", help="full validation run: history, scoring, replay, audit")
    p.add_argument("start", type=date.fromisoformat)
    p.add_argument("end", type=date.fromisoformat, nargs="?", default=_last_weekday(),
                   help="default: the most recent weekday before today")
    p.add_argument("--max-calls", type=int, default=None)
    p.add_argument("--momentum-history-start", type=date.fromisoformat, default=date(2016, 1, 1))
    p.add_argument("--skip-fetch", action="store_true")
    p.add_argument("--batch", action="store_true", help="half-price scoring via the Batches API")
    p.set_defaults(fn=cmd_research)
    sub.add_parser("brain").set_defaults(fn=cmd_brain)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    p = sub.add_parser("proposals"); p.add_argument("--all", action="store_true"); p.set_defaults(fn=cmd_proposals)
    p = sub.add_parser("approve"); p.add_argument("proposal_id", type=int); p.set_defaults(fn=cmd_approve)
    p = sub.add_parser("reject"); p.add_argument("proposal_id", type=int); p.set_defaults(fn=cmd_reject)
    p = sub.add_parser("kill"); p.add_argument("reason"); p.set_defaults(
        fn=lambda a, s: _db(s).set_kill_switch(True, a.reason))
    sub.add_parser("release").set_defaults(fn=lambda a, s: _db(s).set_kill_switch(False, "released by user"))
    a = ap.parse_args(argv)
    a.fn(a, load_settings())


if __name__ == "__main__":
    main()
