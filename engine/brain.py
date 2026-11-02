"""Review brain: runs every night on its own, reads what happened, writes a report.

What it does:
  1. Data watchdog   - missing bars, news-feed gaps, failed stages, stuck orders.
  2. Scorer check    - do the LLM's "good" headlines actually beat its "bad" ones?
  3. Trading review  - fills, slippage, equity, alerts, gate status.
  4. Written review  - an LLM turns the facts into findings and up to 3 proposals.

What it can NOT do: place, cancel or size orders, change the live strategy, or touch the
kill switch's release. Proposals land in a queue; only `approve` turns one into a candidate
strategy version, which then has to pass every audit gate like any other version.
(test_brain.py fails if this module ever imports the order-placing code.)
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import strategies as st
from .config import StrategyVersion

log = logging.getLogger("engine.brain")
NY = ZoneInfo("America/New_York")
MAX_PROPOSALS = 3

# Parameters the brain may propose changing, with hard bounds. Risk limits may only be
# tightened from the defaults, never loosened.
ALLOWED_PARAMS: dict[str, tuple[float, float, type]] = {
    "news.min_signal": (0.30, 0.95, float),
    "news.max_positions": (3, 30, int),
    "news.hold_days": (1, 5, int),
    "momentum.top_n": (5, 40, int),
    "momentum.veto_threshold": (-1.0, 0.0, float),
    "risk.max_position_pct": (0.02, 0.10, float),
    "risk.daily_loss_stop_pct": (0.01, 0.03, float),
}


# -- deterministic checks (no AI involved) --------------------------------------------
def scorer_quality(scores: pd.DataFrame, bars: pd.DataFrame, sv: StrategyVersion) -> dict:
    """For each scored headline, the stock's open->close return on the trade day it informed.
    A working scorer: 'good' items beat 'bad' items."""
    if scores.empty or bars.empty:
        return {"items": 0, "note": "no scored news or bars in window"}
    days = sorted(pd.to_datetime(bars["day"]).dt.date.unique())
    s = scores.copy()
    s["trade_day"] = st.assign_trade_day(s["published_at"], days, sv.news)
    s = s.dropna(subset=["trade_day"])
    o, c = st.pivot(bars, "open"), st.pivot(bars, "close")
    rets = []
    for d, sym in zip(s["trade_day"], s["symbol"]):
        if d in o.index and sym in o.columns and np.isfinite(o.at[d, sym]) and o.at[d, sym] > 0:
            rets.append(c.at[d, sym] / o.at[d, sym] - 1)
        else:
            rets.append(np.nan)
    s["ret"] = rets
    s = s.dropna(subset=["ret"])
    good, bad = s[s["score"] > 0]["ret"], s[s["score"] < 0]["ret"]
    out = {"items": int(len(s)), "good_items": int(len(good)), "bad_items": int(len(bad)),
           "good_mean_ret": float(good.mean()) if len(good) else None,
           "bad_mean_ret": float(bad.mean()) if len(bad) else None,
           "good_hit_rate": float((good > 0).mean()) if len(good) else None}
    if len(good) >= 25 and len(bad) >= 25:
        spread = float(good.mean() - bad.mean())
        se = float(np.sqrt(good.var(ddof=1) / len(good) + bad.var(ddof=1) / len(bad)))
        out.update(spread=spread, spread_t=spread / se if se > 0 else None, working=spread > 0)
    return out


def data_watchdog(bars: pd.DataFrame, news_counts: pd.Series, stages: list[dict], open_orders: list[dict],
                  held: set[str], last_day: date) -> list[dict]:
    issues = []
    if not bars.empty:
        per_day = bars.groupby("day")["symbol"].nunique()
        last = int(per_day.get(last_day, 0))
        typical = float(per_day.iloc[-21:-1].median()) if len(per_day) > 2 else last
        if typical and last < 0.9 * typical:
            issues.append({"severity": "high", "title": "Bars missing",
                           "detail": f"{last} symbols have a bar for {last_day}, typical is {typical:.0f}"})
        have = set(bars[bars["day"] == last_day]["symbol"])
        missing_held = sorted(held - have)
        if missing_held:
            issues.append({"severity": "high", "title": "No price for held stocks",
                           "detail": f"{missing_held[:10]} have no bar for {last_day}"})
    if len(news_counts) >= 5:
        last_n, typical_n = float(news_counts.iloc[-1]), float(news_counts.iloc[-21:-1].median())
        if typical_n > 0 and last_n < 0.4 * typical_n:
            issues.append({"severity": "medium", "title": "News feed thin",
                           "detail": f"{last_n:.0f} items on the last day vs typical {typical_n:.0f}"})
    for s in stages:
        if s["status"] == "failed":
            issues.append({"severity": "high", "title": f"Stage failed: {s['stage']}",
                           "detail": json.dumps(s.get("detail"), default=str)[:300]})
    stuck = [o for o in open_orders if o["status"].startswith("error") or
             (o.get("submitted_at") and (datetime.now(tz=NY) - o["submitted_at"]).days >= 1)]
    if stuck:
        issues.append({"severity": "high", "title": "Orders stuck or errored",
                       "detail": f"{len(stuck)} orders, e.g. {[o['symbol'] for o in stuck[:5]]}"})
    return issues


# -- proposals ----------------------------------------------------------------------
def validate_proposals(raw: list, sv: StrategyVersion) -> tuple[list[dict], list[dict]]:
    """Keep only proposals for allowed parameters, within bounds, that actually change something."""
    kept, dropped = [], []
    for p in (raw or [])[: MAX_PROPOSALS * 2]:
        try:
            param = str(p["param"])
            lo, hi, typ = ALLOWED_PARAMS[param]
            new = typ(p["new_value"])
            old = sv.get_param(param)
            if not (lo <= new <= hi):
                raise ValueError(f"out of bounds [{lo}, {hi}]")
            if param.startswith("risk.") and new > old:
                raise ValueError("risk limits may only be tightened")
            if new == old:
                raise ValueError("no change")
            kept.append({"param": param, "old_value": old, "new_value": new,
                         "rationale": str(p.get("rationale", ""))[:1000]})
        except (KeyError, ValueError, TypeError) as e:
            dropped.append({"proposal": p, "why": str(e)})
        if len(kept) >= MAX_PROPOSALS:
            break
    return kept, dropped


SYSTEM = (
    "You are the nightly reviewer for a paper-trading system. An LLM labels overnight news; fixed rules "
    "pick stocks; a risk agent sizes orders. You review facts and report. You cannot trade. Be specific and "
    "skeptical: one day is noise; never propose a change because of a single day's result. Propose a "
    "parameter change only when the facts show a persistent problem, and say what evidence would confirm it. "
    "Reply with JSON only."
)

PROMPT = """Facts (JSON):
{facts}

Parameters you may propose changing (name: [min, max]); risk limits may only be tightened:
{allowed}

Current values: {current}

Reply exactly as JSON:
{{"summary": "<2-4 sentences>",
  "findings": [{{"severity": "high|medium|low", "title": "...", "detail": "..."}}],
  "proposals": [{{"param": "<name>", "new_value": <number>, "rationale": "<evidence + how to confirm>"}}]}}
Use an empty proposals list when nothing is clearly warranted (most nights)."""


def llm_review(facts: dict, sv: StrategyVersion, api_key: str | None, model: str) -> dict:
    if not api_key:
        return {"summary": "LLM review skipped: no ANTHROPIC_API_KEY. Deterministic checks still ran.",
                "findings": [], "proposals": [], "llm": False}
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    current = {k: sv.get_param(k) for k in ALLOWED_PARAMS}
    msg = client.messages.create(
        model=model, max_tokens=1500, system=SYSTEM,
        messages=[{"role": "user", "content": PROMPT.format(
            facts=json.dumps(facts, default=str)[:30000],
            allowed=json.dumps({k: [v[0], v[1]] for k, v in ALLOWED_PARAMS.items()}),
            current=json.dumps(current))}])
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return parse_review(text)


def parse_review(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        j = json.loads(m.group(0)) if m else {}
    except ValueError:
        j = {}
    if not isinstance(j, dict) or "summary" not in j:
        return {"summary": "LLM review unreadable; deterministic checks still ran.", "findings": [],
                "proposals": [], "llm": False, "raw": (text or "")[:2000]}
    j.setdefault("findings", [])
    j.setdefault("proposals", [])
    j["llm"] = True
    return j


# -- the nightly run ------------------------------------------------------------------
def gather_facts(db, sv: StrategyVersion, mode: str, today: date, score_model: str,
                 prompt_version: str) -> dict:
    lookback = today - timedelta(days=45)
    bars = db.load_bars(lookback, today)
    trade_days = sorted(pd.to_datetime(bars["day"]).dt.date.unique()) if len(bars) else []
    last_day = trade_days[-1] if trade_days else today
    since = datetime.combine(lookback, time(0), NY)
    until = datetime.combine(today, time(23, 59), NY)
    scores = db.load_scores(since, until, score_model, prompt_version)
    news_counts = db.news_counts_by_day(lookback, today)
    fills = db.filled_orders(mode, since=today - timedelta(days=7))
    slips = [abs(o["filled_avg_price"] - o["ref_price"]) / o["ref_price"] * 1e4
             for o in fills if o.get("filled_avg_price") and o.get("ref_price")]
    eq = db.equity_history(mode)
    held = {p["symbol"] for p in db.open_positions(mode)}
    facts = {
        "day": str(today), "mode": mode, "strategy_version": sv.version,
        "watchdog": data_watchdog(bars, news_counts, db.recent_stages(today - timedelta(days=3)),
                                  db.open_orders(mode), held, last_day),
        "scorer_quality_30d": scorer_quality(scores, bars, sv),
        "fills_7d": len(fills),
        "mean_slippage_bps_7d": float(np.mean(slips)) if slips else None,
        "assumed_cost_bps": sv.costs.cost_bps_per_side,
        "equity_last_10": eq.tail(10).to_dict("records") if len(eq) else [],
        "open_positions": len(held),
        "alerts_3d": db.recent_alerts(today - timedelta(days=3)),
        "gates": db.latest_gate_results(sv.version),
        "kill_switch": db.kill_switch(),
        "decisions_today": db.decision_summary(today),
    }
    return facts


def nightly(db, sv: StrategyVersion, mode: str, api_key: str | None, model: str, score_model: str,
            prompt_version: str, today: date | None = None, review_fn=llm_review) -> dict:
    today = today or datetime.now(tz=NY).date()
    facts = gather_facts(db, sv, mode, today, score_model, prompt_version)
    review = review_fn(facts, sv, api_key, model)
    kept, dropped = validate_proposals(review.get("proposals"), sv)
    review["proposals"], review["dropped_proposals"] = kept, dropped
    findings = facts["watchdog"] + [f for f in review.get("findings", []) if isinstance(f, dict)]
    review["findings"] = findings
    report_id = db.save_brain_report(today, facts, review, model if review.get("llm") else None)
    for p in kept:
        db.insert_proposal(report_id, p)
    urgent = [f for f in findings if f.get("severity") == "high"]
    q = facts["scorer_quality_30d"]
    if q.get("working") is False and (q.get("spread_t") or 0) < -1:
        urgent.append({"severity": "high", "title": "LLM scorer not working",
                       "detail": f"good-minus-bad spread {q['spread']:.4f} (t={q['spread_t']:.1f})"})
    if urgent:
        db.raise_alert("brain", {"report_id": report_id, "urgent": urgent})
    return {"report_id": report_id, "summary": review.get("summary"), "urgent": len(urgent),
            "proposals": len(kept)}


def approve(db, proposal_id: int, base: StrategyVersion) -> str:
    """Turn a pending proposal into a candidate version (registered, not active). It must be
    replayed and pass every audit gate; switching the live config to it is a separate, manual step."""
    p = db.get_proposal(proposal_id)
    if p is None or p["status"] != "pending":
        raise ValueError(f"proposal {proposal_id} is not pending")
    name = f"{base.version}-p{proposal_id}"
    cand = base.with_param(p["param"], p["new_value"], name)
    db.register_version(name, "combined", cand.to_json())
    db.decide_proposal(proposal_id, "approved", name)
    return name
