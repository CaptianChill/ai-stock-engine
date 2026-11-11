"""Audit gates. A strategy version reaches real money only when every gate's latest result
is a pass. Thresholds are fixed here, in code, before any result is seen; changing one is a
code change that shows up in review, not a knob turned after a bad backtest.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from ..config import StrategyVersion, uses_llm as _uses_llm
from ..replay import ReplayResult, metrics, run_replay
from . import stats
from .independent import reproduce

# Frozen thresholds -------------------------------------------------------------
DSR_MIN = 0.95
PBO_MAX = 0.50
OOS_FRACTION = 0.40
MAX_SYMBOL_SHARE = 0.25
MAX_DAY_SHARE = 0.25
MIN_DAYS = 60
MIN_NEWS_TRADES = 100
MIN_REBALANCES = 60          # 5 years of monthly momentum rebalances, from long price history
PAPER_MIN_FILLS = 5

REPLAY_GATES = ["no_lookahead", "cost_stress", "out_of_sample", "overfitting", "decay_haircut",
                "concentration", "min_sample", "independent_reproduction", "momentum_history"]
PAPER_GATES = ["paper_run_completed", "paper_match", "kill_switch_tested"]
ALL_GATES = REPLAY_GATES + PAPER_GATES


@dataclass
class GateResult:
    gate: str
    passed: bool
    detail: dict = field(default_factory=dict)




def gate_no_lookahead(res: ReplayResult, scores: pd.DataFrame, sv: StrategyVersion,
                      period_start: date, llm_cutoff: date) -> GateResult:
    d: dict = {"llm_knowledge_cutoff": str(llm_cutoff), "period_start": str(period_start)}
    ok = True
    if _uses_llm(sv):
        if period_start <= llm_cutoff:
            ok = False
            d["issue"] = "replay starts on or before the LLM's knowledge cutoff"
        if len(scores):
            first = pd.to_datetime(scores["published_at"], utc=True).min().date()
            d["first_news"] = str(first)
            if first <= llm_cutoff:
                ok = False
                d["issue_news"] = "news dated on or before the LLM cutoff was used"
    timing = res.provenance.get("max_lead_check", {"violations": 0})
    d["timing"] = timing
    if timing.get("violations", 0) > 0:
        ok = False
    return GateResult("no_lookahead", ok, d)


def gate_cost_stress(bars, scores, sv: StrategyVersion, start: date | None = None,
                     end: date | None = None) -> GateResult:
    stressed = sv.costs.cost_bps_per_side * sv.costs.stress_multiplier
    r = run_replay(bars, scores, sv, cost_bps=stressed, start=start, end=end)
    m = metrics(r.daily, r.trades)
    return GateResult("cost_stress", m["annual_return"] > 0 and m["sharpe"] > 0,
                      {"cost_bps_per_side": stressed, "annual_return": m["annual_return"], "sharpe": m["sharpe"]})


def gate_out_of_sample(daily: pd.Series) -> GateResult:
    n = len(daily)
    cut = int(n * (1 - OOS_FRACTION))
    m_is, m_oos = metrics(daily.iloc[:cut]), metrics(daily.iloc[cut:])
    ok = m_oos.get("sharpe", 0) > 0 and m_oos.get("total_return", 0) > 0
    return GateResult("out_of_sample", ok, {"in_sample": m_is, "out_of_sample": m_oos,
                                            "split_day": str(daily.index[cut]) if cut < n else None})


def gate_overfitting(daily: pd.Series, n_trials: int, variants: pd.DataFrame | None) -> GateResult:
    r = daily.values
    trial_var = None
    detail: dict = {}
    if variants is not None and variants.shape[1] >= 2:
        srs = [stats.sharpe(variants[c].values) for c in variants.columns]
        trial_var = float(np.var(srs, ddof=1))
    dsr = stats.deflated_sharpe(r, max(n_trials, 1), trial_var)
    detail["deflated_sharpe"] = dsr
    ok = dsr["dsr"] >= DSR_MIN
    if variants is not None and variants.shape[1] >= 2:
        try:
            pbo = stats.pbo_cscv(variants.values)
            detail["pbo"] = pbo
            ok = ok and pbo["pbo"] <= PBO_MAX
        except ValueError as e:
            detail["pbo"] = {"skipped": str(e)}
    else:
        detail["pbo"] = {"skipped": "fewer than two variants replayed on this period"}
    return GateResult("overfitting", ok, detail)


def gate_decay_haircut(daily: pd.Series) -> GateResult:
    hair = daily - 0.5 * daily.mean()
    m = metrics(hair)
    return GateResult("decay_haircut", m["annual_return"] > 0 and m["sharpe"] > 0,
                      {"annual_return_at_half_edge": m["annual_return"], "sharpe_at_half_edge": m["sharpe"]})


def gate_concentration(res: ReplayResult) -> GateResult:
    total = float(res.daily.sum())
    if total <= 0:
        return GateResult("concentration", False, {"issue": "no net profit to attribute", "total": total})
    pnl_sym: dict[str, float] = {}
    for t in res.trades:
        pnl_sym[t["symbol"]] = pnl_sym.get(t["symbol"], 0.0) + t["net"] * t.get("portfolio_weight", 0.0)
    pos_total = sum(v for v in pnl_sym.values() if v > 0) or 1e-12
    top_sym, top_sym_pnl = max(pnl_sym.items(), key=lambda kv: kv[1]) if pnl_sym else (None, 0.0)
    top_day = float(res.daily.max())
    sym_share, day_share = top_sym_pnl / pos_total, top_day / total
    return GateResult("concentration", sym_share <= MAX_SYMBOL_SHARE and day_share <= MAX_DAY_SHARE,
                      {"top_symbol": top_sym, "top_symbol_share": sym_share,
                       "top_day": str(res.daily.idxmax()), "top_day_share": day_share})


def gate_min_sample(res: ReplayResult, sv: StrategyVersion) -> GateResult:
    n_news = sum(1 for t in res.trades if t["engine"] == "news")
    from ..strategies import rebalance_days
    n_reb = len(rebalance_days(list(res.daily.index)))
    ok = len(res.daily) >= MIN_DAYS
    if sv.news_weight > 0:
        ok = ok and n_news >= MIN_NEWS_TRADES
    # Momentum's sample size is judged on years of price history in gate_momentum_history.
    return GateResult("min_sample", ok, {"days": len(res.daily), "news_trades": n_news,
                                         "rebalances_in_window": n_reb})


def momentum_only(sv: StrategyVersion) -> StrategyVersion:
    """The core engine alone, without the LLM veto: needs no news, so it can be judged on
    years of history instead of the few months after the LLM's training cutoff."""
    from dataclasses import replace
    return replace(sv, version=f"{sv.version}-mom-base", news_weight=0.0, momentum_weight=1.0,
                   momentum=replace(sv.momentum, llm_veto=False))


def gate_momentum_history(long_bars: pd.DataFrame | None, sv: StrategyVersion, n_trials: int) -> GateResult:
    """Base momentum over long history must be: >= 60 rebalances, profitable at 2x costs,
    profitable in its last 40%, and pass the deflated Sharpe hurdle."""
    if sv.momentum_weight <= 0:
        return GateResult("momentum_history", True, {"note": "momentum engine not used"})
    if long_bars is None or long_bars.empty:
        return GateResult("momentum_history", False, {"issue": "no long price history supplied"})
    from ..strategies import rebalance_days
    mv = momentum_only(sv)
    days = sorted(pd.to_datetime(long_bars["day"]).dt.date.unique())
    start = days[min(len(days) - 1, mv.momentum.lookback_days + 1)]
    empty = pd.DataFrame(columns=["published_at", "symbol", "score", "confidence"])
    res = run_replay(long_bars, empty, mv, use_news=False, start=start)
    stressed = run_replay(long_bars, empty, mv, use_news=False, start=start,
                          cost_bps=sv.costs.cost_bps_per_side * sv.costs.stress_multiplier)
    m, ms = metrics(res.daily, res.trades), metrics(stressed.daily)
    cut = int(len(res.daily) * (1 - OOS_FRACTION))
    oos = metrics(res.daily.iloc[cut:])
    dsr = stats.deflated_sharpe(res.daily.values, max(n_trials, 1))
    n_reb = len(rebalance_days(list(res.daily.index)))
    checks = {"rebalances>=60": n_reb >= MIN_REBALANCES, "profitable_at_2x_costs": ms["annual_return"] > 0,
              "last_40pct_profitable": oos["sharpe"] > 0 and oos["total_return"] > 0,
              "deflated_sharpe>=0.95": dsr["dsr"] >= DSR_MIN}
    return GateResult("momentum_history", all(checks.values()),
                      {"checks": checks, "rebalances": n_reb, "period": [str(res.daily.index[0]), str(res.daily.index[-1])],
                       "metrics": m, "stressed_annual_return": ms["annual_return"], "oos": oos, "deflated_sharpe": dsr})


def gate_independent(bars: pd.DataFrame, res: ReplayResult, sv: StrategyVersion) -> GateResult:
    raw = bars.to_dict("records")
    news_trades = [t for t in res.trades if t["engine"] == "news"]
    out = reproduce(raw, news_trades, sv.costs.cost_bps_per_side, sv.news.hold_days,
                    sv.news.min_dollar_volume, sv.risk.min_price)
    return GateResult("independent_reproduction", out["n_problems"] == 0, out)


def run_replay_gates(bars: pd.DataFrame, scores: pd.DataFrame, sv: StrategyVersion, period_start: date,
                     llm_cutoff: date, n_trials: int, variants: pd.DataFrame | None = None,
                     res: ReplayResult | None = None, period_end: date | None = None,
                     long_bars: pd.DataFrame | None = None) -> tuple[ReplayResult, list[GateResult]]:
    res = res or run_replay(bars, scores, sv, start=period_start, end=period_end)
    gates = [
        gate_no_lookahead(res, scores, sv, period_start, llm_cutoff),
        gate_cost_stress(bars, scores, sv, period_start, period_end),
        gate_out_of_sample(res.daily),
        gate_overfitting(res.daily, n_trials, variants),
        gate_decay_haircut(res.daily),
        gate_concentration(res),
        gate_min_sample(res, sv),
        gate_independent(bars, res, sv),
        gate_momentum_history(long_bars, sv, n_trials),
    ]
    return res, gates


# -- paper-trading gates (need the database) ------------------------------------------
def run_paper_gates(db, sv: StrategyVersion) -> list[GateResult]:
    out = []
    n_closes = db.count_stage_done("close")
    ks = db.last_stage("kill_switch_test")
    out.append(GateResult("paper_run_completed", n_closes >= 1, {"completed_trading_days": n_closes}))
    fills = db.filled_orders("paper")
    slips = [abs(o["filled_avg_price"] - o["ref_price"]) / o["ref_price"] * 1e4
             for o in fills if o.get("filled_avg_price") and o.get("ref_price")]
    mean_slip = float(np.mean(slips)) if slips else None
    out.append(GateResult(
        "paper_match",
        len(slips) >= PAPER_MIN_FILLS and mean_slip is not None and mean_slip <= sv.costs.cost_bps_per_side,
        {"fills": len(slips), "mean_slippage_bps": mean_slip, "allowed_bps": sv.costs.cost_bps_per_side}))
    out.append(GateResult("kill_switch_tested", bool(ks and ks["status"] == "done"),
                          {"last_test": ks["detail"] if ks else None}))
    return out
