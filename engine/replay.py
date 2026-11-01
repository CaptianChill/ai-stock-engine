"""Historical replay with costs charged on every trade.

Position sizing mirrors the live runner: the news engine splits its capital into
max_positions * hold_days equal slots (so overlapping holds never exceed 100% gross),
and momentum splits its capital equally across top_n names.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .config import StrategyVersion
from . import strategies as st

TRADING_DAYS = 252


@dataclass
class ReplayResult:
    daily: pd.Series            # combined net daily return, indexed by date
    news_daily: pd.Series
    momentum_daily: pd.Series
    trades: list[dict]          # every round trip, both engines
    provenance: dict            # facts the look-ahead audit checks


def _news_engine(bars, scores, eligible, sv: StrategyVersion, cost_bps: float):
    p = sv.news
    opens, closes = st.pivot(bars, "open"), st.pivot(bars, "close")
    days = list(closes.index)
    idx = {d: i for i, d in enumerate(days)}
    signals = st.news_signals(scores, days, p)
    picks = st.news_picks(signals, eligible, p)
    slot_w = 1.0 / (p.max_positions * p.hold_days)
    c = cost_bps / 1e4
    daily = pd.Series(0.0, index=days)
    trades = []
    for _, r in picks.iterrows():
        d, sym = r["trade_day"], r["symbol"]
        i = idx.get(d)
        if i is None or sym not in opens.columns:
            continue
        j = i + p.hold_days - 1
        if j >= len(days):
            continue  # incomplete trade at the end of the data: not counted
        entry = opens.at[d, sym]
        path = closes[sym].iloc[i:j + 1]
        if not np.isfinite(entry) or entry <= 0 or not np.all(np.isfinite(path.values)):
            continue
        prev = entry
        for k, px in enumerate(path.values):
            ret = px / prev - 1
            if k == 0:
                ret -= c
            if k == len(path) - 1:
                ret -= c
            daily.iloc[i + k] += slot_w * ret
            prev = px
        gross = path.values[-1] / entry - 1
        trades.append({"engine": "news", "symbol": sym, "entry_day": d, "exit_day": days[j],
                       "entry_px": float(entry), "exit_px": float(path.values[-1]),
                       "gross": float(gross), "net": float(gross - 2 * c),
                       "signal": float(r["signal"]), "weight": slot_w})
    prov = {
        "n_scores": int(len(scores)),
        "min_published": str(pd.to_datetime(scores["published_at"]).min()) if len(scores) else None,
        "max_lead_check": _check_news_timing(signals, p),
    }
    return daily, trades, signals, prov


def _check_news_timing(signals: pd.DataFrame, p) -> dict:
    """Every signal used for day d must come from news published before 09:15 ET on d."""
    if signals.empty:
        return {"violations": 0}
    last = pd.to_datetime(signals["last_published"], utc=True).dt.tz_convert(st.ET)
    cutoff = [pd.Timestamp.combine(d, pd.Timestamp(f"{p.entry_cutoff_hour_et:02d}:{p.entry_cutoff_minute_et:02d}").time())
              .tz_localize(st.ET) for d in signals["trade_day"]]
    bad = int(sum(1 for a, b in zip(last, cutoff) if a >= b))
    return {"violations": bad, "checked": int(len(signals))}


def _momentum_engine(bars, signals, eligible, sv: StrategyVersion, cost_bps: float):
    p = sv.momentum
    opens, closes = st.pivot(bars, "open"), st.pivot(bars, "close")
    days = list(closes.index)
    targets = st.momentum_targets(bars, signals, eligible, p)
    c = cost_bps / 1e4
    daily = pd.Series(0.0, index=days)
    trades = []
    held: dict[str, dict] = {}
    w = 1.0 / p.top_n
    reb = set(targets)
    for i, d in enumerate(days):
        if d in reb:
            new = set(targets[d])
            # Overnight leg for names being sold: prev close -> today's open, then exit cost.
            for sym in list(held):
                if sym not in new:
                    po, pc = opens.at[d, sym], closes[sym].iloc[i - 1]
                    leg = (po / pc - 1) if np.isfinite(po) and np.isfinite(pc) and pc > 0 else 0.0
                    daily.iloc[i] += w * (leg - c)
                    h = held.pop(sym)
                    exit_px = po if np.isfinite(po) else pc
                    gross = exit_px / h["entry_px"] - 1
                    trades.append({"engine": "momentum", "symbol": sym, "entry_day": h["entry_day"],
                                   "exit_day": d, "entry_px": h["entry_px"], "exit_px": float(exit_px),
                                   "gross": float(gross), "net": float(gross - 2 * c), "weight": w})
            # Kept names: normal close-to-close day.
            for sym in held:
                pc, cc = closes[sym].iloc[i - 1], closes.at[d, sym]
                if np.isfinite(pc) and np.isfinite(cc) and pc > 0:
                    daily.iloc[i] += w * (cc / pc - 1)
            # New names: open -> close, entry cost.
            for sym in new - set(held):
                po, cc = opens.at[d, sym], closes.at[d, sym]
                if not (np.isfinite(po) and np.isfinite(cc) and po > 0):
                    continue
                daily.iloc[i] += w * (cc / po - 1 - c)
                held[sym] = {"entry_day": d, "entry_px": float(po)}
            continue
        for sym in held:
            pc, cc = closes[sym].iloc[i - 1], closes.at[d, sym]
            if np.isfinite(pc) and np.isfinite(cc) and pc > 0:
                daily.iloc[i] += w * (cc / pc - 1)
    return daily, trades


def run_replay(bars: pd.DataFrame, scores: pd.DataFrame, sv: StrategyVersion,
               cost_bps: float | None = None, use_news: bool = True,
               start: date | None = None, end: date | None = None) -> ReplayResult:
    """`bars` should begin well before `start` (momentum needs ~a year of history).
    Results are trimmed to [start, end]: only trades entered inside the window count."""
    res = _run_full(bars, scores, sv, cost_bps, use_news)
    if start is None and end is None:
        return res
    lo = start or res.daily.index.min()
    hi = end or res.daily.index.max()
    keep = [d for d in res.daily.index if lo <= d <= hi]

    def trim(s: pd.Series) -> pd.Series:
        return s.loc[keep]

    trades = [t for t in res.trades if lo <= t["entry_day"] <= hi]
    return ReplayResult(daily=trim(res.daily), news_daily=trim(res.news_daily),
                        momentum_daily=trim(res.momentum_daily), trades=trades, provenance=res.provenance)


def _run_full(bars, scores, sv: StrategyVersion, cost_bps, use_news) -> ReplayResult:
    cost = sv.costs.cost_bps_per_side if cost_bps is None else cost_bps
    elig_news = st.eligible_universe(bars, sv.news.universe_size, sv.news.min_dollar_volume, sv.risk.min_price)
    elig_mom = st.eligible_universe(bars, sv.momentum.universe_size, sv.momentum.min_dollar_volume, sv.risk.min_price)
    days = sorted(pd.to_datetime(bars["day"]).dt.date.unique())
    if use_news and sv.news_weight > 0:
        n_daily, n_trades, signals, prov = _news_engine(bars, scores, elig_news, sv, cost)
    else:
        n_daily, n_trades, prov = pd.Series(0.0, index=days), [], {}
        signals = st.news_signals(scores, days, sv.news) if use_news else None
    m_daily, m_trades = _momentum_engine(bars, signals, elig_mom, sv, cost) if sv.momentum_weight > 0 \
        else (pd.Series(0.0, index=days), [])
    daily = sv.news_weight * n_daily + sv.momentum_weight * m_daily
    for t in n_trades:
        t["portfolio_weight"] = t["weight"] * sv.news_weight
    for t in m_trades:
        t["portfolio_weight"] = t["weight"] * sv.momentum_weight
    return ReplayResult(daily=daily, news_daily=n_daily, momentum_daily=m_daily,
                        trades=n_trades + m_trades, provenance=prov)


def metrics(daily: pd.Series, trades: list[dict] | None = None) -> dict:
    r = daily.astype(float)
    n = len(r)
    if n == 0:
        return {"days": 0}
    equity = (1 + r).cumprod()
    total = float(equity.iloc[-1] - 1)
    ann = float(equity.iloc[-1] ** (TRADING_DAYS / n) - 1) if equity.iloc[-1] > 0 else -1.0
    vol = float(r.std(ddof=1) * math.sqrt(TRADING_DAYS)) if n > 1 else 0.0
    sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(TRADING_DAYS)) if n > 1 and r.std(ddof=1) > 0 else 0.0
    dd = float((equity / equity.cummax() - 1).min())
    out = {"days": n, "total_return": total, "annual_return": ann, "annual_vol": vol,
           "sharpe": sharpe, "max_drawdown": dd, "daily_mean": float(r.mean()),
           "daily_std": float(r.std(ddof=1)) if n > 1 else 0.0}
    if trades:
        nets = np.array([t["net"] for t in trades])
        out.update({"n_trades": int(len(nets)), "hit_rate": float((nets > 0).mean()),
                    "avg_net_trade": float(nets.mean())})
    else:
        out.update({"n_trades": 0})
    return out
