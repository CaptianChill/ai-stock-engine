"""Strategy engines as pure functions. The replay and the live runner call the SAME
functions, so what is audited is exactly what trades.

Timing rules (the no-look-ahead contract):
- Universe eligibility for day d uses bars up to d-1 only.
- News counts for day d if published after 16:00 ET on the previous trading day and before
  09:15 ET on day d. News published during market hours is ignored (the market already reacted).
- Entries happen at day d's open (market-on-open order), exits at a later day's close
  (market-on-close order).
"""
from __future__ import annotations

from datetime import date, time

import numpy as np
import pandas as pd

from .config import MomentumParams, NewsParams

ET = "America/New_York"
MARKET_CLOSE = time(16, 0)


# -- shared ---------------------------------------------------------------------
def pivot(bars: pd.DataFrame, field: str) -> pd.DataFrame:
    df = bars.copy()
    df["day"] = pd.to_datetime(df["day"]).dt.date
    return df.pivot(index="day", columns="symbol", values=field).sort_index()


def eligible_universe(bars: pd.DataFrame, universe_size: int, min_dollar_volume: float,
                      min_price: float = 5.0, window: int = 20) -> dict[date, set[str]]:
    """For each day d: top `universe_size` symbols by mean dollar volume over the `window`
    days ending d-1, with prior close >= min_price. Uses only data available before d's open."""
    close = pivot(bars, "close")
    vol = pivot(bars, "volume")
    dv = (close * vol).rolling(window, min_periods=window).mean().shift(1)
    prev_close = close.shift(1)
    out: dict[date, set[str]] = {}
    for d in dv.index:
        row = dv.loc[d]
        ok = row[(row >= min_dollar_volume) & (prev_close.loc[d] >= min_price)].dropna()
        out[d] = set(ok.sort_values(ascending=False).index[:universe_size])
    return out


# -- fast engine: LLM news reaction -------------------------------------------------
def assign_trade_day(published_at: pd.Series, trading_days: list[date], p: NewsParams) -> pd.Series:
    """Map each news timestamp to the trading day whose open it can inform, or NaT if it
    arrived during market hours (or after the last known trading day)."""
    days = np.array(sorted(trading_days), dtype="datetime64[D]")
    day_set = set(trading_days)
    cutoff = time(p.entry_cutoff_hour_et, p.entry_cutoff_minute_et)
    ts = pd.to_datetime(published_at, utc=True).dt.tz_convert(ET)
    out = []
    for t in ts:
        d, tod = t.date(), t.time()
        if d in day_set:
            if tod < cutoff:
                out.append(d)
                continue
            if tod < MARKET_CLOSE:
                out.append(None)  # intraday news: excluded
                continue
            idx = np.searchsorted(days, np.datetime64(d), side="right")
        else:
            idx = np.searchsorted(days, np.datetime64(d), side="left")
        out.append(days[idx].astype(object) if idx < len(days) else None)
    return pd.Series(out, index=published_at.index, dtype=object)


def news_signals(scores: pd.DataFrame, trading_days: list[date], p: NewsParams) -> pd.DataFrame:
    """scores: published_at (tz-aware), symbol, score in {-1,0,1}, confidence in [0,1].
    Returns trade_day, symbol, signal = mean(score*confidence), n_items, last_published."""
    if scores.empty:
        return pd.DataFrame(columns=["trade_day", "symbol", "signal", "n_items", "last_published"])
    df = scores.copy()
    df["trade_day"] = assign_trade_day(df["published_at"], trading_days, p)
    df = df.dropna(subset=["trade_day"])
    df["w"] = df["score"].astype(float) * df["confidence"].astype(float)
    g = df.groupby(["trade_day", "symbol"])
    out = g.agg(signal=("w", "mean"), n_items=("w", "size"), last_published=("published_at", "max")).reset_index()
    return out


def news_picks(signals: pd.DataFrame, eligible: dict[date, set[str]], p: NewsParams) -> pd.DataFrame:
    """Top `max_positions` names per day with signal >= min_signal, from the eligible universe."""
    if signals.empty:
        return signals.assign(rank=[])
    s = signals[signals["signal"] >= p.min_signal].copy()
    s = s[[sym in eligible.get(d, set()) for d, sym in zip(s["trade_day"], s["symbol"])]]
    s = s.sort_values(["trade_day", "signal", "n_items"], ascending=[True, False, False])
    s["rank"] = s.groupby("trade_day").cumcount() + 1
    return s[s["rank"] <= p.max_positions].reset_index(drop=True)


# -- core engine: LLM-filtered momentum -----------------------------------------------
def rebalance_days(trading_days: list[date]) -> list[date]:
    """First trading day of each calendar month."""
    out, seen = [], set()
    for d in sorted(trading_days):
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def rank_momentum(close: pd.DataFrame, i: int, eligible: set[str], veto_days: set[date],
                  signals: pd.DataFrame | None, p: MomentumParams) -> list[str] | None:
    """Targets for the day at position i of close.index (i may equal len(close) for 'today',
    whose bar does not exist yet). Uses closes strictly before position i."""
    if i - 1 - p.lookback_days < 0:
        return None
    recent = close.iloc[i - 1 - p.skip_days]
    past = close.iloc[i - 1 - p.lookback_days]
    mom = (recent / past - 1).dropna()
    mom = mom[[s in eligible for s in mom.index]].sort_values(ascending=False)
    vetoed: set[str] = set()
    if p.llm_veto and signals is not None and not signals.empty:
        recent_news = signals[signals["trade_day"].isin(veto_days)]
        avg = recent_news.groupby("symbol")["signal"].mean()
        vetoed = set(avg[avg < p.veto_threshold].index)
    return [s for s in mom.index if s not in vetoed][: p.top_n]


def momentum_targets(bars: pd.DataFrame, signals: pd.DataFrame | None, eligible: dict[date, set[str]],
                     p: MomentumParams) -> dict[date, list[str]]:
    """For each rebalance day d: rank eligible names by close[d-1-skip]/close[d-1-lookback]-1,
    drop names whose recent news signal is below the veto threshold, keep top_n."""
    close = pivot(bars, "close")
    days = list(close.index)
    pos = {d: i for i, d in enumerate(days)}
    out: dict[date, list[str]] = {}
    for d in rebalance_days(days):
        i = pos[d]
        veto_days = set(days[max(0, i - p.veto_window_days + 1): i + 1])
        picks = rank_momentum(close, i, eligible.get(d, set()), veto_days, signals, p)
        if picks is not None:
            out[d] = picks
    return out


def eligible_next(bars: pd.DataFrame, universe_size: int, min_dollar_volume: float,
                  min_price: float = 5.0, window: int = 20) -> set[str]:
    """Eligibility for the trading day AFTER the last bar in `bars` (live use: 'today')."""
    close = pivot(bars, "close")
    vol = pivot(bars, "volume")
    dv = (close * vol).iloc[-window:]
    if len(dv) < window:
        return set()
    mean_dv = dv.mean(skipna=False)
    last_close = close.iloc[-1]
    ok = mean_dv[(mean_dv >= min_dollar_volume) & (last_close >= min_price)].dropna()
    return set(ok.sort_values(ascending=False).index[:universe_size])
