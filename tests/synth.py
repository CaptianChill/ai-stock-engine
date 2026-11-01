"""Synthetic market with an optional planted news signal, for testing the machinery.
Passing these tests proves the code is correct, NOT that any real strategy makes money."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from engine.config import CostParams, MomentumParams, NewsParams, RiskParams, StrategyVersion

NY = ZoneInfo("America/New_York")


def make_version(**kw) -> StrategyVersion:
    sv = StrategyVersion(
        version="test",
        news=NewsParams(universe_size=40, min_dollar_volume=1e6, max_positions=5, min_signal=0.5, hold_days=2),
        momentum=MomentumParams(universe_size=40, min_dollar_volume=1e6, lookback_days=120, skip_days=10,
                                top_n=5, veto_window_days=5),
        risk=RiskParams(min_price=1.0),
        costs=CostParams(cost_bps_per_side=5.0),
    )
    return replace(sv, **kw)


def make_market(n_symbols=40, n_days=420, edge=0.01, news_per_day=6, seed=7, trend_symbols=0):
    rng = np.random.default_rng(seed)
    days = list(pd.bdate_range("2025-01-01", periods=n_days).date)
    syms = [f"S{i:02d}" for i in range(n_symbols)]
    # planted news: (day index, symbol, sign)
    planted = []
    for i in range(1, n_days - 2):
        for s in rng.choice(syms, size=news_per_day, replace=False):
            planted.append((i, s, int(rng.choice([-1, 1]))))
    boost = {(i, s): sign for i, s, sign in planted}
    rows = []
    for k, s in enumerate(syms):
        drift = 0.006 if k < trend_symbols else 0.0
        close = 50.0
        for i, d in enumerate(days):
            sign = boost.get((i, s), 0) or boost.get((i - 1, s), 0)
            if (i, s) in boost:
                sign = boost[(i, s)]
            elif (i - 1, s) in boost:
                sign = boost[(i - 1, s)]
            else:
                sign = 0
            o = close * (1 + rng.normal(0, 0.004))
            c = o * (1 + drift + rng.normal(0, 0.012) + (edge * sign if sign > 0 else 0))
            rows.append({"symbol": s, "day": d, "open": o, "high": max(o, c) * 1.002,
                         "low": min(o, c) * 0.998, "close": c, "volume": 1e6})
            close = c
    bars = pd.DataFrame(rows)
    news = []
    for n, (i, s, sign) in enumerate(planted):
        ts = datetime.combine(days[i], time(8, 0), NY)
        news.append({"published_at": pd.Timestamp(ts), "symbol": s, "score": sign, "confidence": 0.9,
                     "news_id": f"n{n}"})
    scores = pd.DataFrame(news)
    scores["published_at"] = pd.to_datetime(scores["published_at"], utc=True)
    return bars, scores, days
