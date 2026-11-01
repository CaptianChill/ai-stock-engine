"""Independent reproduction. Deliberately written without pandas pivots or any helper from
replay.py / strategies.py: it re-derives every trade straight from raw bar rows and must
agree with the replay to within floating-point noise."""
from __future__ import annotations

from collections import defaultdict
from datetime import date


def _d(x) -> date:
    return x if isinstance(x, date) else date.fromisoformat(str(x)[:10])


def reproduce(raw_bars: list[dict], trades: list[dict], cost_bps: float, hold_days: int,
              min_dollar_volume: float, min_price: float, tol: float = 1e-9) -> dict:
    by_key = {(b["symbol"], _d(b["day"])): b for b in raw_bars}
    days = sorted({_d(b["day"]) for b in raw_bars})
    pos = {d: i for i, d in enumerate(days)}
    hist = defaultdict(list)
    for b in sorted(raw_bars, key=lambda b: _d(b["day"])):
        hist[b["symbol"]].append(b)
    c = cost_bps / 1e4
    problems = []
    gross_by_day: dict[date, float] = defaultdict(float)
    for t in trades:
        sym, ed, xd = t["symbol"], _d(t["entry_day"]), _d(t["exit_day"])
        e, x = by_key.get((sym, ed)), by_key.get((sym, xd))
        if e is None or x is None:
            problems.append({"trade": t, "issue": "bar missing"})
            continue
        if t["engine"] == "news":
            if abs(e["open"] - t["entry_px"]) > tol:
                problems.append({"trade": t, "issue": "entry price is not the entry-day open"})
            if abs(x["close"] - t["exit_px"]) > tol:
                problems.append({"trade": t, "issue": "exit price is not the exit-day close"})
            if pos[xd] - pos[ed] != hold_days - 1:
                problems.append({"trade": t, "issue": "holding period mismatch"})
            net = x["close"] / e["open"] - 1 - 2 * c
            if abs(net - t["net"]) > 1e-7:
                problems.append({"trade": t, "issue": f"net return mismatch {net} vs {t['net']}"})
            # eligibility from the 20 bars strictly before entry
            prior = [b for b in hist[sym] if _d(b["day"]) < ed][-20:]
            if len(prior) < 20:
                problems.append({"trade": t, "issue": "fewer than 20 prior bars"})
            else:
                dv = sum(b["close"] * b["volume"] for b in prior) / 20
                if dv < min_dollar_volume or prior[-1]["close"] < min_price:
                    problems.append({"trade": t, "issue": "not eligible on entry day"})
            for k in range(pos[ed], pos[xd] + 1):
                gross_by_day[days[k]] += t.get("portfolio_weight", t.get("weight", 0))
    over = {str(d): w for d, w in gross_by_day.items() if w > 1 + 1e-9}
    if over:
        problems.append({"issue": "news engine gross exposure above 100%", "days": over})
    return {"checked": len(trades), "problems": problems[:50], "n_problems": len(problems)}
