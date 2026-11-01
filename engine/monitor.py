"""Monitor agent: compares live/paper results with what the replay said to expect.

Raises an alert and flags the version for re-audit when:
  - live daily returns run more than 2 standard errors below the replay's mean, or
  - real fills slip more than the cost model assumed.
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np

from .config import StrategyVersion

MIN_LIVE_DAYS = 20
DRIFT_Z = -2.0


def drift_z(live: np.ndarray, replay_mean: float, replay_std: float) -> float | None:
    n = len(live)
    if n < 2 or replay_std <= 0:
        return None
    return float((np.mean(live) - replay_mean) / (replay_std / math.sqrt(n)))


def run_monitor(db, sv: StrategyVersion, mode: str) -> dict:
    hist = db.equity_history(mode)
    out: dict = {"live_days": int(len(hist))}
    run = db.latest_replay(sv.version)
    if run is None:
        out["note"] = "no replay on record for this version"
        return out
    if len(hist) >= MIN_LIVE_DAYS + 1:
        eq = hist["equity"].astype(float).values
        live = eq[1:] / eq[:-1] - 1
        z = drift_z(live[-60:], run["metrics"]["daily_mean"], run["metrics"]["daily_std"])
        out["drift_z"] = z
        if z is not None and z < DRIFT_Z:
            db.raise_alert("drift", {"version": sv.version, "z": z, "run_id": run["run_id"]})
            db.mark_stage(date.today(), "reaudit_required", "done", {"z": z})
            out["reaudit_required"] = True
    fills = db.filled_orders(mode)
    slips = [abs(o["filled_avg_price"] - o["ref_price"]) / o["ref_price"] * 1e4
             for o in fills if o.get("filled_avg_price") and o.get("ref_price")]
    if slips:
        out["mean_slippage_bps"] = float(np.mean(slips))
        if out["mean_slippage_bps"] > sv.costs.cost_bps_per_side:
            db.raise_alert("slippage", {"mean_bps": out["mean_slippage_bps"],
                                        "assumed_bps": sv.costs.cost_bps_per_side})
    return out
