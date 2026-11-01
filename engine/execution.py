"""Execution agent. Paper by default. Live requires ALL of:
  1. TRADING_MODE=live
  2. LIVE_CONFIRM set to the exact confirmation phrase
  3. Every audit gate's latest result for the active strategy version is a pass
If any is missing, it refuses before any network call to the live endpoint."""
from __future__ import annotations

import hashlib
from datetime import date

from .alpaca import TradingClient
from .audit.gates import ALL_GATES
from .config import LIVE_CONFIRM_PHRASE, Settings, StrategyVersion


class LiveTradingBlocked(RuntimeError):
    pass


def live_blockers(settings: Settings, gate_results: dict[str, bool]) -> list[str]:
    reasons = []
    if not settings.is_live:
        reasons.append("TRADING_MODE is not 'live'")
    if settings.live_confirm != LIVE_CONFIRM_PHRASE:
        reasons.append("LIVE_CONFIRM phrase missing or wrong")
    for g in ALL_GATES:
        if gate_results.get(g) is not True:
            reasons.append(f"gate '{g}' has not passed")
    return reasons


def trading_client(settings: Settings, db, sv: StrategyVersion) -> TradingClient:
    if settings.is_live:
        blockers = live_blockers(settings, db.latest_gate_results(sv.version))
        if blockers:
            raise LiveTradingBlocked("; ".join(blockers))
        return TradingClient(settings.alpaca_key, settings.alpaca_secret, live=True)
    return TradingClient(settings.alpaca_key, settings.alpaca_secret, live=False)


def client_order_id(day: date, engine: str, symbol: str, side: str, mode: str) -> str:
    """Deterministic id: re-running a stage can never double-submit the same order."""
    raw = f"{mode}|{day.isoformat()}|{engine}|{symbol}|{side}"
    return f"ase-{day:%Y%m%d}-{hashlib.sha1(raw.encode()).hexdigest()[:16]}"
