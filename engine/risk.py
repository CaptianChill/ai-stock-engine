"""Risk agent: the final veto. Sizes every order and rejects anything outside the limits.
Nothing upstream (including the LLM) can bypass it: execution only accepts its output."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import RiskParams, StrategyVersion


@dataclass
class OrderIntent:
    engine: str
    symbol: str
    side: str            # 'buy' or 'sell'
    ref_price: float     # latest price used for sizing
    target_notional: float = 0.0
    qty: int = 0         # for sells: the held quantity
    reason: dict | None = None


@dataclass
class RiskVerdict:
    approved: bool
    qty: int
    reason: str


def news_slot_notional(equity: float, sv: StrategyVersion) -> float:
    slot = equity * sv.news_weight / (sv.news.max_positions * sv.news.hold_days)
    return min(slot, equity * sv.risk.max_position_pct)


def momentum_slot_notional(equity: float, sv: StrategyVersion) -> float:
    slot = equity * sv.momentum_weight / sv.momentum.top_n
    return min(slot, equity * sv.risk.max_position_pct)


def check(intent: OrderIntent, *, equity: float, gross_exposure: float, kill_switch_engaged: bool,
          entries_halted: bool, held_symbols: dict[str, float], p: RiskParams) -> RiskVerdict:
    """gross_exposure: current market value of all long positions plus already-approved buys today."""
    if intent.side == "sell":
        held = held_symbols.get(intent.symbol, 0.0)
        if held <= 0:
            return RiskVerdict(False, 0, "nothing held to sell")
        if intent.qty <= 0:
            return RiskVerdict(False, 0, "sell quantity must be positive")
        # Exits are always allowed, even with the kill switch engaged: they reduce risk.
        return RiskVerdict(True, int(min(intent.qty, held)), "exit approved")

    if intent.side != "buy":
        return RiskVerdict(False, 0, f"unknown side {intent.side}")
    if kill_switch_engaged:
        return RiskVerdict(False, 0, "kill switch engaged")
    if entries_halted:
        return RiskVerdict(False, 0, "daily loss stop: no new entries today")
    if equity <= 0:
        return RiskVerdict(False, 0, "no equity")
    if not (intent.ref_price and math.isfinite(intent.ref_price)) or intent.ref_price < p.min_price:
        return RiskVerdict(False, 0, f"price below minimum {p.min_price}")
    # The cap is per stock across both engines: shares already held count against it.
    cap = equity * p.max_position_pct - held_symbols.get(intent.symbol, 0.0) * intent.ref_price
    notional = min(intent.target_notional, max(cap, 0.0))
    room = equity * p.max_gross_pct - gross_exposure
    notional = min(notional, max(room, 0.0))
    qty = int(notional // intent.ref_price)
    if qty <= 0:
        return RiskVerdict(False, 0, "size rounds to zero shares (or no gross room left)")
    return RiskVerdict(True, qty, "entry approved")
