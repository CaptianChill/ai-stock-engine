"""Scheduler tick for paper (and, once unlocked, live) trading.

Run `python -m engine tick` every ~10 minutes on weekdays. Each call checks the New York
clock and runs whichever stage is due, at most once per day (state is in run_state), so
missed or repeated ticks are harmless and daylight-saving changes don't matter.

Stages (New York time):
  premarket 08:45-09:25  ingest news + bars, LLM-score, pick, risk-check, submit market-on-open orders
  exits     14:30-15:45  submit market-on-close sells for news positions due to exit today
  close     16:30+       sync fills, snapshot equity, loss limits, monitor drift
"""
from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from . import risk as rk
from . import strategies as st
from .alpaca import DataClient
from .config import ACTIVE_VERSION, Settings, StrategyVersion
from .execution import client_order_id, trading_client
from .llm_scorer import Scorer
from .monitor import run_monitor

log = logging.getLogger("engine.runner")
NY = ZoneInfo("America/New_York")
EXCHANGES = {"NYSE", "NASDAQ", "ARCA", "AMEX", "BATS"}


def _trading_days(tc, start: date, end: date) -> list[date]:
    return [date.fromisoformat(c["date"]) for c in tc.calendar(start, end)]


class Runner:
    def __init__(self, settings: Settings, db, sv: StrategyVersion = ACTIVE_VERSION):
        self.s, self.db, self.sv = settings, db, sv
        self.mode = "live" if settings.is_live else "paper"
        self.tc = trading_client(settings, db, sv)
        self.dc = DataClient(settings.alpaca_key, settings.alpaca_secret)

    # -- entry point --------------------------------------------------------
    def tick(self, now: datetime | None = None, force_stage: str | None = None) -> dict:
        now = (now or datetime.now(tz=NY)).astimezone(NY)
        today = now.date()
        cal = _trading_days(self.tc, today - timedelta(days=30), today + timedelta(days=10))
        if today not in cal:
            return {"skipped": "not a trading day"}
        t = now.time()
        due = force_stage or (
            "premarket" if time(8, 45) <= t < time(9, 25) else
            "exits" if time(14, 30) <= t < time(15, 45) else
            "close" if t >= time(16, 30) else None)
        if not due:
            return {"skipped": f"no stage due at {t:%H:%M} ET"}
        lock = getattr(self.db, "stage_lock", None)
        if lock is None:
            return self._run_stage(today, due, cal, force_stage)
        with lock(today, due) as got:
            if not got:
                return {"skipped": f"{due} running elsewhere"}
            return self._run_stage(today, due, cal, force_stage)

    def _run_stage(self, today: date, due: str, cal, force_stage: str | None) -> dict:
        from .notify import notify, stage_message
        if self.db.stage_done(today, due) and not force_stage:
            return {"skipped": f"{due} already done"}
        fn = {"premarket": self.premarket, "exits": self.exits, "close": self.close}[due]
        try:
            detail = fn(today, cal)
            self.db.mark_stage(today, due, "done", detail)
            notify(*stage_message(due, detail, self.mode))
            return {"stage": due, **(detail or {})}
        except Exception as e:  # record and re-raise so the scheduler shows a failure
            self.db.mark_stage(today, due, "failed", {"error": repr(e)})
            notify(f"🔴 {due} FAILED", f"{type(e).__name__}: {str(e)[:300]}", "urgent")
            raise

    # -- helpers ------------------------------------------------------------
    def _universe_symbols(self) -> list[str]:
        return sorted(a["symbol"] for a in self.tc.assets()
                      if a.get("tradable") and a.get("exchange") in EXCHANGES and "." not in a["symbol"])

    def _refresh_bars(self, today: date) -> pd.DataFrame:
        start = today - timedelta(days=int(self.sv.momentum.lookback_days * 1.6) + 30)
        have = self.db.load_bars(today - timedelta(days=10), today - timedelta(days=1))
        fetch_from = today - timedelta(days=10) if len(have) else start
        # Keep to the stocks already tracked (the most-traded set from fetch-bars) so the
        # free database doesn't fill up; fall back to the whole market on a fresh database.
        tracked = sorted(self.db.bars_coverage())
        syms = tracked or self._universe_symbols()
        self.db.upsert_bars(self.dc.daily_bars(syms, fetch_from, today - timedelta(days=1)))
        return self.db.load_bars(start, today - timedelta(days=1))

    def _signals_today(self, today: date, cal: list[date], universe: set[str]) -> pd.DataFrame:
        prev = max(d for d in cal if d < today)
        since = datetime.combine(prev - timedelta(days=14), time(0), NY)
        until = datetime.combine(today, time(9, 15), NY)
        self.db.upsert_news(self.dc.news(since, until))
        rows = self.db.unscored_news(since, until, self.s.llm_model, self.s.llm_prompt_version, universe)
        if rows:
            scorer = Scorer(self.s.anthropic_key, self.s.llm_model, self.s.llm_prompt_version)
            self.db.insert_scores(scorer.score_rows(rows))
        scores = self.db.load_scores(since, until, self.s.llm_model, self.s.llm_prompt_version)
        days = [d for d in cal if d <= today]
        return st.news_signals(scores, days, self.sv.news)

    def _account(self) -> tuple[float, float]:
        a = self.tc.account()
        return float(a["equity"]), float(a.get("long_market_value") or 0.0)

    def _held(self) -> dict[str, float]:
        return {p["symbol"]: float(p["qty"]) for p in self.tc.positions()}

    def _submit(self, today, intent: rk.OrderIntent, verdict: rk.RiskVerdict, tif: str, extra: dict) -> None:
        dec_id = self.db.insert_decision({
            "trade_day": today, "engine": intent.engine, "version": self.sv.version, "symbol": intent.symbol,
            "side": intent.side, "target_qty": verdict.qty, "reason": {**(intent.reason or {}), **extra},
            "risk_status": "approved" if verdict.approved else "rejected", "risk_reason": verdict.reason})
        if not verdict.approved:
            return
        lot_key = intent.symbol if intent.side == "buy" else f"{intent.symbol}:{extra.get('entry_day')}"
        coid = client_order_id(today, intent.engine, lot_key, intent.side, self.mode)
        try:
            o = self.tc.submit_order(intent.symbol, verdict.qty, intent.side, tif, coid)
            status, bid = o.get("status", "submitted"), o.get("id")
        except Exception as e:
            status, bid = f"error: {e}"[:200], None
            log.error("order failed %s %s: %s", intent.symbol, intent.side, e)
        self.db.insert_order({"client_order_id": coid, "decision_id": dec_id, "mode": self.mode,
                              "symbol": intent.symbol, "side": intent.side, "qty": verdict.qty,
                              "status": status, "broker_order_id": bid, "ref_price": None})

    def _entries_halted(self, today: date) -> bool:
        return bool(self.db.get_stage(today, "entries_halted"))

    # -- stages -------------------------------------------------------------
    def premarket(self, today: date, cal: list[date]) -> dict:
        self.sync()
        bars = self._refresh_bars(today)
        sv = self.sv
        uni_news = st.eligible_next(bars, sv.news.universe_size, sv.news.min_dollar_volume, sv.risk.min_price)
        uni_mom = st.eligible_next(bars, sv.momentum.universe_size, sv.momentum.min_dollar_volume, sv.risk.min_price)
        signals = self._signals_today(today, cal, uni_news | uni_mom)
        picks = st.news_picks(signals, {today: uni_news}, sv.news)
        picks = picks[picks["trade_day"] == today]

        equity, gross = self._account()
        held = self._held()
        ks = self.db.kill_switch()["engaged"]
        halted = self._entries_halted(today)
        prices = self.dc.latest_prices(list(set(picks["symbol"]) | set(held)))
        future = [d for d in cal if d >= today]
        exit_due = future[sv.news.hold_days - 1] if len(future) >= sv.news.hold_days else None
        submitted = 0

        # Momentum rebalance on the first trading day of the month.
        mom_detail = None
        is_rebalance = today == min(d for d in cal if d.year == today.year and d.month == today.month)
        if sv.momentum_weight > 0 and is_rebalance:
            close = st.pivot(bars, "close")
            past_days = [d for d in cal if d <= today]
            veto_days = set(past_days[-sv.momentum.veto_window_days:])
            targets = st.rank_momentum(close, len(close), uni_mom, veto_days, signals, sv.momentum) or []
            current = {p["symbol"]: p for p in self.db.open_positions(self.mode, "momentum")}
            for sym, p in current.items():
                if sym not in targets:
                    it = rk.OrderIntent("momentum", sym, "sell", prices.get(sym, 0), qty=int(p["qty"]))
                    v = rk.check(it, equity=equity, gross_exposure=gross, kill_switch_engaged=ks,
                                 entries_halted=halted, held_symbols=held, p=sv.risk)
                    self._submit(today, it, v, "opg", {"rebalance": True, "entry_day": str(p["entry_day"])})
                    if v.approved:
                        submitted += 1
                        gross = max(gross - v.qty * prices.get(sym, 0), 0.0)  # sold at the same open
                        held[sym] = held.get(sym, 0) - v.qty
            new = [s for s in targets if s not in current]
            prices.update(self.dc.latest_prices(new) if new else {})
            for sym in new:
                it = rk.OrderIntent("momentum", sym, "buy", prices.get(sym, 0),
                                    target_notional=rk.momentum_slot_notional(equity, sv))
                v = rk.check(it, equity=equity, gross_exposure=gross, kill_switch_engaged=ks,
                             entries_halted=halted, held_symbols=held, p=sv.risk)
                self._submit(today, it, v, "opg", {"exit_due": None})
                if v.approved:
                    gross += v.qty * it.ref_price
                    held[sym] = held.get(sym, 0) + v.qty
                    submitted += 1
            mom_detail = {"targets": targets}

        for _, r in picks.iterrows():
            sym = r["symbol"]
            it = rk.OrderIntent("news", sym, "buy", prices.get(sym, 0),
                                target_notional=rk.news_slot_notional(equity, sv),
                                reason={"signal": float(r["signal"]), "n_items": int(r["n_items"])})
            v = rk.check(it, equity=equity, gross_exposure=gross, kill_switch_engaged=ks,
                         entries_halted=halted, held_symbols=held, p=sv.risk)
            self._submit(today, it, v, "opg", {"exit_due": str(exit_due) if exit_due else None})
            if v.approved:
                gross += v.qty * it.ref_price
                held[sym] = held.get(sym, 0) + v.qty
                submitted += 1
        return {"news_picks": picks["symbol"].tolist(), "orders_submitted": submitted,
                "momentum": mom_detail, "kill_switch": ks, "entries_halted": halted}

    def exits(self, today: date, cal: list[date]) -> dict:
        self.sync()
        held = self._held()
        out = []
        for p in self.db.open_positions(self.mode, "news"):
            if p["exit_due"] and p["exit_due"] <= today:
                it = rk.OrderIntent("news", p["symbol"], "sell", 0, qty=int(p["qty"]))
                v = rk.check(it, equity=1, gross_exposure=0, kill_switch_engaged=False, entries_halted=False,
                             held_symbols=held, p=self.sv.risk)
                self._submit(today, it, v, "cls", {"entry_day": str(p["entry_day"])})
                out.append(p["symbol"])
        return {"exits_submitted": out}

    def close(self, today: date, cal: list[date]) -> dict:
        self.sync()
        self._stamp_reference_prices(today)
        equity, _ = self._account()
        hist = self.db.equity_history(self.mode)
        prev = float(hist["equity"].iloc[-1]) if len(hist) else equity
        pnl = equity - prev
        self.db.snapshot_equity(today, self.mode, equity, None, pnl)
        detail = {"equity": equity, "pnl_day": pnl}
        nxt = min((d for d in cal if d > today), default=None)
        if prev > 0 and pnl / prev <= -self.sv.risk.daily_loss_stop_pct and nxt:
            self.db.mark_stage(nxt, "entries_halted", "done", {"after_loss_pct": pnl / prev})
            detail["entries_halted_next_day"] = True
        peak = max([equity] + hist["equity"].astype(float).tolist())
        if peak > 0 and (equity / peak - 1) <= -self.sv.risk.max_drawdown_halt_pct:
            self.db.set_kill_switch(True, f"drawdown {(equity / peak - 1):.1%} from peak")
            detail["kill_switch_engaged"] = True
        detail["monitor"] = run_monitor(self.db, self.sv, self.mode)
        return detail

    # -- bookkeeping ----------------------------------------------------------
    def sync(self) -> int:
        """Pull order status from the broker; turn fills into position rows."""
        n = 0
        for o in self.db.open_orders(self.mode):
            if o["status"].startswith("error"):
                continue
            b = self.tc.order_by_client_id(o["client_order_id"])
            st_ = b.get("status")
            fap = float(b["filled_avg_price"]) if b.get("filled_avg_price") else None
            fq = float(b["filled_qty"]) if b.get("filled_qty") else None
            self.db.update_order(o["client_order_id"], st_, fap, fq)
            if st_ != "filled":
                continue
            n += 1
            dec = self.db.get_decision(o["decision_id"])
            if o["side"] == "buy":
                exit_due = (dec["reason"] or {}).get("exit_due")
                self.db.upsert_position({"symbol": o["symbol"], "engine": dec["engine"], "version": dec["version"],
                                         "mode": self.mode, "qty": fq, "entry_day": dec["trade_day"],
                                         "exit_due": date.fromisoformat(exit_due) if exit_due else None,
                                         "entry_price": fap, "status": "open"})
            else:
                # Close only the lot this sell was for (the same stock may be held from two entry days).
                lot = (dec["reason"] or {}).get("entry_day")
                for p in self.db.open_positions(self.mode, dec["engine"]):
                    if p["symbol"] == o["symbol"] and (lot is None or str(p["entry_day"]) == lot):
                        self.db.close_position(p["symbol"], p["engine"], p["entry_day"], self.mode)
        return n

    def _stamp_reference_prices(self, today: date) -> None:
        """Reference = the official open (for market-on-open) or close (market-on-close) from the
        day's bar: exactly what the replay assumed. Slippage = fill vs this price."""
        rows = self.db.filled_orders_missing_ref(self.mode, today)
        if not rows:
            return
        bars = {b["symbol"]: b for b in self.dc.daily_bars(sorted({r["symbol"] for r in rows}), today, today)}
        for r in rows:
            b = bars.get(r["symbol"])
            if not b:
                continue
            is_close_order = r["side"] == "sell" and r["engine"] == "news"
            self.db.set_ref_price(r["client_order_id"], b["close"] if is_close_order else b["open"])
