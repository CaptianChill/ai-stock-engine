"""Simulated trading days: the real Runner against an in-memory database and a fake broker
that fills market-on-open orders at the official open and market-on-close at the close."""
from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from engine.audit.gates import run_paper_gates
from engine.config import Settings
from engine.runner import Runner
from tests.synth import make_market, make_version

NY = ZoneInfo("America/New_York")


class FakeDB:
    def __init__(self, bars: pd.DataFrame, scores: pd.DataFrame):
        self.bars = bars.copy()
        self.scores = scores.copy()
        self.state, self.decisions, self.orders, self.positions = {}, {}, {}, {}
        self.equity, self.alerts, self.ks = [], [], {"engaged": False, "reason": None, "changed_at": None}

    # market data
    def load_bars(self, start, end):
        b = self.bars
        return b[(b["day"] >= start) & (b["day"] <= end)].reset_index(drop=True)

    def upsert_bars(self, rows): return len(list(rows))
    def upsert_news(self, rows): return len(list(rows))
    def unscored_news(self, *a, **k): return []
    def insert_scores(self, rows): return 0

    def load_scores(self, start, end, model, pv):
        s = self.scores
        t = pd.to_datetime(s["published_at"], utc=True)
        return s[(t > pd.Timestamp(start)) & (t <= pd.Timestamp(end))].reset_index(drop=True)

    # state
    def stage_done(self, day, stage): return self.state.get((day, stage), {}).get("status") == "done"
    def mark_stage(self, day, stage, status, detail=None): self.state[(day, stage)] = {"status": status, "detail": detail}
    def get_stage(self, day, stage): return self.state.get((day, stage))
    def count_stage_done(self, stage): return sum(1 for (d, s), v in self.state.items() if s == stage and v["status"] == "done")

    def last_stage(self, stage):
        rows = [v for (d, s), v in sorted(self.state.items(), key=lambda kv: kv[0][0]) if s == stage]
        return rows[-1] if rows else None

    def kill_switch(self): return dict(self.ks)
    def set_kill_switch(self, engaged, reason): self.ks.update(engaged=engaged, reason=reason)

    # trading
    def insert_decision(self, d):
        i = len(self.decisions) + 1
        self.decisions[i] = {**d, "decision_id": i}
        return i

    def get_decision(self, i): return self.decisions[i]
    def insert_order(self, o): self.orders.setdefault(o["client_order_id"], {**o, "filled_avg_price": None, "filled_qty": None})

    def open_orders(self, mode):
        return [o for o in self.orders.values()
                if o["mode"] == mode and o["status"] not in ("filled", "canceled", "expired", "rejected")]

    def update_order(self, coid, status, fap, fq): self.orders[coid].update(status=status, filled_avg_price=fap, filled_qty=fq)
    def filled_orders(self, mode, since=None): return [o for o in self.orders.values() if o["mode"] == mode and o["status"] == "filled"]

    def filled_orders_missing_ref(self, mode, day):
        return [{**o, "engine": self.decisions[o["decision_id"]]["engine"]} for o in self.filled_orders(mode)
                if o["ref_price"] is None and self.decisions[o["decision_id"]]["trade_day"] == day]

    def set_ref_price(self, coid, ref): self.orders[coid]["ref_price"] = ref
    def upsert_position(self, p): self.positions[(p["symbol"], p["engine"], p["entry_day"], p["mode"])] = dict(p)

    def open_positions(self, mode, engine=None):
        return [p for p in self.positions.values() if p["mode"] == mode and p["status"] == "open"
                and (engine is None or p["engine"] == engine)]

    def close_position(self, symbol, engine, entry_day, mode): self.positions[(symbol, engine, entry_day, mode)]["status"] = "closed"
    def snapshot_equity(self, day, mode, equity, cash, pnl): self.equity.append({"snap_day": day, "equity": equity, "pnl_day": pnl})
    def equity_history(self, mode): return pd.DataFrame(self.equity, columns=["snap_day", "equity", "pnl_day"])
    def latest_replay(self, version): return None
    def raise_alert(self, kind, detail): self.alerts.append((kind, detail))


class FakeBroker:
    """Plays both Alpaca clients. `self.today` is the simulated session date."""

    def __init__(self, bars: pd.DataFrame, days: list[date], cash=100_000.0):
        self.bars = bars.set_index(["symbol", "day"]).sort_index()
        self.days, self.cash, self.today = days, cash, None
        self.orders, self.held = {}, {}

    def _bar(self, sym, day): return self.bars.loc[(sym, day)]

    # trading client
    def calendar(self, start, end): return [{"date": d.isoformat()} for d in self.days if start <= d <= end]
    def assets(self): return [{"symbol": s, "tradable": True, "exchange": "NYSE"} for s in self.bars.index.get_level_values(0).unique()]

    def account(self):
        mv = sum(q * self._bar(s, self.today)["open"] for s, q in self.held.items())
        return {"equity": self.cash + mv, "long_market_value": mv}

    def positions(self): return [{"symbol": s, "qty": q} for s, q in self.held.items() if q > 0]

    def submit_order(self, symbol, qty, side, tif, coid):
        assert tif in ("opg", "cls")
        if coid in self.orders:
            raise RuntimeError("duplicate client_order_id")
        self.orders[coid] = {"symbol": symbol, "qty": qty, "side": side, "tif": tif, "day": self.today, "status": "accepted"}
        return {"id": f"b-{coid}", "status": "accepted"}

    def fill_all(self):
        for o in self.orders.values():
            if o["status"] == "filled":
                continue
            px = float(self._bar(o["symbol"], o["day"])["open" if o["tif"] == "opg" else "close"])
            sign = 1 if o["side"] == "buy" else -1
            self.cash -= sign * o["qty"] * px
            self.held[o["symbol"]] = self.held.get(o["symbol"], 0) + sign * o["qty"]
            o.update(status="filled", px=px)

    def order_by_client_id(self, coid):
        o = self.orders[coid]
        return {"status": o["status"], "filled_avg_price": o.get("px"), "filled_qty": o["qty"] if o["status"] == "filled" else None}

    # data client
    def daily_bars(self, symbols, start, end):
        for s in symbols:
            for d in self.days:
                if start <= d <= end and (s, d) in self.bars.index:
                    b = self._bar(s, d)
                    yield {"symbol": s, "day": d, "open": b["open"], "high": b["high"], "low": b["low"],
                           "close": b["close"], "volume": b["volume"]}

    def news(self, *a, **k): return iter(())

    def latest_prices(self, symbols):
        prev = max(d for d in self.days if d < self.today)
        return {s: float(self._bar(s, prev)["close"]) for s in symbols if (s, prev) in self.bars.index}


def make_runner(bars, scores, days, today_idx):
    sv = make_version(momentum_weight=0.0, news_weight=1.0)
    db = FakeDB(bars, scores)
    br = FakeBroker(bars, days)
    r = Runner.__new__(Runner)
    r.s = Settings(None, None, None, None, "m", "p1", date(2020, 1, 1), "paper", None)
    r.db, r.sv, r.mode, r.tc, r.dc = db, sv, "paper", br, br
    return r, db, br


def at(day, hh, mm):
    return datetime.combine(day, time(hh, mm), NY)


@pytest.fixture
def setup():
    bars, scores, days = make_market(n_days=200)
    k = 150
    return make_runner(bars, scores, days, k) + (days, k)


def test_full_round_trip_paper_day(setup):
    r, db, br, days, k = setup
    d0, d1 = days[k], days[k + 1]

    br.today = d0
    out = r.tick(at(d0, 9, 0))
    assert out["stage"] == "premarket" and out["orders_submitted"] > 0
    assert out["orders_submitted"] <= r.sv.news.max_positions
    # every approved buy is sized within the per-position cap
    for o in br.orders.values():
        assert o["qty"] * br.latest_prices([o["symbol"]])[o["symbol"]] <= 100_000 * r.sv.risk.max_position_pct + 1e-6
    # second tick in the same window is a no-op
    assert "already done" in r.tick(at(d0, 9, 10))["skipped"]

    br.fill_all()
    out = r.tick(at(d0, 16, 45))
    assert out["stage"] == "close"
    n_buys = sum(1 for o in br.orders.values() if o["side"] == "buy")
    assert len(db.open_positions("paper", "news")) == n_buys
    assert db.equity and db.equity[-1]["snap_day"] == d0
    # reference prices were stamped from the official open, so paper slippage is zero here
    assert all(o["ref_price"] is not None for o in db.filled_orders("paper"))

    br.today = d1
    out = r.tick(at(d1, 15, 0))
    assert out["stage"] == "exits"
    opened = {k_[0] for k_ in db.positions if k_[2] == d0}
    assert set(out["exits_submitted"]) == opened
    br.fill_all()
    r.tick(at(d1, 16, 45))
    assert all(p["status"] == "closed" for p in db.positions.values() if p["entry_day"] == d0)
    assert all(q == 0 for q in br.held.values())


def test_kill_switch_blocks_entries_live_runner(setup):
    r, db, br, days, k = setup
    db.set_kill_switch(True, "test")
    br.today = days[k]
    out = r.tick(at(days[k], 9, 0))
    assert out["orders_submitted"] == 0
    assert all(d["risk_status"] == "rejected" and d["risk_reason"] == "kill switch engaged"
               for d in db.decisions.values())


def test_non_trading_day_and_off_hours_are_skipped(setup):
    r, db, br, days, k = setup
    br.today = days[k]
    assert "not a trading day" in r.tick(datetime(2025, 7, 5, 9, 0, tzinfo=NY))["skipped"]   # Saturday
    assert "no stage due" in r.tick(at(days[k], 12, 0))["skipped"]


def test_paper_gates_after_simulated_days(setup):
    r, db, br, days, k = setup
    for i in range(k, k + 6):
        br.today = days[i]
        r.tick(at(days[i], 9, 0)); br.fill_all()
        r.tick(at(days[i], 15, 0)); br.fill_all()
        r.tick(at(days[i], 16, 45))
    db.mark_stage(days[k], "kill_switch_test", "done", {})
    gates = {g.gate: g for g in run_paper_gates(db, r.sv)}
    assert gates["paper_run_completed"].passed
    assert gates["paper_match"].passed, gates["paper_match"].detail
    assert gates["kill_switch_tested"].passed


def test_same_stock_two_lots_closes_only_the_exiting_lot(setup):
    r, db, br, days, k = setup
    d0, d1, d2 = days[k], days[k + 1], days[k + 2]
    sym = "S00"
    for d in (d0, d1):
        db.upsert_position({"symbol": sym, "engine": "news", "version": "test", "mode": "paper", "qty": 10,
                            "entry_day": d, "exit_due": d1 if d == d0 else d2, "entry_price": 50.0, "status": "open"})
    br.held[sym] = 20
    br.today = d1
    out = r.tick(at(d1, 15, 0))
    assert out["exits_submitted"] == [sym]
    br.fill_all()
    r.sync()
    status = {p["entry_day"]: p["status"] for p in db.positions.values() if p["symbol"] == sym}
    assert status == {d0: "closed", d1: "open"}
    assert br.held[sym] == 10


def test_momentum_rebalance_buys_then_rotates():
    bars, scores, days = make_market(n_days=260, trend_symbols=5, edge=0.0, seed=5)
    sv = make_version(news_weight=0.0, momentum_weight=1.0)
    firsts = [d for i, d in enumerate(days) if i > 0 and d.month != days[i - 1].month and i > 130]
    m1, m2 = firsts[0], firsts[1]
    r, db, br = make_runner(bars, scores, days, 0)[:3]
    r.sv = sv
    br.today = m1
    out = r.tick(at(m1, 9, 0))
    assert out["momentum"] and len(out["momentum"]["targets"]) == sv.momentum.top_n
    buys = [o for o in br.orders.values() if o["side"] == "buy"]
    assert {o["symbol"] for o in buys} == set(out["momentum"]["targets"])
    br.fill_all(); r.tick(at(m1, 16, 45))
    assert len(db.open_positions("paper", "momentum")) == sv.momentum.top_n
    # next month: names leaving the target list are sold at the open, new ones bought
    br.today = m2
    out2 = r.tick(at(m2, 9, 0))
    t1, t2 = set(out["momentum"]["targets"]), set(out2["momentum"]["targets"])
    sells = {o["symbol"] for o in br.orders.values() if o["side"] == "sell" and o["day"] == m2}
    new_buys = {o["symbol"] for o in br.orders.values() if o["side"] == "buy" and o["day"] == m2}
    assert sells == t1 - t2 and new_buys == t2 - t1
    br.fill_all(); r.tick(at(m2, 16, 45))
    assert {p["symbol"] for p in db.open_positions("paper", "momentum")} == t2
