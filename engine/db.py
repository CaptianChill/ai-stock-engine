"""Thin Postgres layer (psycopg 3). All SQL lives here so the agents stay testable."""
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
import psycopg
from psycopg.rows import dict_row

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema.sql"


class DB:
    def __init__(self, url: str):
        if not url:
            raise RuntimeError("DATABASE_URL is not set")
        self.url = url

    @contextmanager
    def conn(self) -> Iterator[psycopg.Connection]:
        with psycopg.connect(self.url, row_factory=dict_row) as c:
            yield c

    # -- schema --------------------------------------------------------------
    def init_schema(self) -> None:
        with self.conn() as c:
            c.execute(SCHEMA_PATH.read_text())

    # -- market data ----------------------------------------------------------
    def upsert_bars(self, rows: Iterable[dict]) -> int:
        rows = list(rows)
        if not rows:
            return 0
        with self.conn() as c, c.cursor() as cur:
            cur.executemany(
                """insert into bars (symbol, day, open, high, low, close, volume)
                   values (%(symbol)s, %(day)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s)
                   on conflict (symbol, day) do update set open=excluded.open, high=excluded.high,
                   low=excluded.low, close=excluded.close, volume=excluded.volume""",
                rows,
            )
        return len(rows)

    def load_bars(self, start: date, end: date) -> pd.DataFrame:
        with self.conn() as c:
            rows = c.execute(
                "select symbol, day, open, high, low, close, volume from bars "
                "where day between %s and %s order by day, symbol", (start, end)
            ).fetchall()
        return pd.DataFrame(rows, columns=["symbol", "day", "open", "high", "low", "close", "volume"])

    def bars_coverage(self) -> dict[str, tuple[date, date]]:
        """symbol -> (first day, last day) already stored. Lets downloads resume."""
        with self.conn() as c:
            rows = c.execute("select symbol, min(day) as lo, max(day) as hi from bars group by symbol").fetchall()
        return {r["symbol"]: (r["lo"], r["hi"]) for r in rows}

    def last_news_time(self, start: datetime, end: datetime) -> datetime | None:
        with self.conn() as c:
            r = c.execute("select max(published_at) as t from news where published_at > %s and published_at <= %s",
                          (start, end)).fetchone()
        return r["t"] if r else None

    def upsert_news(self, rows: Iterable[dict]) -> int:
        rows = list(rows)
        if not rows:
            return 0
        with self.conn() as c, c.cursor() as cur:
            cur.executemany(
                """insert into news (news_id, published_at, symbols, headline, summary, source)
                   values (%(news_id)s, %(published_at)s, %(symbols)s, %(headline)s, %(summary)s, %(source)s)
                   on conflict (news_id) do nothing""",
                rows,
            )
        return len(rows)

    def unscored_news(self, start: datetime, end: datetime, model: str, prompt_version: str,
                      symbols: set[str] | None = None) -> list[dict]:
        with self.conn() as c:
            rows = c.execute(
                """select n.news_id, n.published_at, s.symbol, n.headline, n.summary
                   from news n cross join lateral unnest(n.symbols) as s(symbol)
                   where n.published_at > %s and n.published_at <= %s
                     and not exists (select 1 from news_scores x where x.news_id = n.news_id
                                     and x.symbol = s.symbol and x.model = %s and x.prompt_version = %s)
                   order by n.published_at""",
                (start, end, model, prompt_version),
            ).fetchall()
        if symbols is not None:
            rows = [r for r in rows if r["symbol"] in symbols]
        return rows

    def insert_scores(self, rows: Iterable[dict]) -> int:
        rows = list(rows)
        if not rows:
            return 0
        with self.conn() as c, c.cursor() as cur:
            cur.executemany(
                """insert into news_scores (news_id, symbol, model, prompt_version, label, score, confidence)
                   values (%(news_id)s, %(symbol)s, %(model)s, %(prompt_version)s, %(label)s, %(score)s, %(confidence)s)
                   on conflict do nothing""",
                rows,
            )
        return len(rows)

    def load_scores(self, start: datetime, end: datetime, model: str, prompt_version: str) -> pd.DataFrame:
        with self.conn() as c:
            rows = c.execute(
                """select n.published_at, x.symbol, x.score, x.confidence, x.news_id
                   from news_scores x join news n using (news_id)
                   where n.published_at > %s and n.published_at <= %s and x.model = %s and x.prompt_version = %s
                   order by n.published_at""",
                (start, end, model, prompt_version),
            ).fetchall()
        return pd.DataFrame(rows, columns=["published_at", "symbol", "score", "confidence", "news_id"])

    # -- strategy, replay and audit -------------------------------------------
    def register_version(self, version: str, engine: str, params: dict) -> None:
        with self.conn() as c:
            c.execute(
                "insert into strategy_versions (version, engine, params) values (%s, %s, %s) "
                "on conflict (version) do nothing",
                (version, engine, json.dumps(params)),
            )

    def count_versions(self) -> int:
        with self.conn() as c:
            return c.execute("select count(*) as n from strategy_versions").fetchone()["n"]

    def save_replay(self, version: str, start: date, end: date, cost_bps: float, metrics: dict,
                    daily: pd.Series, trades: list[dict]) -> int:
        daily_json = {d.isoformat() if hasattr(d, "isoformat") else str(d): float(v) for d, v in daily.items()}
        with self.conn() as c:
            row = c.execute(
                """insert into replay_runs (version, period_start, period_end, cost_bps, metrics, daily_returns, trades)
                   values (%s, %s, %s, %s, %s, %s, %s) returning run_id""",
                (version, start, end, cost_bps, json.dumps(metrics), json.dumps(daily_json),
                 json.dumps(trades, default=str)),
            ).fetchone()
        return row["run_id"]

    def load_replay(self, run_id: int) -> dict:
        with self.conn() as c:
            return c.execute("select * from replay_runs where run_id = %s", (run_id,)).fetchone()

    def replays_for_period(self, start: date, end: date) -> list[dict]:
        with self.conn() as c:
            return c.execute(
                "select run_id, version, daily_returns from replay_runs where period_start = %s and period_end = %s",
                (start, end),
            ).fetchall()

    def save_audit(self, version: str, run_id: int | None, gate: str, passed: bool, detail: dict) -> None:
        with self.conn() as c:
            c.execute(
                "insert into audit_results (version, run_id, gate, passed, detail) values (%s, %s, %s, %s, %s)",
                (version, run_id, gate, passed, json.dumps(detail, default=str)),
            )

    def latest_gate_results(self, version: str) -> dict[str, bool]:
        with self.conn() as c:
            rows = c.execute(
                """select distinct on (gate) gate, passed from audit_results
                   where version = %s order by gate, audited_at desc""",
                (version,),
            ).fetchall()
        return {r["gate"]: r["passed"] for r in rows}

    # -- trading state --------------------------------------------------------
    def stage_done(self, day: date, stage: str) -> bool:
        with self.conn() as c:
            r = c.execute("select status from run_state where trade_day=%s and stage=%s", (day, stage)).fetchone()
        return bool(r and r["status"] == "done")

    def mark_stage(self, day: date, stage: str, status: str, detail: dict | None = None) -> None:
        with self.conn() as c:
            c.execute(
                """insert into run_state (trade_day, stage, status, detail) values (%s, %s, %s, %s)
                   on conflict (trade_day, stage) do update set status=excluded.status,
                   detail=excluded.detail, ran_at=now()""",
                (day, stage, status, json.dumps(detail or {}, default=str)),
            )

    def get_stage(self, day: date, stage: str) -> dict | None:
        with self.conn() as c:
            return c.execute("select * from run_state where trade_day=%s and stage=%s", (day, stage)).fetchone()

    def kill_switch(self) -> dict:
        with self.conn() as c:
            return c.execute("select engaged, reason, changed_at from kill_switch where id=1").fetchone()

    def set_kill_switch(self, engaged: bool, reason: str) -> None:
        with self.conn() as c:
            c.execute("update kill_switch set engaged=%s, reason=%s, changed_at=now() where id=1", (engaged, reason))

    def insert_decision(self, d: dict) -> int:
        with self.conn() as c:
            return c.execute(
                """insert into decisions (trade_day, engine, version, symbol, side, target_qty, reason, risk_status, risk_reason)
                   values (%(trade_day)s, %(engine)s, %(version)s, %(symbol)s, %(side)s, %(target_qty)s,
                   %(reason)s, %(risk_status)s, %(risk_reason)s) returning decision_id""",
                {**d, "reason": json.dumps(d["reason"], default=str)},
            ).fetchone()["decision_id"]

    def insert_order(self, o: dict) -> None:
        with self.conn() as c:
            c.execute(
                """insert into orders (client_order_id, decision_id, mode, symbol, side, qty, status, broker_order_id, ref_price)
                   values (%(client_order_id)s, %(decision_id)s, %(mode)s, %(symbol)s, %(side)s, %(qty)s,
                   %(status)s, %(broker_order_id)s, %(ref_price)s) on conflict do nothing""",
                o,
            )

    def open_orders(self, mode: str) -> list[dict]:
        with self.conn() as c:
            return c.execute(
                "select * from orders where mode=%s and status not in ('filled','canceled','expired','rejected')",
                (mode,),
            ).fetchall()

    def update_order(self, client_order_id: str, status: str, filled_avg_price, filled_qty) -> None:
        with self.conn() as c:
            c.execute(
                """update orders set status=%s, filled_avg_price=%s, filled_qty=%s, updated_at=now()
                   where client_order_id=%s""",
                (status, filled_avg_price, filled_qty, client_order_id),
            )

    def filled_orders(self, mode: str, since: date | None = None) -> list[dict]:
        q = "select * from orders where mode=%s and status='filled'"
        args: list = [mode]
        if since:
            q += " and submitted_at >= %s"
            args.append(since)
        with self.conn() as c:
            return c.execute(q, args).fetchall()

    def upsert_position(self, p: dict) -> None:
        with self.conn() as c:
            c.execute(
                """insert into positions (symbol, engine, version, mode, qty, entry_day, exit_due, entry_price, status)
                   values (%(symbol)s, %(engine)s, %(version)s, %(mode)s, %(qty)s, %(entry_day)s, %(exit_due)s,
                   %(entry_price)s, %(status)s)
                   on conflict (symbol, engine, entry_day, mode) do update set qty=excluded.qty,
                   exit_due=excluded.exit_due, entry_price=excluded.entry_price, status=excluded.status""",
                p,
            )

    def open_positions(self, mode: str, engine: str | None = None) -> list[dict]:
        q = "select * from positions where mode=%s and status='open'"
        args: list = [mode]
        if engine:
            q += " and engine=%s"
            args.append(engine)
        with self.conn() as c:
            return c.execute(q, args).fetchall()

    def close_position(self, symbol: str, engine: str, entry_day: date, mode: str) -> None:
        with self.conn() as c:
            c.execute(
                """update positions set status='closed', closed_at=now()
                   where symbol=%s and engine=%s and entry_day=%s and mode=%s""",
                (symbol, engine, entry_day, mode),
            )

    def snapshot_equity(self, day: date, mode: str, equity: float, cash: float | None, pnl_day: float | None) -> None:
        with self.conn() as c:
            c.execute(
                """insert into equity_snapshots (snap_day, mode, equity, cash, pnl_day) values (%s, %s, %s, %s, %s)
                   on conflict (snap_day, mode) do update set equity=excluded.equity, cash=excluded.cash,
                   pnl_day=excluded.pnl_day, taken_at=now()""",
                (day, mode, equity, cash, pnl_day),
            )

    def equity_history(self, mode: str) -> pd.DataFrame:
        with self.conn() as c:
            rows = c.execute(
                "select snap_day, equity, pnl_day from equity_snapshots where mode=%s order by snap_day", (mode,)
            ).fetchall()
        return pd.DataFrame(rows, columns=["snap_day", "equity", "pnl_day"])

    def get_decision(self, decision_id: int) -> dict | None:
        with self.conn() as c:
            return c.execute("select * from decisions where decision_id=%s", (decision_id,)).fetchone()

    def filled_orders_missing_ref(self, mode: str, day: date) -> list[dict]:
        with self.conn() as c:
            return c.execute(
                """select o.client_order_id, o.symbol, o.side, d.engine from orders o join decisions d
                   using (decision_id) where o.mode=%s and o.status='filled' and o.ref_price is null
                   and d.trade_day=%s""", (mode, day)).fetchall()

    def set_ref_price(self, client_order_id: str, ref: float) -> None:
        with self.conn() as c:
            c.execute("update orders set ref_price=%s where client_order_id=%s", (ref, client_order_id))

    def latest_replay(self, version: str) -> dict | None:
        with self.conn() as c:
            return c.execute("select run_id, metrics from replay_runs where version=%s order by run_id desc limit 1",
                             (version,)).fetchone()

    def count_stage_done(self, stage: str) -> int:
        with self.conn() as c:
            return c.execute("select count(*) as n from run_state where stage=%s and status='done'",
                             (stage,)).fetchone()["n"]

    def last_stage(self, stage: str) -> dict | None:
        with self.conn() as c:
            return c.execute("select status, detail from run_state where stage=%s order by ran_at desc limit 1",
                             (stage,)).fetchone()

    # -- review brain ------------------------------------------------------------
    def news_counts_by_day(self, start: date, end: date) -> pd.Series:
        with self.conn() as c:
            rows = c.execute(
                """select (published_at at time zone 'America/New_York')::date as d, count(*) as n from news
                   where published_at >= %s and published_at < %s + 1 group by 1 order by 1""", (start, end)).fetchall()
        return pd.Series({r["d"]: r["n"] for r in rows}, dtype=float)

    def recent_stages(self, since: date) -> list[dict]:
        with self.conn() as c:
            return c.execute("select trade_day, stage, status, detail from run_state where trade_day >= %s "
                             "order by trade_day, stage", (since,)).fetchall()

    def recent_alerts(self, since: date) -> list[dict]:
        with self.conn() as c:
            return c.execute("select raised_at, kind, detail from monitor_alerts where raised_at >= %s "
                             "order by raised_at desc limit 50", (since,)).fetchall()

    def decision_summary(self, day: date) -> dict:
        with self.conn() as c:
            rows = c.execute("""select engine, side, risk_status, count(*) as n from decisions where trade_day=%s
                                group by 1, 2, 3""", (day,)).fetchall()
        return {f"{r['engine']}/{r['side']}/{r['risk_status']}": r["n"] for r in rows}

    def save_brain_report(self, day: date, facts: dict, review: dict, model: str | None) -> int:
        with self.conn() as c:
            return c.execute(
                """insert into brain_reports (report_day, facts, review, model) values (%s, %s, %s, %s)
                   on conflict (report_day) do update set facts=excluded.facts, review=excluded.review,
                   model=excluded.model, created_at=now() returning report_id""",
                (day, json.dumps(facts, default=str), json.dumps(review, default=str), model)).fetchone()["report_id"]

    def insert_proposal(self, report_id: int, p: dict) -> int:
        with self.conn() as c:
            return c.execute(
                """insert into brain_proposals (report_id, param, old_value, new_value, rationale)
                   values (%s, %s, %s, %s, %s) returning proposal_id""",
                (report_id, p["param"], json.dumps(p["old_value"]), json.dumps(p["new_value"]), p["rationale"]),
            ).fetchone()["proposal_id"]

    def get_proposal(self, proposal_id: int) -> dict | None:
        with self.conn() as c:
            return c.execute("select * from brain_proposals where proposal_id=%s", (proposal_id,)).fetchone()

    def list_proposals(self, status: str | None = "pending") -> list[dict]:
        q, args = "select * from brain_proposals", []
        if status:
            q += " where status=%s"
            args.append(status)
        with self.conn() as c:
            return c.execute(q + " order by proposal_id", args).fetchall()

    def decide_proposal(self, proposal_id: int, status: str, candidate_version: str | None = None) -> None:
        with self.conn() as c:
            c.execute("""update brain_proposals set status=%s, candidate_version=%s, decided_at=now()
                         where proposal_id=%s""", (status, candidate_version, proposal_id))

    def latest_brain_report(self) -> dict | None:
        with self.conn() as c:
            return c.execute("select * from brain_reports order by report_day desc limit 1").fetchone()

    def version_params(self, version: str) -> dict | None:
        with self.conn() as c:
            r = c.execute("select params from strategy_versions where version=%s", (version,)).fetchone()
        return r["params"] if r else None

    def raise_alert(self, kind: str, detail: dict) -> None:
        with self.conn() as c:
            c.execute("insert into monitor_alerts (kind, detail) values (%s, %s)",
                      (kind, json.dumps(detail, default=str)))
