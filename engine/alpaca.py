"""Alpaca REST clients: market data (bars, news, snapshots) and trading (orders, account).

Plain `requests` instead of the SDK keeps the surface small and the calls auditable.
Paper and live use the same code; only the base URL differs, and execution.py decides
which one is allowed.
"""
from __future__ import annotations

import os
import time
from datetime import date, datetime, timezone
from typing import Iterator

import requests

DATA_URL = "https://data.alpaca.markets"
PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"


class AlpacaError(RuntimeError):
    pass


class _Base:
    def __init__(self, key: str | None, secret: str | None, timeout: float = 30.0):
        if not key or not secret:
            raise AlpacaError("ALPACA_API_KEY and ALPACA_API_SECRET must be set")
        self.s = requests.Session()
        self.s.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
        self.timeout = timeout

    def _req(self, method: str, url: str, **kw) -> dict | list:
        for attempt in range(5):
            r = self.s.request(method, url, timeout=self.timeout, **kw)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            if r.status_code >= 400:
                raise AlpacaError(f"{method} {url} -> {r.status_code}: {r.text[:500]}")
            return r.json() if r.content else {}
        raise AlpacaError(f"{method} {url} failed after retries")


class DataClient(_Base):
    def __init__(self, key, secret, feed: str | None = None):
        super().__init__(key, secret)
        self.feed = feed or os.environ.get("ALPACA_DATA_FEED", "iex")

    def daily_bars(self, symbols: list[str], start: date, end: date, feed: str | None = None) -> Iterator[dict]:
        """Split- and dividend-adjusted daily bars, chunked 100 symbols per request.
        History defaults to the SIP feed (full market volume; free plans get it except the
        latest 15 minutes) and falls back to IEX if the account isn't allowed SIP."""
        feed = feed or os.environ.get("ALPACA_HISTORY_FEED", "sip")
        for i in range(0, len(symbols), 100):
            chunk = symbols[i:i + 100]
            params = {
                "symbols": ",".join(chunk), "timeframe": "1Day", "start": start.isoformat(),
                "end": end.isoformat(), "limit": 10000, "adjustment": "all", "feed": feed,
            }
            while True:
                try:
                    j = self._req("GET", f"{DATA_URL}/v2/stocks/bars", params=params)
                except AlpacaError as e:
                    if params["feed"] != "iex" and ("403" in str(e) or "subscription" in str(e).lower()):
                        feed = params["feed"] = "iex"
                        continue
                    raise
                for sym, bars in (j.get("bars") or {}).items():
                    for b in bars:
                        yield {"symbol": sym, "day": b["t"][:10], "open": b["o"], "high": b["h"],
                               "low": b["l"], "close": b["c"], "volume": b["v"]}
                tok = j.get("next_page_token")
                if not tok:
                    break
                params["page_token"] = tok

    def news(self, start: datetime, end: datetime, symbols: list[str] | None = None) -> Iterator[dict]:
        params = {"start": start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                  "end": end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                  "limit": 50, "sort": "asc", "include_content": "false"}
        if symbols:
            params["symbols"] = ",".join(symbols)
        while True:
            j = self._req("GET", f"{DATA_URL}/v1beta1/news", params=params)
            for n in j.get("news", []):
                if not n.get("symbols"):
                    continue
                yield {"news_id": str(n["id"]), "published_at": n["created_at"], "symbols": n["symbols"],
                       "headline": n.get("headline") or "", "summary": n.get("summary") or None,
                       "source": n.get("source")}
            tok = j.get("next_page_token")
            if not tok:
                break
            params["page_token"] = tok

    def latest_prices(self, symbols: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for i in range(0, len(symbols), 100):
            j = self._req("GET", f"{DATA_URL}/v2/stocks/snapshots",
                          params={"symbols": ",".join(symbols[i:i + 100]), "feed": self.feed})
            for sym, snap in j.items():
                trade = (snap or {}).get("latestTrade") or {}
                bar = (snap or {}).get("dailyBar") or (snap or {}).get("prevDailyBar") or {}
                p = trade.get("p") or bar.get("c")
                if p:
                    out[sym] = float(p)
        return out


class TradingClient(_Base):
    def __init__(self, key, secret, live: bool):
        super().__init__(key, secret)
        self.base = LIVE_URL if live else PAPER_URL
        self.live = live

    def account(self) -> dict:
        return self._req("GET", f"{self.base}/v2/account")

    def assets(self) -> list[dict]:
        return self._req("GET", f"{self.base}/v2/assets", params={"status": "active", "asset_class": "us_equity"})

    def calendar(self, start: date, end: date) -> list[dict]:
        return self._req("GET", f"{self.base}/v2/calendar", params={"start": start.isoformat(), "end": end.isoformat()})

    def submit_order(self, symbol: str, qty: int, side: str, time_in_force: str, client_order_id: str) -> dict:
        """time_in_force 'opg' = market-on-open, 'cls' = market-on-close. These match the replay's
        entry-at-open / exit-at-close prices, so paper results are comparable to the replay."""
        body = {"symbol": symbol, "qty": str(int(qty)), "side": side, "type": "market",
                "time_in_force": time_in_force, "client_order_id": client_order_id}
        return self._req("POST", f"{self.base}/v2/orders", json=body)

    def order_by_client_id(self, client_order_id: str) -> dict:
        return self._req("GET", f"{self.base}/v2/orders:by_client_order_id",
                         params={"client_order_id": client_order_id})

    def positions(self) -> list[dict]:
        return self._req("GET", f"{self.base}/v2/positions")
