"""Phone alerts through ntfy (https://ntfy.sh). Set NOTIFY_URL to the engine's own topic,
e.g. https://ntfy.sh/<topic>, and subscribe to that topic in the ntfy app.

Short, plain messages with an emoji for urgency. Alerts never break trading: any failure to
send is swallowed (and logged), and with NOTIFY_URL unset nothing is sent at all.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("engine.notify")

_PRIORITY = {"low": "2", "default": "3", "high": "4", "urgent": "5"}


def notify(title: str, message: str, priority: str = "default") -> bool:
    url = os.environ.get("NOTIFY_URL")
    if not url:
        return False
    try:
        import requests
        r = requests.post(url, data=message.encode("utf-8"), timeout=10,
                          headers={"Title": title.encode("utf-8"),
                                   "Priority": _PRIORITY.get(priority, "3")})
        return r.ok
    except Exception as e:  # alerts must never stop the engine
        log.warning("notify failed: %r", e)
        return False


def _money(x) -> str:
    try:
        return f"${float(x):,.0f}"
    except (TypeError, ValueError):
        return "?"


def stage_message(stage: str, d: dict | None, mode: str) -> tuple[str, str, str]:
    d = d or {}
    tag = "PAPER" if mode == "paper" else "LIVE"
    if stage == "premarket":
        picks = d.get("news_picks") or []
        n = d.get("orders_submitted", 0)
        if d.get("kill_switch"):
            return (f"🛑 {tag} kill switch on", "No new buys today.", "high")
        if d.get("entries_halted"):
            return (f"🟠 {tag} buys paused", "Yesterday's loss limit hit. Exits only today.", "high")
        body = f"{n} buy order(s) placed." + (f" Picks: {', '.join(picks[:10])}" if picks else " No strong news today.")
        return (f"🟢 {tag} morning orders", body, "default")
    if stage == "exits":
        out = d.get("exits_submitted") or []
        return (f"🔵 {tag} exits", f"Selling at close: {', '.join(out)}" if out else "Nothing to sell today.", "low")
    if stage == "close":
        pnl = float(d.get("pnl_day") or 0)
        emoji = "🟢" if pnl > 0 else ("⚪" if pnl == 0 else "🔴")
        lines = [f"Equity {_money(d.get('equity'))}  |  Day {'+' if pnl >= 0 else '-'}{_money(abs(pnl))}"]
        pri = "default"
        if d.get("entries_halted_next_day"):
            lines.append("Loss limit hit: no new buys tomorrow.")
            pri = "high"
        if d.get("kill_switch_engaged"):
            lines.append("KILL SWITCH ENGAGED (drawdown limit).")
            pri = "urgent"
        return (f"{emoji} {tag} close", "\n".join(lines), pri)
    return (f"{tag} {stage}", str(d)[:300], "default")


def brain_message(review: dict | None) -> tuple[str, str, str]:
    review = review or {}
    findings = review.get("findings") or []
    hi = [f for f in findings if str(f.get("severity", "")).lower() == "high"]
    title = "🔴 Nightly review: problems" if hi else "🧠 Nightly review"
    body = (review.get("summary") or "No summary.")[:500]
    if hi:
        body += "\n" + "\n".join(f"• {f.get('title')}" for f in hi[:3])
    return (title, body, "high" if hi else "low")
