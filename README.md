# AI Stock Engine

An LLM reads company news; rules decide what trades. Two engines, one risk agent with the final
say, and audit gates that keep it on paper until the evidence is in.

- **Fast engine (60%)**: the LLM labels each overnight headline good / bad / irrelevant with a
  confidence. Strongest positive names (liquid stocks only) are bought market-on-open and sold
  market-on-close after 2 days.
- **Core engine (40%)**: monthly 12-1 momentum; the LLM vetoes names with clearly bad recent news.
- **Risk agent**: sizes every order, caps positions (10%) and gross exposure (100%, no leverage),
  stops new entries for a day after a 3% daily loss, engages the kill switch at a 15% drawdown.
  Exits are always allowed.
- **Paper by default.** Live trading needs `TRADING_MODE=live`, `LIVE_CONFIRM="I accept real-money risk"`,
  and every audit gate's latest result passing. The scheduled workflow is hard-wired to paper.

Not financial advice. Passing tests proves the machinery is correct, not that the strategy makes money.

## Review brain (runs on its own every weekday night)

After the close, the *review-brain* workflow runs with no one involved:

1. **Data watchdog**: missing bars, thin news feed, failed stages, stuck or errored orders.
2. **Scorer check**: over the last 30 days, did headlines the LLM called "good" beat the ones it
   called "bad"? If not, it raises an urgent alert: the scorer has stopped working.
3. **Trading review**: fills, slippage vs the cost model, equity, alerts, gate status.
4. **Written review + proposals**: Claude writes findings and may propose up to 3 parameter changes,
   only from an allow-list with hard bounds (risk limits can only be tightened).

The brain **cannot** place, size or cancel orders, change the active strategy, or release the kill
switch (a test fails if its code ever imports the order path). Its output goes to `brain_reports`,
`brain_proposals` and urgent items to `monitor_alerts`.

**Proposals wait for you.** Actions → *review-brain* → Run → action `approve` (or `reject`) + the
proposal number. Approving only creates a *candidate* version (e.g. `v1.0.0-p3`) for testing:
`python -m engine replay START END --version v1.0.0-p3`, then `audit` the run. It trades only if it
passes every gate and you switch `ACTIVE_VERSION` in `engine/config.py` to it.

## Running on your Windows PC (current setup)

The schedule runs on your own PC through Windows Task Scheduler (GitHub's scheduled runs are off).

1. Open **PowerShell** and run:
   ```
   irm https://raw.githubusercontent.com/CaptianChill/ai-stock-engine/main/windows/setup.ps1 -OutFile $env:TEMP\ase-setup.ps1; powershell -ExecutionPolicy Bypass -File $env:TEMP\ase-setup.ps1
   ```
   It installs Python 3.12 and Git if needed, downloads the code to `%USERPROFILE%\ai-stock-engine`,
   asks for your four keys (hidden typing, saved only in `.env` on this PC), runs the tests, checks
   the database, and registers two tasks:
   - **AI Stock Engine - Trader**: every 10 min, weekdays 6:30 AM-5:30 PM (wakes the PC).
   - **AI Stock Engine - Brain**: weekdays 6:15 PM.
2. Start the first validation run: `%USERPROFILE%\ai-stock-engine\windows\research.cmd` (a few hours).
3. Check on things any time: `windows\status.cmd`. Logs are in the `logs` folder.

The PC must be on, or asleep with wake timers allowed, on weekday mornings and afternoons.

## Setup on GitHub Actions (alternative)

1. **Push this folder to a new private GitHub repo** (e.g. `CaptianChill/ai-stock-engine`).
2. **Accounts/keys you need:**
   - Alpaca account → Paper Trading → API keys (free).
   - Anthropic API key (for the LLM scorer).
   - Database: already created — Neon project `ai-stock-engine` (`rapid-king-35176996`), schema applied.
     Copy its connection string from the Neon console.
3. **GitHub repo → Settings → Secrets and variables → Actions:**
   - Secrets: `DATABASE_URL`, `ALPACA_API_KEY`, `ALPACA_API_SECRET`, `ANTHROPIC_API_KEY`
   - Variables (optional): `LLM_MODEL` (default `claude-haiku-5-5`), `LLM_KNOWLEDGE_CUTOFF`
     (set to the pinned model's published training cutoff; default `2026-07-01`), `ALPACA_DATA_FEED`,
     `BRAIN_MODEL` (default `claude-sonnet-5-5`).

## The compressed validation run

1. **Replay + audit (hours):** Actions → *replay-and-audit* → Run, start `2026-07-02` (after the LLM
   cutoff), end = last trading day. Use `max_llm_calls` for a cheap first pass. The log ends with
   PASS/FAIL per gate and the kill-switch self-test.
2. **The 24-hour paper run** starts as soon as the repo and secrets are in place: *trading-tick* runs
   every 10 minutes on weekdays (paper trading never needs the gates; only live does). Premarket (08:45–09:25 ET) scores news and places market-on-open orders;
   14:30–15:45 ET places market-on-close exits; after 16:30 ET it syncs fills, records equity, and
   runs the monitor. To run a stage by hand: *trading-tick* → Run → stage.
3. **After at least one full paper day:** run `python -m engine paper-gates` (locally with a `.env`, or
   add it as a step). That checks fills against the official open/close the replay assumed.
4. `python -m engine status` shows every gate, the kill switch and whether live is locked.

If replay gates fail, the strategy version does not get real money. Change parameters in
`engine/config.py`, **bump the version string**, and re-run. Every version tried counts against the
overfitting gate (deflated Sharpe), so tuning until it passes gets harder on purpose.

## Audit gates

| Gate | Passes when |
| --- | --- |
| no_lookahead | replay starts after the LLM cutoff; every signal predates 09:15 ET on its trade day |
| cost_stress | still profitable at 2× assumed costs (10 bps per side) |
| out_of_sample | last 40% of the period is profitable with frozen parameters |
| overfitting | deflated Sharpe ≥ 0.95 given every version tried; PBO ≤ 0.5 once ≥2 variants exist |
| decay_haircut | still profitable with half the measured edge |
| concentration | no stock > 25% of profit, no day > 25% of profit |
| min_sample | ≥ 60 days and ≥ 100 news trades in the replay window |
| momentum_history | base momentum (no LLM) since 2016: ≥ 60 rebalances, profitable at 2× costs, last 40% profitable, deflated Sharpe ≥ 0.95 |
| independent_reproduction | a separate implementation re-derives every trade from raw bars and agrees |
| paper_run_completed | at least one full paper trading day |
| paper_match | ≥ 5 fills; average fill vs official open/close within assumed cost |
| kill_switch_tested | self-test proved entries blocked, exits allowed, loss stop works |

## Known limits (read before trusting any number)

- **Short LLM history.** News on or before the model's training cutoff can't be used (the model may
  have seen the outcome), so the news engine's replay covers only a few months. Momentum is judged
  separately on price history since 2016 (no LLM involved, so no cutoff problem); the LLM veto on top
  of it is only judged in the short window.
- **Survivorship bias.** History is fetched for currently listed stocks; delisted names are missing,
  which flatters momentum results.
- **Data feeds.** Price history uses Alpaca's SIP feed (full market volume; free plans get it except
  the latest 15 minutes) and falls back to IEX if refused. Live snapshots use IEX.
- Entry/exit use market-on-open/close auctions so paper fills are comparable to the replay.

## Commands

```
python -m engine init-db
python -m engine fetch-bars 2016-01-01 2026-10-30
python -m engine fetch-history 2026-07-02 2026-10-30
python -m engine score-news 2026-07-02 2026-10-30 --max-calls 5000
python -m engine replay 2026-07-02 2026-10-30
python -m engine audit <run_id>
python -m engine test-kill-switch
python -m engine paper-gates
python -m engine tick [--stage premarket|exits|close]
python -m engine status
python -m engine kill "reason" | release
python -m engine brain | report | proposals [--all] | approve ID | reject ID
python -m pytest -q
```
