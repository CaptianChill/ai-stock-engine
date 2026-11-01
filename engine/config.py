"""Central settings. Everything tunable lives here and is read from the environment.

Strategy parameters are frozen per strategy *version*: change a number, bump the version,
and the old version's replay and audit results stay attached to the old numbers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from datetime import date


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


@dataclass(frozen=True)
class NewsParams:
    universe_size: int = 500             # top N by 20-day dollar volume
    min_dollar_volume: float = 20e6      # skip thin names even if they make the top N
    max_positions: int = 10
    min_signal: float = 0.6              # mean(score*confidence) needed to enter
    hold_days: int = 2                   # exit at close of entry day + hold_days - 1
    entry_cutoff_hour_et: int = 9        # news must be published before 09:15 ET to count
    entry_cutoff_minute_et: int = 15


@dataclass(frozen=True)
class MomentumParams:
    universe_size: int = 500
    min_dollar_volume: float = 20e6
    lookback_days: int = 252
    skip_days: int = 21                  # skip the latest month (12-1 momentum)
    top_n: int = 15
    llm_veto: bool = True
    veto_threshold: float = -0.5         # mean news signal over veto window below this = drop
    veto_window_days: int = 10


@dataclass(frozen=True)
class RiskParams:
    max_position_pct: float = 0.10       # of equity, per name
    max_gross_pct: float = 1.00          # no leverage
    daily_loss_stop_pct: float = 0.03    # halt for the day after -3% equity
    max_drawdown_halt_pct: float = 0.15  # engage kill switch at -15% from peak
    allow_shorts: bool = False
    min_price: float = 5.0


@dataclass(frozen=True)
class CostParams:
    cost_bps_per_side: float = 5.0       # spread + slippage + fees, each way
    stress_multiplier: float = 2.0       # gate: must survive 2x costs


@dataclass(frozen=True)
class StrategyVersion:
    version: str
    news_weight: float = 0.6
    momentum_weight: float = 0.4
    news: NewsParams = field(default_factory=NewsParams)
    momentum: MomentumParams = field(default_factory=MomentumParams)
    risk: RiskParams = field(default_factory=RiskParams)
    costs: CostParams = field(default_factory=CostParams)

    def to_json(self) -> dict:
        return asdict(self)


# The active version. Bump the string whenever any parameter above changes.
ACTIVE_VERSION = StrategyVersion(version="v1.0.0")


@dataclass(frozen=True)
class Settings:
    database_url: str | None
    alpaca_key: str | None
    alpaca_secret: str | None
    anthropic_key: str | None
    llm_model: str
    llm_prompt_version: str
    # News dated on or before this day is not used to judge the LLM: the model may have
    # seen what happened next during training. Set it to the pinned model's documented
    # training-data cutoff. The default is deliberately late (conservative).
    llm_knowledge_cutoff: date
    trading_mode: str                     # 'paper' or 'live'
    live_confirm: str | None              # must equal LIVE_CONFIRM_PHRASE for live

    @property
    def is_live(self) -> bool:
        return self.trading_mode == "live"


LIVE_CONFIRM_PHRASE = "I accept real-money risk"


def load_settings() -> Settings:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    mode = (_env("TRADING_MODE", "paper") or "paper").lower()
    if mode not in ("paper", "live"):
        raise ValueError(f"TRADING_MODE must be 'paper' or 'live', got {mode!r}")
    return Settings(
        database_url=_env("DATABASE_URL"),
        alpaca_key=_env("ALPACA_API_KEY"),
        alpaca_secret=_env("ALPACA_API_SECRET"),
        anthropic_key=_env("ANTHROPIC_API_KEY"),
        llm_model=_env("LLM_MODEL", "claude-haiku-5-5"),
        llm_prompt_version=_env("LLM_PROMPT_VERSION", "p1"),
        llm_knowledge_cutoff=date.fromisoformat(_env("LLM_KNOWLEDGE_CUTOFF", "2026-07-01")),
        trading_mode=mode,
        live_confirm=_env("LIVE_CONFIRM"),
    )
