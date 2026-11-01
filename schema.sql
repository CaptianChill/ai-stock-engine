-- AI Stock Engine schema. Idempotent: safe to run more than once.
-- Every row that feeds a decision carries a timestamp so the replay can prove no look-ahead.

create table if not exists bars (
  symbol      text        not null,
  day         date        not null,
  open        double precision not null,
  high        double precision not null,
  low         double precision not null,
  close       double precision not null,
  volume      double precision not null,
  primary key (symbol, day)
);

create table if not exists news (
  news_id      text primary key,
  published_at timestamptz not null,
  symbols      text[]      not null,
  headline     text        not null,
  summary      text,
  source       text,
  fetched_at   timestamptz not null default now()
);
create index if not exists news_published_idx on news (published_at);

-- One LLM verdict per (headline, symbol, model, prompt version). Re-scoring with a new
-- model creates new rows instead of overwriting, so old replays stay reproducible.
create table if not exists news_scores (
  news_id        text        not null references news(news_id),
  symbol         text        not null,
  model          text        not null,
  prompt_version text        not null,
  label          text        not null check (label in ('good','bad','irrelevant')),
  score          smallint    not null check (score in (-1,0,1)),
  confidence     double precision not null check (confidence between 0 and 1),
  scored_at      timestamptz not null default now(),
  primary key (news_id, symbol, model, prompt_version)
);

create table if not exists strategy_versions (
  version     text primary key,
  engine      text not null check (engine in ('news_reaction','momentum','combined')),
  params      jsonb not null,
  created_at  timestamptz not null default now()
);

create table if not exists replay_runs (
  run_id        bigserial primary key,
  version       text not null references strategy_versions(version),
  started_at    timestamptz not null default now(),
  period_start  date not null,
  period_end    date not null,
  cost_bps      double precision not null,
  metrics       jsonb not null,
  daily_returns jsonb not null,     -- {"YYYY-MM-DD": return}
  trades        jsonb not null      -- list of trade dicts
);

create table if not exists audit_results (
  audit_id    bigserial primary key,
  version     text not null references strategy_versions(version),
  run_id      bigint references replay_runs(run_id),
  gate        text not null,
  passed      boolean not null,
  detail      jsonb not null,
  audited_at  timestamptz not null default now()
);

-- Paper/live trading state ------------------------------------------------------
create table if not exists decisions (
  decision_id bigserial primary key,
  trade_day   date not null,
  engine      text not null,
  version     text not null,
  symbol      text not null,
  side        text not null check (side in ('buy','sell')),
  target_qty  double precision not null,
  reason      jsonb not null,
  risk_status text not null check (risk_status in ('approved','rejected')),
  risk_reason text,
  created_at  timestamptz not null default now()
);

create table if not exists orders (
  client_order_id text primary key,
  decision_id     bigint references decisions(decision_id),
  mode            text not null check (mode in ('paper','live')),
  symbol          text not null,
  side            text not null,
  qty             double precision not null,
  status          text not null,
  broker_order_id text,
  submitted_at    timestamptz not null default now(),
  filled_avg_price double precision,
  filled_qty      double precision,
  ref_price       double precision,   -- price the cost model assumed
  updated_at      timestamptz not null default now()
);

create table if not exists positions (
  symbol      text not null,
  engine      text not null,
  version     text not null,
  mode        text not null,
  qty         double precision not null,
  entry_day   date not null,
  exit_due    date,
  entry_price double precision,
  status      text not null check (status in ('open','closed')),
  opened_at   timestamptz not null default now(),
  closed_at   timestamptz,
  primary key (symbol, engine, entry_day, mode)
);

create table if not exists equity_snapshots (
  snap_day    date not null,
  mode        text not null,
  equity      double precision not null,
  cash        double precision,
  pnl_day     double precision,
  taken_at    timestamptz not null default now(),
  primary key (snap_day, mode)
);

-- Which scheduled stage ran on which day, so every stage is idempotent.
create table if not exists run_state (
  trade_day  date not null,
  stage      text not null,
  status     text not null,
  detail     jsonb,
  ran_at     timestamptz not null default now(),
  primary key (trade_day, stage)
);

-- Kill switch: a single row. Trading stops while engaged = true.
create table if not exists kill_switch (
  id          smallint primary key default 1 check (id = 1),
  engaged     boolean not null default false,
  reason      text,
  changed_at  timestamptz not null default now()
);
insert into kill_switch (id, engaged) values (1, false) on conflict (id) do nothing;

create table if not exists monitor_alerts (
  alert_id   bigserial primary key,
  raised_at  timestamptz not null default now(),
  kind       text not null,
  detail     jsonb not null
);
