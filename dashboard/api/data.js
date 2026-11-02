// Read-only data feed for Mission Control. Runs on Vercel as a serverless function.
// Env: DATABASE_URL (the read-only dash_ro connection string), DASHBOARD_PIN (6+ characters).
import { neon } from '@neondatabase/serverless';
import { timingSafeEqual, createHash } from 'node:crypto';

const h = (s) => createHash('sha256').update(String(s)).digest();
const pinOk = (given) => {
  const pin = process.env.DASHBOARD_PIN || '';
  if (pin.length < 6 || !given) return false;
  return timingSafeEqual(h(given), h(pin));
};

export default async function handler(req, res) {
  res.setHeader('Cache-Control', 'no-store');
  if (!process.env.DATABASE_URL || !process.env.DASHBOARD_PIN) {
    return res.status(503).json({ error: 'setup', message: 'DATABASE_URL and DASHBOARD_PIN are not set in Vercel yet.' });
  }
  if (!pinOk(req.headers['x-pin'])) {
    await new Promise((r) => setTimeout(r, 800)); // slow down guessing
    return res.status(401).json({ error: 'pin' });
  }
  const sql = neon(process.env.DATABASE_URL);
  try {
    const [progress, gates, runs, newsWeekly, spy, topNews, stages, kill, equity, positions, alerts, brain, proposals] =
      await Promise.all([
        sql`select
              (select reltuples::bigint from pg_class where relname = 'bars') as bars_est,
              (select count(distinct symbol) from bars where day = (select max(day) from bars)) as symbols_last_day,
              (select min(day) from bars) as bars_from, (select max(day) from bars) as bars_to,
              (select reltuples::bigint from pg_class where relname = 'news') as news_est,
              (select max(published_at) from news) as news_latest,
              (select count(*) from news_scores) as scored,
              (select max(scored_at) from news_scores) as scored_latest,
              (select count(*) from replay_runs) as replays,
              (select count(*) from audit_results) as audits`,
        sql`select distinct on (gate) gate, passed, detail, audited_at, version
              from audit_results order by gate, audited_at desc`,
        sql`select run_id, version, period_start, period_end, started_at, metrics
              from replay_runs order by run_id desc limit 5`,
        sql`select date_trunc('week', published_at)::date as wk, count(*)::int as n
              from news where published_at > now() - interval '26 weeks' group by 1 order by 1`,
        sql`select date_trunc('month', day)::date as m, (array_agg(close order by day desc))[1] as close
              from bars where symbol = 'SPY' and day > now() - interval '3 years' group by 1 order by 1`,
        sql`select s.symbol, s.label, s.confidence, n.headline, n.published_at
              from news_scores s join news n using (news_id)
              where s.label <> 'irrelevant' order by n.published_at desc limit 12`,
        sql`select trade_day, stage, status, ran_at from run_state order by ran_at desc limit 12`,
        sql`select engaged, reason, changed_at from kill_switch where id = 1`,
        sql`select snap_day, mode, equity, pnl_day from equity_snapshots order by snap_day desc limit 60`,
        sql`select symbol, engine, mode, qty, entry_day, exit_due, entry_price from positions
              where status = 'open' order by entry_day desc limit 30`,
        sql`select raised_at, kind, detail from monitor_alerts order by raised_at desc limit 10`,
        sql`select report_day, review, created_at from brain_reports order by report_day desc limit 1`,
        sql`select proposal_id, param, old_value, new_value, rationale, status, created_at
              from brain_proposals order by proposal_id desc limit 10`,
      ]);
    return res.status(200).json({
      now: new Date().toISOString(),
      progress: progress[0], gates, runs, newsWeekly, spy, topNews, stages,
      kill: kill[0] || null, equity, positions, alerts, brain: brain[0] || null, proposals,
    });
  } catch (e) {
    return res.status(500).json({ error: 'db', message: String(e.message || e).slice(0, 300) });
  }
}
