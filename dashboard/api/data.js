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
  // Each query stands alone: one failing (e.g. a table the read-only login can't see) blanks
  // only its own panel instead of the whole page.
  const q = {
    equity: sql`select snap_day, equity, cash, pnl_day from equity_snapshots
                where mode = 'paper' order by snap_day`,
    holdings: sql`select trade_day, detail, ran_at from run_state where stage = 'holdings'
                  order by trade_day desc limit 1`,
    rebalance: sql`select trade_day, detail, ran_at from run_state where stage = 'momentum_rebalanced'
                   order by trade_day desc limit 1`,
    positions: sql`select symbol, engine, version, qty, entry_day, entry_price from positions
                   where status = 'open' and mode = 'paper' order by symbol`,
    orders: sql`select o.symbol, o.side, o.qty, o.status, o.filled_avg_price, o.filled_qty, o.submitted_at,
                       d.trade_day, d.engine, d.version, d.risk_status, d.risk_reason, d.reason
                from orders o left join decisions d using (decision_id)
                where o.mode = 'paper' order by o.submitted_at desc limit 40`,
    rejected: sql`select trade_day, symbol, side, risk_reason from decisions
                  where risk_status = 'rejected' order by created_at desc limit 10`,
    spyTrend: sql`select (array_agg(close order by day desc))[1] as last, avg(close) as avg200,
                         count(*)::int as n, max(day) as asof
                  from (select day, close from bars where symbol = 'SPY' order by day desc limit 200) s`,
    spy: sql`select day, close from bars where symbol = 'SPY'
             and day >= coalesce((select min(snap_day) from equity_snapshots where mode = 'paper'), now()::date - 90)
             order by day`,
    stages: sql`select trade_day, stage, status, ran_at, detail from run_state
                where stage in ('premarket', 'exits', 'close') order by ran_at desc limit 15`,
    kill: sql`select engaged, reason, changed_at from kill_switch where id = 1`,
    alerts: sql`select raised_at, kind, detail from monitor_alerts order by raised_at desc limit 8`,
    gates: sql`select distinct on (gate) gate, passed, audited_at, version
               from audit_results order by gate, audited_at desc`,
    runs: sql`select run_id, version, period_start, period_end, started_at, metrics
              from replay_runs order by run_id desc limit 5`,
    brain: sql`select report_day, review, created_at from brain_reports order by report_day desc limit 1`,
  };
  const keys = Object.keys(q);
  const settled = await Promise.allSettled(Object.values(q));
  const out = { now: new Date().toISOString(), errors: {} };
  settled.forEach((r, i) => {
    if (r.status === 'fulfilled') out[keys[i]] = r.value;
    else { out[keys[i]] = null; out.errors[keys[i]] = String(r.reason?.message || r.reason).slice(0, 200); }
  });
  for (const k of ['holdings', 'rebalance', 'kill', 'spyTrend', 'brain']) out[k] = out[k]?.[0] || null;
  if (keys.every((k) => out.errors[k])) {
    return res.status(500).json({ error: 'db', message: Object.values(out.errors)[0] });
  }
  return res.status(200).json(out);
}
