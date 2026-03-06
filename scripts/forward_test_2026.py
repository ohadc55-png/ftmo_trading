"""Forward Test: 2026 YTD (Jan 1 - Today)
Combines DB data (full warmup) + Yahoo Finance (fresh data to today)
Strategy V3: BRK+MTF+VOL | 07:00-23:59 | th=5.0 | 2c(1TP+1R) | SL<=25 | SmartDL$750 | RR 5.0
"""
import sys, sqlite3, time, html, json
from pathlib import Path

project_root = Path(r"c:\Ohad\ohad\אפליקציות\ftmo")
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf

from src.utils.config import load_config
from src.utils.timeutils import add_session_flags
from src.engines.mtf_trend import compute_mtf_trend
from src.engines.volume import compute_volume
from src.engines.key_levels import compute_key_levels
from src.engines.vwap import compute_vwap
from src.backtest.simulator import simulate_trades, trades_to_dataframe

# ─── 1) Load DB data for full indicator warmup ──────────────────────────
print("Loading DB data (2025-2026 for warmup)...")
t0 = time.time()
conn = sqlite3.connect(str(project_root / "nq_data.db"))
df_1m_db = pd.read_sql_query(
    "SELECT datetime, open, high, low, close, volume FROM ohlcv_1m "
    "WHERE datetime >= '2025-01-01' ORDER BY datetime", conn)
conn.close()
df_1m_db["datetime"] = pd.to_datetime(df_1m_db["datetime"])
df_1m_db = df_1m_db.set_index("datetime")
if df_1m_db.index.tz is None:
    df_1m_db.index = df_1m_db.index.tz_localize("UTC")
df_1m_db.index = df_1m_db.index.tz_convert("America/New_York")
for col in ["open", "high", "low", "close", "volume"]:
    df_1m_db[col] = pd.to_numeric(df_1m_db[col], errors="coerce")
df_1m_db = df_1m_db.dropna(subset=["open", "high", "low", "close"])
df_1m_db["volume"] = df_1m_db["volume"].fillna(0)

db_max = df_1m_db.index.max()
print(f"  DB: {len(df_1m_db):,} 1m bars, up to {db_max}")

# ─── 2) Download Yahoo Finance data (fresh, up to today) ────────────────
print("Downloading fresh Yahoo Finance data (60d, 5min)...")
ticker = yf.Ticker("NQ=F")
yf_5m = ticker.history(period="60d", interval="5m")
yf_5m = yf_5m.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
yf_5m = yf_5m[["open", "high", "low", "close", "volume"]].copy()
if yf_5m.index.tz is None:
    yf_5m.index = yf_5m.index.tz_localize("UTC")
yf_5m.index = yf_5m.index.tz_convert("America/New_York")
for col in ["open", "high", "low", "close", "volume"]:
    yf_5m[col] = pd.to_numeric(yf_5m[col], errors="coerce")
yf_5m = yf_5m.dropna(subset=["open", "high", "low", "close"])
yf_5m["volume"] = yf_5m["volume"].fillna(0)

yf_max = yf_5m.index.max()
print(f"  Yahoo: {len(yf_5m):,} 5m bars, up to {yf_max}")

# ─── 3) Combine: resample DB to 5m, then append fresh Yahoo data ────────
print("Combining DB + Yahoo Finance data...")
df_5m_db = df_1m_db.resample("5min").agg({
    "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
}).dropna(subset=["open", "high", "low", "close"])

# Only take Yahoo data that's AFTER the DB ends
yf_new = yf_5m[yf_5m.index > db_max]
print(f"  DB 5m bars: {len(df_5m_db):,}")
print(f"  Yahoo new bars (after {db_max.strftime('%Y-%m-%d')}): {len(yf_new):,}")

df_5m = pd.concat([df_5m_db, yf_new])
df_5m = df_5m[~df_5m.index.duplicated(keep='last')]
df_5m = df_5m.sort_index()
print(f"  Combined: {len(df_5m):,} 5m bars")
print(f"  Range: {df_5m.index[0]} to {df_5m.index[-1]}")

# ─── 4) Run Pipeline ────────────────────────────────────────────────────
cfg = load_config(project_root / "config" / "settings.yaml")
ecfg = cfg.engines; bt = cfg.backtest; capital = 100_000.0

print("Running pipeline...")
t0 = time.time()
df = df_5m.copy()
df = add_session_flags(df, tz=cfg.timezone,
    premarket_start_hour=ecfg.key_levels.premarket_start_hour,
    premarket_end_hour=ecfg.key_levels.premarket_end_hour,
    rth_start_hour=ecfg.vwap.rth_start_hour,
    rth_start_minute=ecfg.vwap.rth_start_minute)
df = compute_mtf_trend(df, ecfg.mtf_trend, cfg.mtf_timeframes)
df = compute_volume(df, ecfg.volume)
df = compute_key_levels(df, ecfg.key_levels)
df = compute_vwap(df, ecfg.vwap)
df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=bt.atr_period)
print(f"  Pipeline: {time.time()-t0:.0f}s")

# ─── 5) Composite scoring (same as V3) ──────────────────────────────────
mtf_arr = df["mtf_score"].fillna(0).values
vol_arr = df["vol_score"].fillna(0).values
lvl_arr = df["levels_score"].fillna(0).values
direction_arr = df["mtf_direction"].fillna("neutral").values
bull_mask_arr = (direction_arr == "bull")
bear_mask_arr = (direction_arr == "bear")
hour_arr = df.index.hour.values

engines_list = ["MTF", "VOL", "BRK"]
scores_map = {"MTF": mtf_arr, "VOL": vol_arr, "BRK": lvl_arr}
sw = np.array([3, 2, 3], dtype=float)
bw = np.array([3, 1, 4], dtype=float)
sell_score = sum(scores_map[e] * sw[i] for i, e in enumerate(engines_list)) / sw.sum()
sell_score = np.clip(sell_score, 0, 10)
buy_score = sum(scores_map[e] * bw[i] for i, e in enumerate(engines_list)) / bw.sum()
buy_score = np.clip(buy_score, 0, 10)

threshold = 5.0
exclude_hours = [0, 1, 2, 3, 4, 5, 6]

buy_elig = (buy_score >= threshold) & bull_mask_arr
sell_elig = (sell_score >= threshold) & bear_mask_arr
hour_mask = np.ones(len(df), dtype=bool)
for h in exclude_hours:
    hour_mask &= (hour_arr != h)
buy_elig = buy_elig & hour_mask
sell_elig = sell_elig & hour_mask

def apply_cooldown(eligible, scores, th):
    READY, FIRED, COOLING = 0, 1, 2
    n = len(eligible); signals = np.zeros(n, dtype=bool); state = READY
    for i in range(n):
        if state == READY:
            if eligible[i]: state = FIRED; signals[i] = True
        elif state == FIRED:
            if scores[i] < th: state = COOLING
        elif state == COOLING:
            if eligible[i]: state = FIRED; signals[i] = True
    return signals

sig_buy = apply_cooldown(buy_elig, buy_score, threshold)
sig_sell = apply_cooldown(sell_elig, sell_score, threshold)

df["composite_score"] = sell_score
bm = sig_buy.astype(bool)
if bm.any():
    df.loc[df.index[bm], "composite_score"] = buy_score[bm]
df["signal_buy"] = sig_buy
df["signal_sell"] = sig_sell

# ─── 6) Simulate with RR 5.0 ────────────────────────────────────────────
RR_RATIO = 5.0
MAX_SL = 25
SMART_DL = 750
rp = bt.risk_pct / 100 if bt.risk_pct > 1 else bt.risk_pct

trades = simulate_trades(df, starting_capital=capital,
    point_value=bt.point_value, sl_atr_mult=bt.sl_atr_mult,
    rr_ratio=RR_RATIO, risk_pct=rp,
    max_bars_held=bt.max_bars_held,
    use_runner=True, tp_contracts=1, runner_contracts=1,
    trail_atr_mult=bt.trail_atr_mult, runner_max_bars=bt.runner_max_bars,
    max_contracts=4)
tdf_raw = trades_to_dataframe(trades)

# SL<=25
tdf_raw = tdf_raw[tdf_raw["sl_distance"] <= MAX_SL].reset_index(drop=True)
print(f"  {len(tdf_raw):,} trades after SL<={MAX_SL} filter (all dates)")

# SmartDL
tdf_raw["trade_date"] = tdf_raw["entry_time"].dt.date
keep = []
for date, group in tdf_raw.groupby("trade_date"):
    daily_pnl = 0
    for idx, row in group.iterrows():
        if daily_pnl <= -SMART_DL:
            break
        potential_loss = row["sl_distance"] * 40
        remaining_budget = SMART_DL + daily_pnl
        if potential_loss > remaining_budget:
            continue
        keep.append(idx)
        daily_pnl += row["pnl_dollars"]
tdf_all = tdf_raw.loc[keep].reset_index(drop=True)

# Recalc account
account = capital
for i in range(len(tdf_all)):
    tdf_all.loc[tdf_all.index[i], "account_before"] = account
    account += tdf_all.loc[tdf_all.index[i], "pnl_dollars"]
    tdf_all.loc[tdf_all.index[i], "account_after"] = account

tdf_all["year"] = tdf_all["entry_time"].dt.year
tdf_all["month"] = tdf_all["entry_time"].dt.month
tdf_all["hour"] = tdf_all["entry_time"].dt.hour
tdf_all["dow"] = tdf_all["entry_time"].dt.dayofweek

# ─── 7) Filter to 2026 only ─────────────────────────────────────────────
tdf = tdf_all[tdf_all["year"] == 2026].reset_index(drop=True)

# Recalc account for 2026 view
account = capital
for i in range(len(tdf)):
    tdf.loc[tdf.index[i], "account_before"] = account
    account += tdf.loc[tdf.index[i], "pnl_dollars"]
    tdf.loc[tdf.index[i], "account_after"] = account

tdf["hour"] = tdf["entry_time"].dt.hour
tdf["dow"] = tdf["entry_time"].dt.dayofweek
tdf["trade_date2"] = tdf["entry_time"].dt.date

print(f"\n{'='*60}")
print(f"  2026 YTD FORWARD TEST (Jan 1 - {tdf['entry_time'].max().strftime('%b %d')})")
print(f"{'='*60}")
print(f"  Trades: {len(tdf)}")

if len(tdf) == 0:
    print("  No trades found!")
    sys.exit(1)

# Stats
wins = (tdf["pnl_dollars"] > 0).sum()
losses = len(tdf) - wins
wr = wins / len(tdf) * 100
total_pnl = tdf["pnl_dollars"].sum()
avg_pnl = tdf["pnl_dollars"].mean()
avg_win = tdf.loc[tdf["pnl_dollars"] > 0, "pnl_dollars"].mean() if wins > 0 else 0
avg_loss = tdf.loc[tdf["pnl_dollars"] <= 0, "pnl_dollars"].mean() if losses > 0 else 0
best = tdf["pnl_dollars"].max()
worst = tdf["pnl_dollars"].min()
gp = tdf.loc[tdf["pnl_dollars"] > 0, "pnl_dollars"].sum() if wins > 0 else 0
gl = abs(tdf.loc[tdf["pnl_dollars"] <= 0, "pnl_dollars"].sum()) if losses > 0 else 0
pf = gp / gl if gl > 0 else float("inf")
expectancy = (wr / 100 * avg_win) + ((1 - wr / 100) * avg_loss)

equity = [capital]
for _, row in tdf.iterrows():
    equity.append(equity[-1] + row["pnl_dollars"])
eq = pd.Series(equity)
peak = eq.expanding().max()
dd = eq - peak
max_dd = dd.min()
peak_val = eq[:dd.idxmin() + 1].max()
max_dd_pct = (max_dd / peak_val * 100) if peak_val > 0 else 0
final_account = equity[-1]

signs = (tdf["pnl_dollars"] > 0).astype(int).values
max_w = max_l = cur_w = cur_l = 0
for s_val in signs:
    if s_val == 1: cur_w += 1; cur_l = 0
    else: cur_l += 1; cur_w = 0
    max_w = max(max_w, cur_w); max_l = max(max_l, cur_l)

avg_bars = tdf["bars_held"].mean()

daily_pnl_series = tdf.groupby("trade_date2")["pnl_dollars"].sum()
worst_day = daily_pnl_series.min()
best_day = daily_pnl_series.max()
trading_days = len(daily_pnl_series)
winning_days = (daily_pnl_series > 0).sum()
losing_days = (daily_pnl_series < 0).sum()

buy_df = tdf[tdf["direction"] == "buy"]
sell_df = tdf[tdf["direction"] == "sell"]

# Monthly breakdown
monthly_data = []
for m in sorted(tdf["month"].unique()):
    mdf = tdf[tdf["month"] == m]
    mn = len(mdf); mw = (mdf["pnl_dollars"] > 0).sum()
    mwr = mw / mn * 100 if mn > 0 else 0
    mpnl = mdf["pnl_dollars"].sum()
    mgp = mdf.loc[mdf["pnl_dollars"] > 0, "pnl_dollars"].sum()
    mgl = abs(mdf.loc[mdf["pnl_dollars"] <= 0, "pnl_dollars"].sum())
    mpf = mgp / mgl if mgl > 0 else 0
    monthly_data.append({"month": m, "trades": mn, "wins": mw, "wr": mwr, "pnl": mpnl, "pf": mpf})

first_date = tdf["entry_time"].min().strftime("%b %d")
last_date = tdf["entry_time"].max().strftime("%b %d")

print(f"  Win Rate: {wr:.1f}%")
print(f"  Total P&L: ${total_pnl:+,.0f}")
print(f"  Profit Factor: {pf:.2f}")
print(f"  Max DD: ${max_dd:+,.0f}")
print(f"  Final Account: ${final_account:,.0f}")
print(f"  Trading Days: {trading_days} ({winning_days}W / {losing_days}L)")

# V3 backtest benchmarks (per month avg)
v3_monthly_avg_pnl = 1_280_993 / (10.1 * 12)
v3_monthly_avg_trades = 6094 / (10.1 * 12)
v3_avg_wr = 39.4

# ═══════════════════════════════════════════════════════════════════════════
# Generate HTML
# ═══════════════════════════════════════════════════════════════════════════
print("\nGenerating HTML report...")

def fmt_money(val):
    if val >= 0: return f"+${val:,.0f}"
    return f"-${abs(val):,.0f}"

def pnl_class(val):
    if val > 0: return "positive"
    elif val < 0: return "negative"
    return ""

def outcome_badge(outcome):
    o = outcome.upper()
    if "SL" in o: cls = "badge-sl"
    elif "TP" in o:
        cls = "badge-runner" if any(x in outcome.lower() for x in ["trail", "be", "timeout"]) else "badge-tp"
    elif "timeout" in o.lower(): cls = "badge-timeout"
    else: cls = "badge-tp"
    return f'<span class="badge {cls}">{html.escape(outcome)}</span>'

def direction_badge(direction):
    if direction == "buy": return '<span class="badge badge-buy">LONG</span>'
    return '<span class="badge badge-sell">SHORT</span>'

def heatmap_class(val, max_abs):
    if max_abs == 0 or val == 0: return ""
    ratio = abs(val) / max_abs
    if val > 0:
        return "heatmap-green-3" if ratio > 0.66 else ("heatmap-green-2" if ratio > 0.33 else "heatmap-green-1")
    elif val < 0:
        return "heatmap-red-3" if ratio > 0.66 else ("heatmap-red-2" if ratio > 0.33 else "heatmap-red-1")
    return ""

# Daily equity for chart
daily_eq = []
eq_val = capital
for dt, dpnl in daily_pnl_series.items():
    eq_val += dpnl
    ddf_day = tdf[tdf["trade_date2"] == dt]
    daily_eq.append({
        "date": str(dt), "equity": round(eq_val, 2), "pnl": round(dpnl, 2),
        "trades": len(ddf_day), "wins": int((ddf_day["pnl_dollars"] > 0).sum()),
    })
daily_eq_json = json.dumps(daily_eq)

months_traded = len(tdf["month"].unique())
total_months_pnl = total_pnl
annualized_pnl = total_pnl / months_traded * 12 if months_traded > 0 else 0

# Determine verdict
if total_pnl > 0 and pf > 1.3:
    verdict_cls = "pass"; verdict_icon = "PASS"; verdict_color = "var(--green)"
elif total_pnl > 0:
    verdict_cls = "neutral"; verdict_icon = "MARGINAL"; verdict_color = "var(--orange)"
else:
    verdict_cls = "fail"; verdict_icon = "UNDERPERFORM"; verdict_color = "var(--red)"

CSS = """:root {
    --bg: #0d1117; --card-bg: #161b22; --border: #30363d;
    --text: #e6edf3; --text-muted: #8b949e;
    --green: #3fb950; --red: #f85149; --blue: #58a6ff;
    --orange: #d29922; --purple: #bc8cff; --accent: #1f6feb; --gold: #ffd700;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; background: var(--bg); color: var(--text); line-height: 1.5; padding: 20px; }
.container { max-width: 1500px; margin: 0 auto; }
h1 { font-size: 28px; font-weight: 700; margin-bottom: 8px; color: var(--blue); }
h2 { font-size: 20px; font-weight: 600; margin: 30px 0 16px; color: var(--text); border-bottom: 1px solid var(--border); padding-bottom: 8px; }
h3 { font-size: 16px; font-weight: 600; margin: 20px 0 10px; color: var(--text-muted); }
.subtitle { color: var(--text-muted); font-size: 14px; margin-bottom: 24px; }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }
.stat-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
.stat-card .label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
.stat-card .value { font-size: 22px; font-weight: 700; }
.stat-card .value.positive { color: var(--green); }
.stat-card .value.negative { color: var(--red); }
.stat-card .value.neutral { color: var(--blue); }
.stat-card .compare { font-size: 11px; margin-top: 4px; }
.stat-card .compare.up { color: var(--green); }
.stat-card .compare.down { color: var(--red); }
table { width: 100%; border-collapse: collapse; background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; margin-bottom: 24px; font-size: 13px; }
thead { background: #1c2129; }
th { padding: 10px 12px; text-align: left; font-weight: 600; color: var(--text-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid var(--border); white-space: nowrap; cursor: pointer; user-select: none; }
th:hover { color: var(--blue); }
td { padding: 8px 12px; border-bottom: 1px solid var(--border); white-space: nowrap; }
tr:last-child td { border-bottom: none; }
tbody tr:hover { background: #1c2129; }
.positive { color: var(--green); }
.negative { color: var(--red); }
.neutral { color: var(--blue); }
.muted { color: var(--text-muted); }
.text-right { text-align: right; }
.font-mono { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }
.badge-buy { background: rgba(63, 185, 80, 0.2); color: var(--green); }
.badge-sell { background: rgba(248, 81, 73, 0.2); color: var(--red); }
.badge-tp { background: rgba(63, 185, 80, 0.2); color: var(--green); }
.badge-sl { background: rgba(248, 81, 73, 0.2); color: var(--red); }
.badge-timeout { background: rgba(210, 153, 34, 0.2); color: var(--orange); }
.badge-runner { background: rgba(88, 166, 255, 0.2); color: var(--blue); }
.equity-chart { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 24px; position: relative; }
#chartTooltip { position: absolute; background: #1c2129; border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; font-size: 12px; pointer-events: none; z-index: 100; display: none; white-space: nowrap; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }
#chartTooltip .tt-date { color: var(--blue); font-weight: 600; margin-bottom: 4px; }
#chartTooltip .tt-row { display: flex; justify-content: space-between; gap: 16px; }
#chartTooltip .tt-label { color: var(--text-muted); }
#chartCanvas { cursor: crosshair; }
.verdict-box { background: var(--card-bg); border-radius: 12px; padding: 24px; margin: 24px 0; }
.verdict-box.pass { border: 2px solid var(--green); }
.verdict-box.fail { border: 2px solid var(--red); }
.verdict-box.neutral { border: 2px solid var(--orange); }
.verdict-title { font-size: 20px; font-weight: 700; margin-bottom: 12px; }
.verdict-text { font-size: 15px; line-height: 1.8; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.heatmap-cell { font-weight: 600; font-size: 12px; padding: 6px 8px; }
.heatmap-green-1 { background: rgba(63, 185, 80, 0.15); }
.heatmap-green-2 { background: rgba(63, 185, 80, 0.3); }
.heatmap-green-3 { background: rgba(63, 185, 80, 0.45); }
.heatmap-red-1 { background: rgba(248, 81, 73, 0.15); }
.heatmap-red-2 { background: rgba(248, 81, 73, 0.3); }
.heatmap-red-3 { background: rgba(248, 81, 73, 0.45); }
.tab-nav { display: flex; gap: 0; border-bottom: 1px solid var(--border); margin-bottom: 20px; flex-wrap: wrap; }
.tab-btn { padding: 10px 20px; cursor: pointer; border: none; background: none; color: var(--text-muted); font-size: 14px; border-bottom: 2px solid transparent; transition: all 0.2s; }
.tab-btn:hover { color: var(--text); }
.tab-btn.active { color: var(--blue); border-bottom-color: var(--blue); }
.tab-content { display: none; }
.tab-content.active { display: block; }
.filter-bar { display: flex; gap: 12px; margin-bottom: 16px; align-items: center; flex-wrap: wrap; }
.filter-bar select { background: var(--card-bg); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; font-size: 13px; }
.filter-bar label { color: var(--text-muted); font-size: 12px; text-transform: uppercase; }
.trade-count { color: var(--text-muted); font-size: 13px; }
.footer { margin-top: 40px; padding: 20px 0; border-top: 1px solid var(--border); color: var(--text-muted); font-size: 12px; text-align: center; }
@media (max-width: 900px) { .two-col { grid-template-columns: 1fr; } }
"""

month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Forward Test 2026 YTD - SmartDL V3</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">

<h1>Forward Test: 2026 YTD <span style="font-size:16px;color:var(--text-muted)">(DB + Yahoo Finance Live Data)</span></h1>
<p class="subtitle">Strategy V3: BRK+MTF+VOL | 07:00-23:59 | th=5.0 | 2c (1TP+1R) | SL &le; {MAX_SL} pts | Smart DL ${SMART_DL} | RR 1:{RR_RATIO}<br>
Data: SQLite DB (warmup) + Yahoo Finance NQ=F (fresh to today) | Period: {first_date} &ndash; {last_date}, 2026 | {trading_days} trading days</p>

<div class="tab-nav">
  <button class="tab-btn active" onclick="showTab('overview')">Overview</button>
  <button class="tab-btn" onclick="showTab('monthly')">Monthly</button>
  <button class="tab-btn" onclick="showTab('daily')">Daily Breakdown</button>
  <button class="tab-btn" onclick="showTab('trades')">All Trades ({len(tdf)})</button>
</div>
"""

# ━━━━━ TAB 1: OVERVIEW ━━━━━
page += '<div id="tab-overview" class="tab-content active">\n'

# Verdict
page += f"""<div class="verdict-box {verdict_cls}">
<div class="verdict-title" style="color:{verdict_color}">{verdict_icon} &mdash; 2026 YTD ({first_date} &ndash; {last_date})</div>
<div class="verdict-text">
"""
if verdict_cls == "pass":
    page += f"""Strategy V3 (RR 5.0) is performing <span class="positive">strong</span> in 2026 live data.<br>
<strong>{len(tdf)}</strong> trades over <strong>{trading_days}</strong> trading days with <span class="positive">{fmt_money(total_pnl)}</span> P&L and PF of <strong>{pf:.2f}</strong>.<br>
Win Rate: <strong>{wr:.1f}%</strong> | Max DD: <strong>{fmt_money(max_dd)}</strong> | {winning_days}/{trading_days} winning days ({winning_days/trading_days*100:.0f}%).<br>
Annualized pace: ~<span class="positive">{fmt_money(annualized_pnl)}</span>/year (V3 backtest avg: ${1_280_993/10.1:,.0f}/year)."""
elif verdict_cls == "neutral":
    page += f"""Strategy is <span style="color:var(--orange)">marginally profitable</span>: {fmt_money(total_pnl)} P&L, PF {pf:.2f}."""
else:
    page += f"""Strategy is <span class="negative">underperforming</span>: {fmt_money(total_pnl)} P&L."""
page += """</div></div>\n"""

# Stats cards
def compare_html(actual, expected, higher_is_better=True):
    if expected == 0: return ''
    pct = (actual / expected - 1) * 100
    cls = "up" if (pct > 0) == higher_is_better else "down"
    return f'<div class="compare {cls}">vs V3 avg: {pct:+.0f}%</div>'

page += '<h2>Key Metrics</h2>\n<div class="stats-grid">\n'
cards = [
    ("Total Trades", f"{len(tdf)}", "neutral", compare_html(len(tdf)/months_traded, v3_monthly_avg_trades)),
    ("Win Rate", f"{wr:.1f}%", "positive" if wr >= 40 else "", ""),
    ("Total P&L", fmt_money(total_pnl), pnl_class(total_pnl), compare_html(total_pnl/months_traded, v3_monthly_avg_pnl)),
    ("Profit Factor", f"{pf:.2f}", "positive" if pf > 1 else "negative", ""),
    ("Avg Trade", fmt_money(avg_pnl), pnl_class(avg_pnl), ""),
    ("Avg Winner", fmt_money(avg_win), "positive", ""),
    ("Avg Loser", fmt_money(avg_loss), "negative", ""),
    ("Expectancy", fmt_money(expectancy), pnl_class(expectancy), ""),
    ("Max Drawdown", fmt_money(max_dd), "negative", ""),
    ("Max DD %", f"{max_dd_pct:.1f}%", "negative", ""),
    ("Best Trade", fmt_money(best), "positive", ""),
    ("Worst Trade", fmt_money(worst), "negative", ""),
    ("Best Day", fmt_money(best_day), "positive", ""),
    ("Worst Day", fmt_money(worst_day), "negative", ""),
    ("Trading Days", f"{trading_days} ({winning_days}W/{losing_days}L)", "neutral", ""),
    ("Consec W/L", f"{max_w}W / {max_l}L", "neutral", ""),
    ("Avg Bars Held", f"{avg_bars:.1f}", "neutral", ""),
    ("Final Account", f"${final_account:,.0f}", pnl_class(final_account - capital), ""),
    ("Gross Profit", fmt_money(gp), "positive", ""),
    ("Gross Loss", fmt_money(gl), "negative", ""),
    ("Annualized Pace", fmt_money(annualized_pnl), pnl_class(annualized_pnl), ""),
]
for lbl, val, cls, cmp in cards:
    page += f'  <div class="stat-card"><div class="label">{lbl}</div><div class="value {cls}">{val}</div>{cmp}</div>\n'
page += '</div>\n'

# Equity chart
page += '<h2>Equity Curve <span style="font-size:12px;color:var(--text-muted)">(daily, hover for details)</span></h2>\n'
page += '<div class="equity-chart"><canvas id="chartCanvas" width="1400" height="320"></canvas><div id="chartTooltip"></div></div>\n'

# Direction
page += '<h2>Direction Breakdown</h2>\n<div class="two-col">\n'
for dir_df, label, badge_cls in [(buy_df, "LONG (BUY)", "badge-buy"), (sell_df, "SHORT (SELL)", "badge-sell")]:
    if len(dir_df) == 0:
        page += f'<div><h3><span class="badge {badge_cls}">{label}</span> &mdash; 0 trades</h3></div>\n'
        continue
    dn = len(dir_df); dw = (dir_df["pnl_dollars"] > 0).sum()
    dwr = dw / dn * 100; dpnl = dir_df["pnl_dollars"].sum()
    davg = dir_df["pnl_dollars"].mean()
    davgw = dir_df.loc[dir_df["pnl_dollars"] > 0, "pnl_dollars"].mean() if dw > 0 else 0
    davgl = dir_df.loc[dir_df["pnl_dollars"] <= 0, "pnl_dollars"].mean() if (dn - dw) > 0 else 0
    dgp = dir_df.loc[dir_df["pnl_dollars"] > 0, "pnl_dollars"].sum()
    dgl = abs(dir_df.loc[dir_df["pnl_dollars"] <= 0, "pnl_dollars"].sum())
    dpf = dgp / dgl if dgl > 0 else 0
    page += f"""<div>
<h3><span class="badge {badge_cls}">{label}</span> &mdash; {dn} trades ({dn/len(tdf)*100:.0f}%)</h3>
<table><tbody>
<tr><td>P&L</td><td class="text-right {pnl_class(dpnl)}" style="font-weight:600">{fmt_money(dpnl)}</td></tr>
<tr><td>Win Rate</td><td class="text-right">{dwr:.1f}%</td></tr>
<tr><td>Profit Factor</td><td class="text-right">{dpf:.2f}</td></tr>
<tr><td>Avg Trade</td><td class="text-right {pnl_class(davg)}">{fmt_money(davg)}</td></tr>
<tr><td>Avg Winner</td><td class="text-right positive">{fmt_money(davgw)}</td></tr>
<tr><td>Avg Loser</td><td class="text-right negative">{fmt_money(davgl)}</td></tr>
<tr><td>Best</td><td class="text-right positive">{fmt_money(dir_df['pnl_dollars'].max())}</td></tr>
<tr><td>Worst</td><td class="text-right negative">{fmt_money(dir_df['pnl_dollars'].min())}</td></tr>
</tbody></table></div>\n"""
page += '</div>\n'

# Outcomes
page += '<h2>Outcome Distribution</h2>\n<table>\n<thead><tr>'
page += '<th>Outcome</th><th class="text-right">Count</th><th class="text-right">%</th>'
page += '<th class="text-right">Avg P&L</th><th class="text-right">Total P&L</th><th class="text-right">Avg Bars</th></tr></thead><tbody>\n'
for outcome in tdf["outcome"].value_counts().index:
    odf = tdf[tdf["outcome"] == outcome]
    on = len(odf); opct = on / len(tdf) * 100
    oavg = odf["pnl_dollars"].mean(); otot = odf["pnl_dollars"].sum()
    obars = odf["bars_held"].mean()
    page += f'<tr><td>{outcome_badge(outcome)}</td><td class="text-right">{on}</td><td class="text-right">{opct:.1f}%</td>'
    page += f'<td class="text-right {pnl_class(oavg)}">{fmt_money(oavg)}</td>'
    page += f'<td class="text-right {pnl_class(otot)}">{fmt_money(otot)}</td>'
    page += f'<td class="text-right">{obars:.1f}</td></tr>\n'
page += '</tbody></table>\n'

# Hourly
page += '<h2>Hourly Performance</h2>\n<table>\n<thead><tr>'
page += '<th>Hour</th><th class="text-right">Trades</th><th class="text-right">Buy</th><th class="text-right">Sell</th>'
page += '<th class="text-right">WR%</th><th class="text-right">P&L</th><th class="text-right">Avg</th></tr></thead><tbody>\n'
for h in range(7, 24):
    hdf = tdf[tdf["hour"] == h]
    if len(hdf) == 0: continue
    hn = len(hdf); hbuy = len(hdf[hdf["direction"] == "buy"]); hsell = hn - hbuy
    hwr = (hdf["pnl_dollars"] > 0).mean() * 100
    hpnl = hdf["pnl_dollars"].sum(); havg = hdf["pnl_dollars"].mean()
    page += f'<tr><td>{h:02d}:00</td><td class="text-right">{hn}</td><td class="text-right">{hbuy}</td><td class="text-right">{hsell}</td>'
    page += f'<td class="text-right">{hwr:.1f}%</td><td class="text-right {pnl_class(hpnl)}">{fmt_money(hpnl)}</td>'
    page += f'<td class="text-right {pnl_class(havg)}">{fmt_money(havg)}</td></tr>\n'
page += '</tbody></table>\n'
page += '</div>\n'

# ━━━━━ TAB 2: MONTHLY ━━━━━
page += '<div id="tab-monthly" class="tab-content">\n'
page += '<h2>Monthly Summary</h2>\n<table>\n<thead><tr>'
page += '<th>Month</th><th class="text-right">Trades</th><th class="text-right">Wins</th><th class="text-right">WR%</th>'
page += '<th class="text-right">P&L</th><th class="text-right">PF</th><th class="text-right">vs V3 Avg</th></tr></thead><tbody>\n'
for md in monthly_data:
    m = md["month"]
    vs = md["pnl"] - v3_monthly_avg_pnl
    vs_cls = pnl_class(vs)
    page += f'<tr><td><strong>{month_names[m]} 2026</strong></td>'
    page += f'<td class="text-right">{md["trades"]}</td><td class="text-right">{md["wins"]}</td>'
    page += f'<td class="text-right">{md["wr"]:.1f}%</td>'
    page += f'<td class="text-right {pnl_class(md["pnl"])}" style="font-weight:600">{fmt_money(md["pnl"])}</td>'
    page += f'<td class="text-right">{md["pf"]:.2f}</td>'
    page += f'<td class="text-right {vs_cls}">{fmt_money(vs)}</td></tr>\n'
page += f'<tr style="font-weight:700"><td>YTD TOTAL</td><td class="text-right">{len(tdf)}</td>'
page += f'<td class="text-right">{int(wins)}</td><td class="text-right">{wr:.1f}%</td>'
page += f'<td class="text-right {pnl_class(total_pnl)}">{fmt_money(total_pnl)}</td>'
page += f'<td class="text-right">{pf:.2f}</td><td></td></tr>\n'
page += f'<tr class="muted"><td>V3 Backtest Avg/Month</td><td class="text-right">{v3_monthly_avg_trades:.0f}</td>'
page += f'<td></td><td class="text-right">{v3_avg_wr:.1f}%</td>'
page += f'<td class="text-right">{fmt_money(v3_monthly_avg_pnl)}</td><td class="text-right">1.75</td><td></td></tr>\n'
page += '</tbody></table>\n'
page += '</div>\n'

# ━━━━━ TAB 3: DAILY ━━━━━
page += '<div id="tab-daily" class="tab-content">\n'
page += '<h2>Daily P&L</h2>\n<table>\n<thead><tr>'
page += '<th>Date</th><th>Day</th><th class="text-right">Trades</th><th class="text-right">Wins</th>'
page += '<th class="text-right">WR%</th><th class="text-right">P&L</th><th class="text-right">Cumulative</th></tr></thead><tbody>\n'

day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
cum_pnl = 0
for date in sorted(tdf["trade_date2"].unique()):
    ddf = tdf[tdf["trade_date2"] == date]
    dn = len(ddf); dw = (ddf["pnl_dollars"] > 0).sum()
    dwr = dw / dn * 100 if dn > 0 else 0
    dpnl = ddf["pnl_dollars"].sum(); cum_pnl += dpnl
    dow = pd.Timestamp(date).dayofweek
    page += f'<tr><td class="font-mono">{date}</td><td>{day_names[dow]}</td>'
    page += f'<td class="text-right">{dn}</td><td class="text-right">{dw}</td>'
    page += f'<td class="text-right">{dwr:.0f}%</td>'
    page += f'<td class="text-right {pnl_class(dpnl)}" style="font-weight:600">{fmt_money(dpnl)}</td>'
    page += f'<td class="text-right {pnl_class(cum_pnl)}">{fmt_money(cum_pnl)}</td></tr>\n'
page += f'<tr style="font-weight:700"><td>TOTAL</td><td></td>'
page += f'<td class="text-right">{len(tdf)}</td><td class="text-right">{int(wins)}</td>'
page += f'<td class="text-right">{wr:.0f}%</td>'
page += f'<td class="text-right {pnl_class(total_pnl)}">{fmt_money(total_pnl)}</td>'
page += f'<td class="text-right {pnl_class(total_pnl)}">{fmt_money(total_pnl)}</td></tr>\n'
page += '</tbody></table>\n'

# DOW
page += '<h2>Day of Week</h2>\n<table>\n<thead><tr>'
page += '<th>Day</th><th class="text-right">Trades</th><th class="text-right">WR%</th><th class="text-right">P&L</th><th class="text-right">Avg</th></tr></thead><tbody>\n'
for d in sorted(tdf["dow"].unique()):
    ddf = tdf[tdf["dow"] == d]
    if len(ddf) == 0: continue
    dn = len(ddf); dwr = (ddf["pnl_dollars"] > 0).mean() * 100; dpnl = ddf["pnl_dollars"].sum()
    davg = ddf["pnl_dollars"].mean()
    page += f'<tr><td>{day_names[d]}</td><td class="text-right">{dn}</td><td class="text-right">{dwr:.1f}%</td>'
    page += f'<td class="text-right {pnl_class(dpnl)}">{fmt_money(dpnl)}</td>'
    page += f'<td class="text-right {pnl_class(davg)}">{fmt_money(davg)}</td></tr>\n'
page += '</tbody></table>\n'
page += '</div>\n'

# ━━━━━ TAB 4: ALL TRADES ━━━━━
page += '<div id="tab-trades" class="tab-content">\n'
page += '<h2>All 2026 Trades</h2>\n'

# Filters
month_opts = "".join(f'<option value="{m}">{month_names[m]}</option>' for m in sorted(tdf["month"].unique()))
page += f"""<div class="filter-bar">
  <div><label>Month</label><br><select id="filterMonth" onchange="filterTrades()"><option value="all">All</option>{month_opts}</select></div>
  <div><label>Direction</label><br><select id="filterDir" onchange="filterTrades()"><option value="all">All</option><option value="buy">LONG</option><option value="sell">SHORT</option></select></div>
  <div><label>Outcome</label><br><select id="filterOutcome" onchange="filterTrades()"><option value="all">All</option><option value="SL">SL</option><option value="TP+trail">TP+trail</option><option value="TP+BE">TP+BE</option><option value="timeout">Timeout</option></select></div>
  <div><label>P&L</label><br><select id="filterPnl" onchange="filterTrades()"><option value="all">All</option><option value="winners">Winners</option><option value="losers">Losers</option></select></div>
  <div style="margin-left:auto"><span class="trade-count" id="tradeCount">{len(tdf)} trades</span></div>
</div>
"""

page += """<table id="tradeTable">
<thead><tr>
<th onclick="sortTable(0)">Entry Time</th>
<th onclick="sortTable(1)">Direction</th>
<th onclick="sortTable(2)" class="text-right">Entry</th>
<th onclick="sortTable(3)" class="text-right">SL</th>
<th onclick="sortTable(4)" class="text-right">TP</th>
<th onclick="sortTable(5)" class="text-right">Runner</th>
<th onclick="sortTable(6)" class="text-right">SL Dist</th>
<th onclick="sortTable(7)" class="text-right">Score</th>
<th onclick="sortTable(8)">Outcome</th>
<th onclick="sortTable(9)" class="text-right">P&L Pts</th>
<th onclick="sortTable(10)" class="text-right">P&L $</th>
<th onclick="sortTable(11)" class="text-right">Bars</th>
<th onclick="sortTable(12)" class="text-right">Account</th>
</tr></thead>
<tbody id="tradeBody">
"""

for _, row in tdf.iterrows():
    pnl_val = row["pnl_dollars"]
    cls = pnl_class(pnl_val)
    mn = int(row["month"])
    runner_ep = row.get("runner_exit_price")
    if pd.notna(runner_ep) and runner_ep is not None:
        runner_txt = f'{runner_ep:,.2f}'
        runner_cls = "positive" if "trail" in str(row["outcome"]) else ("negative" if "BE" in str(row["outcome"]) else "muted")
    else:
        runner_txt = "---"
        runner_cls = "muted"
    page += f'<tr data-month="{mn}" data-dir="{row["direction"]}" data-outcome="{row["outcome"]}" data-pnl="{pnl_val:.2f}">'
    page += f'<td class="font-mono">{str(row["entry_time"])[:16]}</td>'
    page += f'<td>{direction_badge(row["direction"])}</td>'
    page += f'<td class="text-right font-mono">{row["entry_price"]:,.2f}</td>'
    page += f'<td class="text-right font-mono">{row["sl_price"]:,.2f}</td>'
    page += f'<td class="text-right font-mono">{row["tp_price"]:,.2f}</td>'
    page += f'<td class="text-right font-mono {runner_cls}">{runner_txt}</td>'
    page += f'<td class="text-right">{row["sl_distance"]:.1f}</td>'
    page += f'<td class="text-right">{row["score"]:.2f}</td>'
    page += f'<td>{outcome_badge(row["outcome"])}</td>'
    page += f'<td class="text-right {cls}">{row["pnl_points"]:+.1f}</td>'
    page += f'<td class="text-right {cls}" style="font-weight:600">{fmt_money(pnl_val)}</td>'
    page += f'<td class="text-right">{int(row["bars_held"])}</td>'
    page += f'<td class="text-right font-mono">${row["account_after"]:,.0f}</td>'
    page += '</tr>\n'

page += '</tbody></table>\n</div>\n'

# Footer
page += f"""<div class="footer">
Forward Test Generated {time.strftime('%Y-%m-%d %H:%M')} | DB + Yahoo Finance NQ=F | Strategy V3 (RR {RR_RATIO}) | {len(tdf)} trades | {first_date} &ndash; {last_date} 2026
</div>
"""

# ─── JavaScript ───────────────────────────────────────────────────────────
page += f"""
<script>
const dailyData = {daily_eq_json};
const startCapital = {capital};

function showTab(name) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}}

function drawChart() {{
  const canvas = document.getElementById('chartCanvas');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = (rect.width - 40) * dpr;
  canvas.height = 320 * dpr;
  canvas.style.width = (rect.width - 40) + 'px';
  canvas.style.height = '320px';
  ctx.scale(dpr, dpr);
  const W = canvas.width / dpr, H = canvas.height / dpr;
  const pad = {{ top: 20, right: 20, bottom: 30, left: 70 }};
  const plotW = W - pad.left - pad.right, plotH = H - pad.top - pad.bottom;
  if (dailyData.length === 0) return;
  const equities = dailyData.map(d => d.equity);
  const minEq = Math.min(startCapital, ...equities) - 1000;
  const maxEq = Math.max(startCapital, ...equities) + 1000;
  const range = maxEq - minEq || 1;
  const toX = i => pad.left + (i / Math.max(dailyData.length - 1, 1)) * plotW;
  const toY = v => pad.top + (1 - (v - minEq) / range) * plotH;
  ctx.fillStyle = '#161b22'; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 0.5;
  for (let i = 0; i <= 5; i++) {{
    const y = pad.top + (i / 5) * plotH;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
    const val = maxEq - (i / 5) * range;
    ctx.fillStyle = '#8b949e'; ctx.font = '11px -apple-system, sans-serif';
    ctx.textAlign = 'right'; ctx.fillText('$' + Math.round(val).toLocaleString(), pad.left - 8, y + 4);
  }}
  const startY = toY(startCapital);
  ctx.strokeStyle = '#d29922'; ctx.lineWidth = 1; ctx.setLineDash([5, 5]);
  ctx.beginPath(); ctx.moveTo(pad.left, startY); ctx.lineTo(W - pad.right, startY); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#d29922'; ctx.font = '10px -apple-system, sans-serif';
  ctx.textAlign = 'left'; ctx.fillText('$100K', W - pad.right + 4, startY + 4);
  // Fill
  ctx.beginPath(); ctx.moveTo(toX(0), toY(startCapital));
  for (let i = 0; i < dailyData.length; i++) ctx.lineTo(toX(i), toY(dailyData[i].equity));
  ctx.lineTo(toX(dailyData.length - 1), toY(startCapital)); ctx.closePath();
  const lastEq = dailyData[dailyData.length - 1].equity;
  ctx.fillStyle = lastEq >= startCapital ? 'rgba(63,185,80,0.1)' : 'rgba(248,81,73,0.1)'; ctx.fill();
  // Line
  ctx.beginPath(); ctx.moveTo(toX(0), toY(dailyData[0].equity));
  for (let i = 1; i < dailyData.length; i++) ctx.lineTo(toX(i), toY(dailyData[i].equity));
  ctx.strokeStyle = lastEq >= startCapital ? '#3fb950' : '#f85149'; ctx.lineWidth = 2; ctx.stroke();
  // Dots + labels
  ctx.font = '9px -apple-system, sans-serif'; ctx.textAlign = 'center';
  let lastMonth = '';
  for (let i = 0; i < dailyData.length; i++) {{
    const x = toX(i), y = toY(dailyData[i].equity);
    ctx.fillStyle = dailyData[i].pnl >= 0 ? '#3fb950' : '#f85149';
    ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI * 2); ctx.fill();
    const mo = dailyData[i].date.substring(0, 7);
    if (mo !== lastMonth) {{ lastMonth = mo; ctx.fillStyle = '#8b949e'; ctx.fillText(dailyData[i].date.substring(5, 10), x, H - 8); }}
    else if (i % 5 === 0) {{ ctx.fillStyle = '#8b949e'; ctx.fillText(dailyData[i].date.substring(8, 10), x, H - 8); }}
  }}
  canvas._chartData = {{ toX, toY, pad, plotW, plotH, W, H }};
}}

function setupChartHover() {{
  const canvas = document.getElementById('chartCanvas');
  const tooltip = document.getElementById('chartTooltip');
  canvas.addEventListener('mousemove', function(e) {{
    const rect = canvas.getBoundingClientRect(); const x = e.clientX - rect.left;
    const cd = canvas._chartData;
    if (!cd || x < cd.pad.left || x > cd.W - cd.pad.right) {{ tooltip.style.display = 'none'; return; }}
    const idx = Math.round((x - cd.pad.left) / cd.plotW * (dailyData.length - 1));
    if (idx < 0 || idx >= dailyData.length) {{ tooltip.style.display = 'none'; return; }}
    const d = dailyData[idx];
    const pc = d.pnl >= 0 ? '#3fb950' : '#f85149';
    const ps = d.pnl >= 0 ? '+' : '';
    const wr = d.trades > 0 ? ((d.wins / d.trades) * 100).toFixed(0) : '0';
    tooltip.innerHTML = `<div class="tt-date">${{d.date}}</div>
      <div class="tt-row"><span class="tt-label">Equity:</span> <span style="color:#58a6ff;font-weight:600">$${{Math.round(d.equity).toLocaleString()}}</span></div>
      <div class="tt-row"><span class="tt-label">Day P&L:</span> <span style="color:${{pc}};font-weight:600">${{ps}}$${{Math.round(d.pnl).toLocaleString()}}</span></div>
      <div class="tt-row"><span class="tt-label">Trades:</span> <span>${{d.trades}} (${{d.wins}}W / ${{d.trades - d.wins}}L) ${{wr}}%</span></div>`;
    tooltip.style.display = 'block';
    let tx = e.clientX - rect.left + 16, ty = e.clientY - rect.top - 40;
    if (tx + 220 > rect.width) tx = e.clientX - rect.left - 230;
    if (ty < 0) ty = 10;
    tooltip.style.left = tx + 'px'; tooltip.style.top = ty + 'px';
  }});
  canvas.addEventListener('mouseleave', () => {{ tooltip.style.display = 'none'; }});
}}

function filterTrades() {{
  const month = document.getElementById('filterMonth').value;
  const dir = document.getElementById('filterDir').value;
  const outcome = document.getElementById('filterOutcome').value;
  const pnlF = document.getElementById('filterPnl').value;
  const rows = document.querySelectorAll('#tradeBody tr');
  let count = 0;
  rows.forEach(row => {{
    let show = true;
    if (month !== 'all' && row.dataset.month !== month) show = false;
    if (dir !== 'all' && row.dataset.dir !== dir) show = false;
    if (outcome !== 'all' && !row.dataset.outcome.includes(outcome)) show = false;
    if (pnlF === 'winners' && parseFloat(row.dataset.pnl) <= 0) show = false;
    if (pnlF === 'losers' && parseFloat(row.dataset.pnl) > 0) show = false;
    row.style.display = show ? '' : 'none';
    if (show) count++;
  }});
  document.getElementById('tradeCount').textContent = count + ' trades';
}}

let sortDir = {{}};
function sortTable(col) {{
  const tbody = document.getElementById('tradeBody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  const dir = sortDir[col] = !sortDir[col];
  rows.sort((a, b) => {{
    let av = a.cells[col].textContent.replace(/[$,+]/g, '').trim();
    let bv = b.cells[col].textContent.replace(/[$,+]/g, '').trim();
    let an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) return dir ? an - bn : bn - an;
    return dir ? av.localeCompare(bv) : bv.localeCompare(av);
  }});
  rows.forEach(row => tbody.appendChild(row));
}}

document.addEventListener('DOMContentLoaded', () => {{ drawChart(); setupChartHover(); }});
window.addEventListener('resize', drawChart);
</script>
"""

page += '</div></body></html>'

outfile = project_root / "reports" / "forward_test_2026.html"
outfile.write_text(page, encoding="utf-8")
print(f"\nReport saved to: {outfile}")
