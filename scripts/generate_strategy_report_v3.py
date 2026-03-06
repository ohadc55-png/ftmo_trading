"""Generate comprehensive HTML dashboard for:
BRK+MTF+VOL | 07:00-23:59 (no 00-06) | th=5.0 | 2c(1TP+1R) | SL<=25 | SmartDL$750 | RR 5.0
V3: Upgraded TP1 from RR 2.5 → RR 5.0 based on RR comparison analysis
"""
import sys, sqlite3, time, html, json
from pathlib import Path

project_root = Path(r"c:\Ohad\ohad\אפליקציות\ftmo")
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import pandas_ta as ta

from src.utils.config import load_config
from src.utils.timeutils import add_session_flags
from src.engines.mtf_trend import compute_mtf_trend
from src.engines.volume import compute_volume
from src.engines.key_levels import compute_key_levels
from src.engines.vwap import compute_vwap
from src.backtest.simulator import simulate_trades, trades_to_dataframe

# ─── Data Loading ─────────────────────────────────────────────────────────
print("Loading 10 years (2016-2026)...")
t0 = time.time()
conn = sqlite3.connect(str(project_root / "nq_data.db"))
df_1m = pd.read_sql_query(
    "SELECT datetime, open, high, low, close, volume FROM ohlcv_1m "
    "WHERE datetime >= '2016-01-01' ORDER BY datetime", conn)
conn.close()
df_1m["datetime"] = pd.to_datetime(df_1m["datetime"])
df_1m = df_1m.set_index("datetime")
if df_1m.index.tz is None:
    df_1m.index = df_1m.index.tz_localize("UTC")
df_1m.index = df_1m.index.tz_convert("America/New_York")
for col in ["open", "high", "low", "close", "volume"]:
    df_1m[col] = pd.to_numeric(df_1m[col], errors="coerce")
df_1m = df_1m.dropna(subset=["open", "high", "low", "close"])
df_1m["volume"] = df_1m["volume"].fillna(0)
df_5m = df_1m.resample("5min").agg({
    "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
}).dropna(subset=["open", "high", "low", "close"])
print(f"  {len(df_5m):,} bars in {time.time()-t0:.0f}s")

cfg = load_config(project_root / "config" / "settings.yaml")
ecfg = cfg.engines; bt = cfg.backtest; capital = 100_000.0
years = (df_5m.index[-1] - df_5m.index[0]).days / 365.25

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

# ─── Custom composite scoring ────────────────────────────────────────────
mtf_arr = df["mtf_score"].fillna(0).values
vol_arr = df["vol_score"].fillna(0).values
lvl_arr = df["levels_score"].fillna(0).values
direction_arr = df["mtf_direction"].fillna("neutral").values
bull_mask = (direction_arr == "bull")
bear_mask = (direction_arr == "bear")
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

buy_elig = (buy_score >= threshold) & bull_mask
sell_elig = (sell_score >= threshold) & bear_mask
hour_mask = np.ones(len(df), dtype=bool)
for h in exclude_hours:
    hour_mask &= (hour_arr != h)
buy_elig = buy_elig & hour_mask
sell_elig = sell_elig & hour_mask

# Cooldown
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

# Simulate
rp = bt.risk_pct / 100 if bt.risk_pct > 1 else bt.risk_pct
# V3: RR 5.0 (upgraded from 2.5 based on RR comparison analysis)
RR_RATIO = 5.0
trades = simulate_trades(df, starting_capital=capital,
    point_value=bt.point_value, sl_atr_mult=bt.sl_atr_mult,
    rr_ratio=RR_RATIO, risk_pct=rp,
    max_bars_held=bt.max_bars_held,
    use_runner=True, tp_contracts=1, runner_contracts=1,
    trail_atr_mult=bt.trail_atr_mult, runner_max_bars=bt.runner_max_bars,
    max_contracts=4)
tdf_raw = trades_to_dataframe(trades)

# ─── NEW: SL<=25 cap ─────────────────────────────────────────────────────
MAX_SL = 25
tdf_raw = tdf_raw[tdf_raw["sl_distance"] <= MAX_SL].reset_index(drop=True)
print(f"  {len(tdf_raw):,} trades after SL<={MAX_SL} filter")

# ─── NEW: Smart DL$750 ───────────────────────────────────────────────────
SMART_DL = 750
tdf_raw["trade_date"] = tdf_raw["entry_time"].dt.date
keep = []
for date, group in tdf_raw.groupby("trade_date"):
    daily_pnl = 0
    for idx, row in group.iterrows():
        if daily_pnl <= -SMART_DL:
            break
        # Smart guard: check if worst case (full SL) would exceed budget
        potential_loss = row["sl_distance"] * 40  # 2c x $20/pt
        remaining_budget = SMART_DL + daily_pnl
        if potential_loss > remaining_budget:
            continue  # Skip but keep looking for smaller-SL trades
        keep.append(idx)
        daily_pnl += row["pnl_dollars"]
tdf = tdf_raw.loc[keep].reset_index(drop=True)

# Recalculate account values after filtering
account = capital
for i in range(len(tdf)):
    tdf.loc[tdf.index[i], "account_before"] = account
    account += tdf.loc[tdf.index[i], "pnl_dollars"]
    tdf.loc[tdf.index[i], "account_after"] = account

tdf["year"] = tdf["entry_time"].dt.year
tdf["month"] = tdf["entry_time"].dt.month
tdf["hour"] = tdf["entry_time"].dt.hour
tdf["dow"] = tdf["entry_time"].dt.dayofweek

print(f"  {len(tdf)} trades after SmartDL${SMART_DL} filter")
print(f"  P&L: ${tdf['pnl_dollars'].sum():+,.0f}")

# ─── Stats ────────────────────────────────────────────────────────────────
def calc_stats(group):
    n = len(group)
    if n == 0: return None
    wins = (group["pnl_dollars"] > 0).sum()
    losses = n - wins
    wr = wins / n * 100
    total_pnl = group["pnl_dollars"].sum()
    avg_pnl = group["pnl_dollars"].mean()
    avg_win = group.loc[group["pnl_dollars"] > 0, "pnl_dollars"].mean() if wins > 0 else 0
    avg_loss = group.loc[group["pnl_dollars"] <= 0, "pnl_dollars"].mean() if losses > 0 else 0
    best = group["pnl_dollars"].max(); worst = group["pnl_dollars"].min()
    gp = group.loc[group["pnl_dollars"] > 0, "pnl_dollars"].sum() if wins > 0 else 0
    gl = abs(group.loc[group["pnl_dollars"] <= 0, "pnl_dollars"].sum()) if losses > 0 else 0
    pf = gp / gl if gl > 0 else float("inf")
    expectancy = (wr / 100 * avg_win) + ((1 - wr / 100) * avg_loss)
    equity = [capital]
    for _, row in group.iterrows():
        equity.append(equity[-1] + row["pnl_dollars"])
    eq = pd.Series(equity)
    peak = eq.expanding().max()
    dd = eq - peak
    max_dd = dd.min()
    peak_val = eq[:dd.idxmin() + 1].max()
    max_dd_pct = (max_dd / peak_val * 100) if peak_val > 0 else 0
    signs = (group["pnl_dollars"] > 0).astype(int).values
    max_w = max_l = cur_w = cur_l = 0
    for s_val in signs:
        if s_val == 1: cur_w += 1; cur_l = 0
        else: cur_l += 1; cur_w = 0
        max_w = max(max_w, cur_w); max_l = max(max_l, cur_l)
    avg_bars = group["bars_held"].mean()
    final_account = equity[-1]
    total_return = (final_account - capital) / capital * 100
    cagr = ((final_account / capital) ** (1 / years) - 1) * 100 if final_account > 0 else -100
    pnl_dd = total_pnl / abs(max_dd) if max_dd < 0 else float("inf")
    return {
        "trades": n, "wins": int(wins), "losses": int(losses), "wr": wr,
        "total_pnl": total_pnl, "avg_pnl": avg_pnl,
        "avg_win": avg_win if not pd.isna(avg_win) else 0,
        "avg_loss": avg_loss if not pd.isna(avg_loss) else 0,
        "best": best, "worst": worst,
        "gross_profit": gp, "gross_loss": gl,
        "profit_factor": pf, "expectancy": expectancy,
        "max_dd": max_dd, "max_dd_pct": max_dd_pct,
        "max_consec_w": max_w, "max_consec_l": max_l,
        "avg_bars": avg_bars, "final_account": final_account,
        "total_return": total_return, "cagr": cagr, "pnl_dd": pnl_dd,
        "equity": equity,
    }

stats = calc_stats(tdf)

# ─── Helper functions ─────────────────────────────────────────────────────
def pnl_class(val):
    if val > 0: return "positive"
    elif val < 0: return "negative"
    return ""

def fmt_money(val):
    if val >= 0: return f"+${val:,.0f}"
    return f"-${abs(val):,.0f}"

def fmt_pct(val): return f"{val:.1f}%"

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
    if max_abs == 0: return ""
    ratio = abs(val) / max_abs
    if val > 0:
        return "heatmap-green-3" if ratio > 0.66 else ("heatmap-green-2" if ratio > 0.33 else "heatmap-green-1")
    elif val < 0:
        return "heatmap-red-3" if ratio > 0.66 else ("heatmap-red-2" if ratio > 0.33 else "heatmap-red-1")
    return ""

# ─── Build daily equity data for interactive chart ────────────────────────
tdf_daily = tdf.copy()
tdf_daily["trade_date"] = tdf_daily["entry_time"].dt.date
daily_agg = tdf_daily.groupby("trade_date").agg(
    pnl=("pnl_dollars", "sum"),
    n_trades=("pnl_dollars", "count"),
    wins=("pnl_dollars", lambda x: (x > 0).sum()),
).reset_index()

# Build cumulative equity by day
daily_equity = []
eq_val = capital
for _, row in daily_agg.iterrows():
    eq_val += row["pnl"]
    daily_equity.append({
        "date": str(row["trade_date"]),
        "equity": round(eq_val, 2),
        "pnl": round(row["pnl"], 2),
        "trades": int(row["n_trades"]),
        "wins": int(row["wins"]),
    })

daily_equity_json = json.dumps(daily_equity)

# ─── Build trade log JSON ────────────────────────────────────────────────
trade_records = []
for _, row in tdf.iterrows():
    trade_records.append({
        "entry_time": str(row["entry_time"]),
        "exit_time": str(row["exit_time"]),
        "direction": row["direction"],
        "entry_price": round(row["entry_price"], 2),
        "exit_price": round(row["exit_price"], 2),
        "sl_price": round(row["sl_price"], 2),
        "tp_price": round(row["tp_price"], 2),
        "sl_distance": round(row["sl_distance"], 1),
        "contracts": int(row["contracts"]),
        "pnl_dollars": round(row["pnl_dollars"], 2),
        "pnl_points": round(row["pnl_points"], 1),
        "score": round(row["score"], 2),
        "outcome": row["outcome"],
        "bars_held": int(row["bars_held"]),
        "account_after": round(row["account_after"], 2),
        "year": int(row["year"]),
        "runner_pnl": round(row["runner_pnl"], 2) if pd.notna(row.get("runner_pnl")) else None,
        "runner_exit_price": round(row["runner_exit_price"], 2) if pd.notna(row.get("runner_exit_price")) else None,
    })

# ─── CSS ──────────────────────────────────────────────────────────────────
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
.text-center { text-align: center; }
.font-mono { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px; }
.heatmap-cell { font-weight: 600; font-size: 12px; padding: 6px 8px; }
.heatmap-green-1 { background: rgba(63, 185, 80, 0.15); }
.heatmap-green-2 { background: rgba(63, 185, 80, 0.3); }
.heatmap-green-3 { background: rgba(63, 185, 80, 0.45); }
.heatmap-red-1 { background: rgba(248, 81, 73, 0.15); }
.heatmap-red-2 { background: rgba(248, 81, 73, 0.3); }
.heatmap-red-3 { background: rgba(248, 81, 73, 0.45); }
.equity-chart { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 24px; position: relative; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }
.badge-buy { background: rgba(63, 185, 80, 0.2); color: var(--green); }
.badge-sell { background: rgba(248, 81, 73, 0.2); color: var(--red); }
.badge-tp { background: rgba(63, 185, 80, 0.2); color: var(--green); }
.badge-sl { background: rgba(248, 81, 73, 0.2); color: var(--red); }
.badge-timeout { background: rgba(210, 153, 34, 0.2); color: var(--orange); }
.badge-runner { background: rgba(88, 166, 255, 0.2); color: var(--blue); }
.tab-nav { display: flex; gap: 0; border-bottom: 1px solid var(--border); margin-bottom: 20px; flex-wrap: wrap; }
.tab-btn { padding: 10px 20px; cursor: pointer; border: none; background: none; color: var(--text-muted); font-size: 14px; border-bottom: 2px solid transparent; transition: all 0.2s; }
.tab-btn:hover { color: var(--text); }
.tab-btn.active { color: var(--blue); border-bottom-color: var(--blue); }
.tab-content { display: none; }
.tab-content.active { display: block; }
.pro-con-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }
.pro-box, .con-box { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 20px; }
.pro-box { border-left: 3px solid var(--green); }
.con-box { border-left: 3px solid var(--red); }
.pro-box h3 { color: var(--green); margin-top: 0; }
.con-box h3 { color: var(--red); margin-top: 0; }
.pro-box ul, .con-box ul { list-style: none; padding: 0; }
.pro-box li, .con-box li { padding: 4px 0; font-size: 14px; }
.pro-box li::before { content: "+"; color: var(--green); font-weight: 700; margin-right: 8px; }
.con-box li::before { content: "-"; color: var(--red); font-weight: 700; margin-right: 8px; }
.filter-bar { display: flex; gap: 12px; margin-bottom: 16px; align-items: center; flex-wrap: wrap; }
.filter-bar select, .filter-bar input { background: var(--card-bg); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; font-size: 13px; }
.filter-bar label { color: var(--text-muted); font-size: 12px; text-transform: uppercase; }
.trade-count { color: var(--text-muted); font-size: 13px; }
.trade-summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 24px; }
.summary-item { text-align: center; }
.summary-label { display: block; font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px; }
.summary-value { font-size: 16px; font-weight: 700; }
#chartTooltip { position: absolute; background: #1c2129; border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; font-size: 12px; pointer-events: none; z-index: 100; display: none; white-space: nowrap; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }
#chartTooltip .tt-date { color: var(--blue); font-weight: 600; margin-bottom: 4px; }
#chartTooltip .tt-row { display: flex; justify-content: space-between; gap: 16px; }
#chartTooltip .tt-label { color: var(--text-muted); }
#chartCanvas { cursor: crosshair; }
.footer { margin-top: 40px; padding: 20px 0; border-top: 1px solid var(--border); color: var(--text-muted); font-size: 12px; text-align: center; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
@media (max-width: 900px) { .two-col, .pro-con-grid { grid-template-columns: 1fr; } }
@media print { body { background: white; color: black; } .stat-card, table { border: 1px solid #ddd; } }
"""

# ─── Build HTML ───────────────────────────────────────────────────────────
s = stats
first_close = df_5m["close"].iloc[0]
last_close = df_5m["close"].iloc[-1]
index_return = (last_close / first_close - 1) * 100
index_cagr = ((last_close / first_close) ** (1 / years) - 1) * 100

page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NQ Strategy Dashboard - SmartDL V3 (RR 5.0)</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">

<h1>NQ Futures Strategy Dashboard V3 <span style="color:var(--gold,#ffd700);font-size:18px">(RR 5.0)</span></h1>
<p class="subtitle">BRK+MTF+VOL | Hours 07:00-23:59 | Threshold 5.0 | 2c (1TP+1R) | SL &le; {MAX_SL} pts | Smart DL ${SMART_DL} | <strong>RR 1:{RR_RATIO}</strong> | Jan 2016 &ndash; Mar 2026 ({years:.1f} years)<br>
<span style="color:var(--green)">Upgraded from V2 (RR 2.5 &rarr; RR 5.0) &mdash; +36% Total P&L improvement</span></p>

<div class="tab-nav">
  <button class="tab-btn active" onclick="showTab('overview')">Overview</button>
  <button class="tab-btn" onclick="showTab('breakdowns')">Breakdowns</button>
  <button class="tab-btn" onclick="showTab('monthly')">Monthly Heatmap</button>
  <button class="tab-btn" onclick="showTab('tradelog')">Trade Log ({s['trades']:,})</button>
</div>
"""

# ━━━━━ TAB 1: OVERVIEW ━━━━━
page += '<div id="tab-overview" class="tab-content active">\n'
page += '<h2>Key Performance Metrics</h2>\n<div class="stats-grid">\n'

# Compute worst day for display
tdf_tmp = tdf.copy()
tdf_tmp["trade_date"] = tdf_tmp["entry_time"].dt.date
daily_pnl_series = tdf_tmp.groupby("trade_date")["pnl_dollars"].sum()
worst_day = daily_pnl_series.min()
avg_losing_day = daily_pnl_series[daily_pnl_series < 0].mean()

cards = [
    ("Total Trades", f"{s['trades']:,}", "neutral"),
    ("Win Rate", fmt_pct(s["wr"]), "positive" if s["wr"] > 40 else "negative"),
    ("Total P&L", fmt_money(s["total_pnl"]), pnl_class(s["total_pnl"])),
    ("CAGR", f"{s['cagr']:.1f}%", "positive" if s["cagr"] > 0 else "negative"),
    ("Total Return", f"{s['total_return']:+,.0f}%", pnl_class(s["total_return"])),
    ("Profit Factor", f"{s['profit_factor']:.2f}", "positive" if s["profit_factor"] > 1 else "negative"),
    ("Max Drawdown", fmt_money(s["max_dd"]), "negative"),
    ("Max DD %", f"{s['max_dd_pct']:.1f}%", "negative"),
    ("P&L / DD Ratio", f"{s['pnl_dd']:.1f}", "positive" if s["pnl_dd"] > 20 else "neutral"),
    ("Avg Trade", fmt_money(s["avg_pnl"]), pnl_class(s["avg_pnl"])),
    ("Avg Winner", fmt_money(s["avg_win"]), "positive"),
    ("Avg Loser", fmt_money(s["avg_loss"]), "negative"),
    ("Best Trade", fmt_money(s["best"]), "positive"),
    ("Worst Trade", fmt_money(s["worst"]), "negative"),
    ("Worst Day", fmt_money(worst_day), "negative"),
    ("Consec W/L", f"{s['max_consec_w']}W / {s['max_consec_l']}L", "neutral"),
    ("NQ Index Return", f"{index_return:+,.0f}%", "neutral"),
    ("NQ Index CAGR", f"{index_cagr:.1f}%", "neutral"),
]
for lbl, val, cls in cards:
    page += f'  <div class="stat-card"><div class="label">{lbl}</div><div class="value {cls}">{val}</div></div>\n'
page += '</div>\n'

# Interactive equity curve (Canvas)
page += '<h2>Equity Curve <span style="font-size:12px;color:var(--text-muted)">(hover for daily details)</span></h2>\n'
page += '<div class="equity-chart">\n'
page += '  <canvas id="chartCanvas" width="1400" height="320"></canvas>\n'
page += '  <div id="chartTooltip"></div>\n'
page += '</div>\n'

# Strengths & Weaknesses
buy_s = calc_stats(tdf[tdf["direction"] == "buy"])
sell_s = calc_stats(tdf[tdf["direction"] == "sell"])
profitable_years = sum(1 for yr, grp in tdf.groupby("year") if grp["pnl_dollars"].sum() > 0)
total_years = tdf["year"].nunique()

page += '<h2>Strategy Profile</h2>\n<div class="pro-con-grid">\n'
page += f"""<div class="pro-box"><h3>Strengths</h3><ul>
<li>CAGR {s['cagr']:.1f}% beats NQ index ({index_cagr:.1f}% CAGR) by {s['cagr']-index_cagr:.0f}pp</li>
<li>Total return +{s['total_return']:,.0f}% vs NQ index +{index_return:,.0f}%</li>
<li>{profitable_years}/{total_years} profitable years ({profitable_years/total_years*100:.0f}%)</li>
<li>P&L/DD ratio of {s['pnl_dd']:.1f} (excellent risk-adjusted return)</li>
<li>RR 5.0 upgrade: +36% P&L vs V2 baseline (RR 2.5)</li>
<li>Profit Factor {s['profit_factor']:.2f} (up from 1.52 in V2)</li>
<li>Smart DL guarantees worst day &le; ${SMART_DL} (actual: {fmt_money(worst_day)})</li>
<li>Max SL = {MAX_SL}pts = max $1,000 loss per trade (controlled risk)</li>
<li>LONG trades dominate: PF {buy_s['profit_factor']:.2f}, WR {buy_s['wr']:.1f}%</li>
<li>Higher avg winner compensates for lower win rate</li>
<li>No overnight risk (07-23 only, no 00-06)</li>
</ul></div>
<div class="con-box"><h3>Weaknesses / Risks</h3><ul>
<li>Win rate {s['wr']:.1f}% (lower than V2's 43.2% due to higher TP target)</li>
<li>Max {s['max_consec_l']} consecutive losses (psychological challenge)</li>
<li>SHORT side weaker: PF {sell_s['profit_factor']:.2f}, WR {sell_s['wr']:.1f}%</li>
<li>Skips high-volatility setups (SL>{MAX_SL}pts filtered out)</li>
<li>~{(daily_pnl_series < 0).sum()}/{len(daily_pnl_series)} ({(daily_pnl_series < 0).mean()*100:.0f}%) trading days are losing days</li>
<li>Max DD of {fmt_money(s['max_dd'])} ({s['max_dd_pct']:.1f}% of peak)</li>
<li>Backtest only - real slippage/commissions not included</li>
<li>Requires 2 NQ contracts ($40/pt exposure per trade)</li>
</ul></div></div>
"""

# Strategy config summary
page += f"""<h2>Strategy Configuration</h2>
<table><thead><tr><th>Parameter</th><th>Value</th><th>Change from V2</th></tr></thead><tbody>
<tr><td>Engines</td><td>MTF Trend + Volume + Key Levels (no VWAP)</td><td class="muted">unchanged</td></tr>
<tr><td>SELL Weights</td><td>MTF=3, VOL=2, BRK=3 (total=8)</td><td class="muted">unchanged</td></tr>
<tr><td>BUY Weights</td><td>MTF=3, VOL=1, BRK=4 (total=8)</td><td class="muted">unchanged</td></tr>
<tr><td>Threshold</td><td>5.0</td><td class="muted">unchanged</td></tr>
<tr><td>Trading Hours</td><td>07:00-23:59 NY (no 00:00-06:59)</td><td class="muted">unchanged</td></tr>
<tr><td>Max SL Distance</td><td style="color:var(--orange)">{MAX_SL} points (max ${MAX_SL*40:,} loss per trade)</td><td class="muted">unchanged</td></tr>
<tr><td>Daily Loss Limit</td><td style="color:var(--orange)">Smart DL ${SMART_DL} (pre-trade risk guard)</td><td class="muted">unchanged</td></tr>
<tr><td>Contracts</td><td>2 per trade (1 TP + 1 Runner)</td><td class="muted">unchanged</td></tr>
<tr><td>Risk per Trade</td><td>ATR * 1.5 (max {MAX_SL} pts SL)</td><td class="muted">unchanged</td></tr>
<tr><td style="font-weight:700">Reward:Risk (TP1)</td><td style="color:var(--green);font-weight:700">1:{RR_RATIO}</td><td style="color:var(--green);font-weight:700">was 1:2.5 &rarr; 1:{RR_RATIO} (+{((RR_RATIO/2.5)-1)*100:.0f}%)</td></tr>
<tr><td>Runner</td><td>Trail at ATR * 1.5, max 120 bars</td><td class="muted">unchanged</td></tr>
<tr><td>Timeout</td><td>60 bars (5h)</td><td class="muted">unchanged</td></tr>
<tr><td>Direction Filter</td><td>MTF direction (bull/bear)</td><td class="muted">unchanged</td></tr>
</tbody></table>
"""
page += '</div>\n'

# ━━━━━ TAB 2: BREAKDOWNS ━━━━━
page += '<div id="tab-breakdowns" class="tab-content">\n'

# Direction breakdown
page += '<h2>Long vs Short</h2>\n<div class="two-col">\n'
for direction, label in [("buy", "LONG (BUY)"), ("sell", "SHORT (SELL)")]:
    ddf = tdf[tdf["direction"] == direction]
    ds = calc_stats(ddf)
    if ds is None: continue
    pct = len(ddf) / len(tdf) * 100
    pnl_pct = ds["total_pnl"] / s["total_pnl"] * 100 if s["total_pnl"] != 0 else 0
    badge_cls = "badge-buy" if direction == "buy" else "badge-sell"
    page += f"""<div>
<h3><span class="badge {badge_cls}">{label}</span> &mdash; {ds['trades']:,} trades ({pct:.0f}%)</h3>
<table><tbody>
<tr><td>P&L</td><td class="text-right {pnl_class(ds['total_pnl'])}">{fmt_money(ds['total_pnl'])} ({pnl_pct:.0f}% of total)</td></tr>
<tr><td>Win Rate</td><td class="text-right">{ds['wr']:.1f}%</td></tr>
<tr><td>Profit Factor</td><td class="text-right">{ds['profit_factor']:.2f}</td></tr>
<tr><td>Avg Trade</td><td class="text-right {pnl_class(ds['avg_pnl'])}">{fmt_money(ds['avg_pnl'])}</td></tr>
<tr><td>Avg Winner</td><td class="text-right positive">{fmt_money(ds['avg_win'])}</td></tr>
<tr><td>Avg Loser</td><td class="text-right negative">{fmt_money(ds['avg_loss'])}</td></tr>
<tr><td>Best</td><td class="text-right positive">{fmt_money(ds['best'])}</td></tr>
<tr><td>Worst</td><td class="text-right negative">{fmt_money(ds['worst'])}</td></tr>
<tr><td>Max DD</td><td class="text-right negative">{fmt_money(ds['max_dd'])}</td></tr>
<tr><td>Consec W/L</td><td class="text-right">{ds['max_consec_w']}W / {ds['max_consec_l']}L</td></tr>
</tbody></table></div>\n"""
page += '</div>\n'

# Yearly breakdown
page += '<h2>Yearly Performance</h2>\n<table>\n<thead><tr>'
page += '<th>Year</th><th class="text-right">Trades</th><th class="text-right">Wins</th><th class="text-right">WR%</th>'
page += '<th class="text-right">P&L</th><th class="text-right">PF</th><th class="text-right">Max DD</th>'
page += '<th class="text-right">Buy P&L</th><th class="text-right">Sell P&L</th></tr></thead><tbody>\n'
for yr in sorted(tdf["year"].unique()):
    ydf = tdf[tdf["year"] == yr]
    yn = len(ydf); yw = (ydf["pnl_dollars"] > 0).sum()
    ywr = yw / yn * 100; ypnl = ydf["pnl_dollars"].sum()
    ygp = ydf.loc[ydf["pnl_dollars"] > 0, "pnl_dollars"].sum()
    ygl = abs(ydf.loc[ydf["pnl_dollars"] <= 0, "pnl_dollars"].sum())
    ypf = ygp / ygl if ygl > 0 else 0
    ycum = ydf["pnl_dollars"].cumsum(); ydd = (ycum - ycum.cummax()).min()
    buy_pnl = ydf.loc[ydf["direction"] == "buy", "pnl_dollars"].sum()
    sell_pnl = ydf.loc[ydf["direction"] == "sell", "pnl_dollars"].sum()
    page += f'<tr><td><strong>{yr}</strong></td><td class="text-right">{yn}</td><td class="text-right">{yw}</td>'
    page += f'<td class="text-right">{ywr:.1f}%</td><td class="text-right {pnl_class(ypnl)}">{fmt_money(ypnl)}</td>'
    page += f'<td class="text-right">{ypf:.2f}</td><td class="text-right negative">{fmt_money(ydd)}</td>'
    page += f'<td class="text-right {pnl_class(buy_pnl)}">{fmt_money(buy_pnl)}</td>'
    page += f'<td class="text-right {pnl_class(sell_pnl)}">{fmt_money(sell_pnl)}</td></tr>\n'
page += f'<tr style="font-weight:700"><td>TOTAL</td><td class="text-right">{s["trades"]}</td><td class="text-right">{s["wins"]}</td>'
page += f'<td class="text-right">{s["wr"]:.1f}%</td><td class="text-right {pnl_class(s["total_pnl"])}">{fmt_money(s["total_pnl"])}</td>'
page += f'<td class="text-right">{s["profit_factor"]:.2f}</td><td class="text-right negative">{fmt_money(s["max_dd"])}</td>'
page += f'<td class="text-right {pnl_class(buy_s["total_pnl"])}">{fmt_money(buy_s["total_pnl"])}</td>'
page += f'<td class="text-right {pnl_class(sell_s["total_pnl"])}">{fmt_money(sell_s["total_pnl"])}</td></tr>\n'
page += '</tbody></table>\n'

# Hourly breakdown
page += '<h2>Hourly Performance</h2>\n<table>\n<thead><tr>'
page += '<th>Hour</th><th class="text-right">Trades</th><th class="text-right">Buy</th><th class="text-right">Sell</th>'
page += '<th class="text-right">WR%</th><th class="text-right">P&L</th><th class="text-right">Avg</th><th class="text-right">PF</th></tr></thead><tbody>\n'
for h in range(7, 24):
    hdf = tdf[tdf["hour"] == h]
    if len(hdf) == 0: continue
    hn = len(hdf); hbuy = len(hdf[hdf["direction"] == "buy"]); hsell = hn - hbuy
    hwr = (hdf["pnl_dollars"] > 0).mean() * 100
    hpnl = hdf["pnl_dollars"].sum(); havg = hdf["pnl_dollars"].mean()
    hgp = hdf.loc[hdf["pnl_dollars"] > 0, "pnl_dollars"].sum()
    hgl = abs(hdf.loc[hdf["pnl_dollars"] <= 0, "pnl_dollars"].sum())
    hpf = hgp / hgl if hgl > 0 else 0
    page += f'<tr><td>{h:02d}:00</td><td class="text-right">{hn}</td><td class="text-right">{hbuy}</td><td class="text-right">{hsell}</td>'
    page += f'<td class="text-right">{hwr:.1f}%</td><td class="text-right {pnl_class(hpnl)}">{fmt_money(hpnl)}</td>'
    page += f'<td class="text-right {pnl_class(havg)}">{fmt_money(havg)}</td><td class="text-right">{hpf:.2f}</td></tr>\n'
page += '</tbody></table>\n'

# Outcome distribution
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

# Day of week
page += '<h2>Day of Week</h2>\n<table>\n<thead><tr>'
page += '<th>Day</th><th class="text-right">Trades</th><th class="text-right">WR%</th><th class="text-right">P&L</th><th class="text-right">PF</th></tr></thead><tbody>\n'
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
for d in sorted(tdf["dow"].unique()):
    ddf = tdf[tdf["dow"] == d]
    dn = len(ddf); dwr = (ddf["pnl_dollars"] > 0).mean() * 100; dpnl = ddf["pnl_dollars"].sum()
    dgp = ddf.loc[ddf["pnl_dollars"] > 0, "pnl_dollars"].sum()
    dgl = abs(ddf.loc[ddf["pnl_dollars"] <= 0, "pnl_dollars"].sum())
    dpf = dgp / dgl if dgl > 0 else 0
    page += f'<tr><td>{day_names[d]}</td><td class="text-right">{dn}</td><td class="text-right">{dwr:.1f}%</td>'
    page += f'<td class="text-right {pnl_class(dpnl)}">{fmt_money(dpnl)}</td><td class="text-right">{dpf:.2f}</td></tr>\n'
page += '</tbody></table>\n'
page += '</div>\n'

# ━━━━━ TAB 3: MONTHLY HEATMAP ━━━━━
page += '<div id="tab-monthly" class="tab-content">\n'
page += '<h2>Monthly P&L Heatmap</h2>\n'

months_list = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
monthly_pnl = {}
for yr in sorted(tdf["year"].unique()):
    for m in range(1, 13):
        mdf = tdf[(tdf["year"] == yr) & (tdf["month"] == m)]
        monthly_pnl[(yr, m)] = mdf["pnl_dollars"].sum() if len(mdf) > 0 else None

max_abs_monthly = max(abs(v) for v in monthly_pnl.values() if v is not None and v != 0) if monthly_pnl else 1

page += '<table>\n<thead><tr><th>Year</th>'
for m in months_list:
    page += f'<th class="text-right">{m}</th>'
page += '<th class="text-right" style="font-weight:700">Total</th></tr></thead><tbody>\n'

for yr in sorted(tdf["year"].unique()):
    page += f'<tr><td><strong>{yr}</strong></td>'
    yr_total = 0
    for m in range(1, 13):
        val = monthly_pnl.get((yr, m))
        if val is None:
            page += '<td class="text-right heatmap-cell muted">---</td>'
        else:
            yr_total += val
            cls = heatmap_class(val, max_abs_monthly)
            page += f'<td class="text-right heatmap-cell {cls} {pnl_class(val)}">{fmt_money(val)}</td>'
    page += f'<td class="text-right heatmap-cell" style="font-weight:700;{" color:var(--green)" if yr_total > 0 else " color:var(--red)"}">{fmt_money(yr_total)}</td></tr>\n'

page += '<tr style="font-weight:700"><td>TOTAL</td>'
for m in range(1, 13):
    m_total = sum(monthly_pnl.get((yr, m), 0) or 0 for yr in tdf["year"].unique())
    cls = heatmap_class(m_total, max_abs_monthly)
    page += f'<td class="text-right heatmap-cell {cls} {pnl_class(m_total)}">{fmt_money(m_total)}</td>'
grand_total = sum(v for v in monthly_pnl.values() if v is not None)
page += f'<td class="text-right heatmap-cell {pnl_class(grand_total)}">{fmt_money(grand_total)}</td></tr>\n'
page += '</tbody></table>\n'

# Monthly trade count heatmap
page += '<h2>Monthly Trade Count</h2>\n<table>\n<thead><tr><th>Year</th>'
for m in months_list:
    page += f'<th class="text-right">{m}</th>'
page += '<th class="text-right">Total</th></tr></thead><tbody>\n'
for yr in sorted(tdf["year"].unique()):
    page += f'<tr><td><strong>{yr}</strong></td>'
    yr_count = 0
    for m in range(1, 13):
        mdf = tdf[(tdf["year"] == yr) & (tdf["month"] == m)]
        n = len(mdf); yr_count += n
        page += f'<td class="text-right muted">{n if n > 0 else "---"}</td>'
    page += f'<td class="text-right">{yr_count}</td></tr>\n'
page += '</tbody></table>\n'

# Monthly win rate heatmap
page += '<h2>Monthly Win Rate</h2>\n<table>\n<thead><tr><th>Year</th>'
for m in months_list:
    page += f'<th class="text-right">{m}</th>'
page += '</tr></thead><tbody>\n'
for yr in sorted(tdf["year"].unique()):
    page += f'<tr><td><strong>{yr}</strong></td>'
    for m in range(1, 13):
        mdf = tdf[(tdf["year"] == yr) & (tdf["month"] == m)]
        if len(mdf) == 0:
            page += '<td class="text-right muted">---</td>'
        else:
            wr = (mdf["pnl_dollars"] > 0).mean() * 100
            cls = "positive" if wr >= 45 else ("negative" if wr < 35 else "")
            page += f'<td class="text-right {cls}">{wr:.0f}%</td>'
    page += '</tr>\n'
page += '</tbody></table>\n'
page += '</div>\n'

# ━━━━━ TAB 4: TRADE LOG ━━━━━
page += '<div id="tab-tradelog" class="tab-content">\n'
page += '<h2>Trade Log</h2>\n'

# Filter bar
year_options = "".join(f'<option value="{yr}">{yr}</option>' for yr in sorted(tdf["year"].unique()))
month_names_js = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
month_options = "".join(f'<option value="{i+1}">{m}</option>' for i, m in enumerate(month_names_js))
page += f"""<div class="filter-bar">
  <div><label>Year</label><br><select id="filterYear" onchange="onYearChange()"><option value="all">All Years</option>{year_options}</select></div>
  <div><label>Month</label><br><select id="filterMonth" onchange="filterTrades()" disabled><option value="all">All Months</option>{month_options}</select></div>
  <div><label>Direction</label><br><select id="filterDir" onchange="filterTrades()"><option value="all">All</option><option value="buy">LONG</option><option value="sell">SHORT</option></select></div>
  <div><label>Outcome</label><br><select id="filterOutcome" onchange="filterTrades()"><option value="all">All</option><option value="SL">SL</option><option value="TP+trail">TP+trail</option><option value="timeout">Timeout</option><option value="TP+BE">TP+BE</option></select></div>
  <div><label>P&L</label><br><select id="filterPnl" onchange="filterTrades()"><option value="all">All</option><option value="winners">Winners</option><option value="losers">Losers</option></select></div>
  <div style="margin-left:auto"><span class="trade-count" id="tradeCount">{len(tdf):,} trades</span></div>
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
    pnl = row["pnl_dollars"]
    cls = pnl_class(pnl)
    yr = int(row["year"])
    mn = int(row["month"])
    runner_ep = row.get("runner_exit_price")
    if pd.notna(runner_ep) and runner_ep is not None:
        runner_txt = f'{runner_ep:,.2f}'
        runner_cls = "positive" if "trail" in str(row["outcome"]) else ("negative" if "BE" in str(row["outcome"]) else "muted")
    else:
        runner_txt = "---"
        runner_cls = "muted"
    page += f'<tr data-year="{yr}" data-month="{mn}" data-dir="{row["direction"]}" data-outcome="{row["outcome"]}" data-pnl="{pnl:.2f}">'
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
    page += f'<td class="text-right {cls}" style="font-weight:600">{fmt_money(pnl)}</td>'
    page += f'<td class="text-right">{int(row["bars_held"])}</td>'
    page += f'<td class="text-right font-mono">${row["account_after"]:,.0f}</td>'
    page += '</tr>\n'

page += '</tbody></table>\n'

# Summary bar
page += """<div id="tradeSummary" class="trade-summary">
  <div class="summary-item"><span class="summary-label">Trades</span><span class="summary-value" id="sumTrades">-</span></div>
  <div class="summary-item"><span class="summary-label">Wins</span><span class="summary-value positive" id="sumWins">-</span></div>
  <div class="summary-item"><span class="summary-label">Losses</span><span class="summary-value negative" id="sumLosses">-</span></div>
  <div class="summary-item"><span class="summary-label">Win Rate</span><span class="summary-value" id="sumWR">-</span></div>
  <div class="summary-item"><span class="summary-label">Total P&L</span><span class="summary-value" id="sumPnl">-</span></div>
  <div class="summary-item"><span class="summary-label">Avg Trade</span><span class="summary-value" id="sumAvg">-</span></div>
  <div class="summary-item"><span class="summary-label">Avg Winner</span><span class="summary-value positive" id="sumAvgWin">-</span></div>
  <div class="summary-item"><span class="summary-label">Avg Loser</span><span class="summary-value negative" id="sumAvgLoss">-</span></div>
  <div class="summary-item"><span class="summary-label">Profit Factor</span><span class="summary-value" id="sumPF">-</span></div>
  <div class="summary-item"><span class="summary-label">Best</span><span class="summary-value positive" id="sumBest">-</span></div>
  <div class="summary-item"><span class="summary-label">Worst</span><span class="summary-value negative" id="sumWorst">-</span></div>
  <div class="summary-item"><span class="summary-label">Max DD</span><span class="summary-value negative" id="sumDD">-</span></div>
</div>
"""

page += '</div>\n'

# Footer
page += f"""<div class="footer">
Generated {time.strftime('%Y-%m-%d %H:%M')} | V3 (RR {RR_RATIO}) | BRK+MTF+VOL | SL&le;{MAX_SL} + SmartDL${SMART_DL} | {len(tdf):,} trades | {years:.1f} years | NQ Futures $20/pt
</div>
"""

# ─── JavaScript ───────────────────────────────────────────────────────────
page += f"""
<script>
// Daily equity data for chart
const dailyData = {daily_equity_json};
const startCapital = {capital};

// ─── Interactive Equity Chart ─────────────────────────────
function drawChart() {{
  const canvas = document.getElementById('chartCanvas');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * dpr - 40 * dpr;
  canvas.height = 320 * dpr;
  canvas.style.width = (rect.width - 40) + 'px';
  canvas.style.height = '320px';
  ctx.scale(dpr, dpr);

  const W = canvas.width / dpr;
  const H = canvas.height / dpr;
  const pad = {{ top: 20, right: 20, bottom: 30, left: 70 }};
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  if (dailyData.length === 0) return;

  const equities = dailyData.map(d => d.equity);
  const minEq = Math.min(startCapital, ...equities);
  const maxEq = Math.max(startCapital, ...equities);
  const range = maxEq - minEq || 1;

  const toX = i => pad.left + (i / (dailyData.length - 1)) * plotW;
  const toY = v => pad.top + (1 - (v - minEq) / range) * plotH;

  // Background
  ctx.fillStyle = '#161b22';
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {{
    const y = pad.top + (i / 4) * plotH;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
    const val = maxEq - (i / 4) * range;
    ctx.fillStyle = '#8b949e'; ctx.font = '11px -apple-system, sans-serif';
    ctx.textAlign = 'right'; ctx.fillText('$' + Math.round(val).toLocaleString(), pad.left - 8, y + 4);
  }}

  // Starting capital line
  const startY = toY(startCapital);
  ctx.strokeStyle = '#d29922'; ctx.lineWidth = 1; ctx.setLineDash([5, 5]);
  ctx.beginPath(); ctx.moveTo(pad.left, startY); ctx.lineTo(W - pad.right, startY); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#d29922'; ctx.font = '10px -apple-system, sans-serif';
  ctx.textAlign = 'left'; ctx.fillText('$100K', W - pad.right + 4, startY + 4);

  // Fill area
  ctx.beginPath();
  ctx.moveTo(toX(0), toY(startCapital));
  for (let i = 0; i < dailyData.length; i++) ctx.lineTo(toX(i), toY(dailyData[i].equity));
  ctx.lineTo(toX(dailyData.length - 1), toY(startCapital));
  ctx.closePath();
  ctx.fillStyle = 'rgba(88,166,255,0.08)';
  ctx.fill();

  // Line
  ctx.beginPath();
  ctx.moveTo(toX(0), toY(dailyData[0].equity));
  for (let i = 1; i < dailyData.length; i++) ctx.lineTo(toX(i), toY(dailyData[i].equity));
  ctx.strokeStyle = '#58a6ff'; ctx.lineWidth = 1.5; ctx.stroke();

  // Year labels on X axis
  let lastYear = '';
  ctx.fillStyle = '#8b949e'; ctx.font = '10px -apple-system, sans-serif'; ctx.textAlign = 'center';
  for (let i = 0; i < dailyData.length; i++) {{
    const yr = dailyData[i].date.substring(0, 4);
    if (yr !== lastYear) {{
      lastYear = yr;
      ctx.fillText(yr, toX(i), H - 8);
      ctx.strokeStyle = '#30363d'; ctx.lineWidth = 0.3;
      ctx.beginPath(); ctx.moveTo(toX(i), pad.top); ctx.lineTo(toX(i), H - pad.bottom); ctx.stroke();
    }}
  }}

  // Store chart coords for hover
  canvas._chartData = {{ toX, toY, pad, plotW, plotH, W, H }};
}}

// Hover tooltip
function setupChartHover() {{
  const canvas = document.getElementById('chartCanvas');
  const tooltip = document.getElementById('chartTooltip');

  canvas.addEventListener('mousemove', function(e) {{
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const cd = canvas._chartData;
    if (!cd || x < cd.pad.left || x > cd.W - cd.pad.right) {{
      tooltip.style.display = 'none';
      return;
    }}
    const ratio = (x - cd.pad.left) / cd.plotW;
    const idx = Math.round(ratio * (dailyData.length - 1));
    if (idx < 0 || idx >= dailyData.length) {{ tooltip.style.display = 'none'; return; }}

    const d = dailyData[idx];
    const pnlColor = d.pnl >= 0 ? '#3fb950' : '#f85149';
    const pnlSign = d.pnl >= 0 ? '+' : '';
    const wr = d.trades > 0 ? ((d.wins / d.trades) * 100).toFixed(0) : '0';

    tooltip.innerHTML = `
      <div class="tt-date">${{d.date}}</div>
      <div class="tt-row"><span class="tt-label">Equity:</span> <span style="color:#58a6ff;font-weight:600">$${{Math.round(d.equity).toLocaleString()}}</span></div>
      <div class="tt-row"><span class="tt-label">Day P&L:</span> <span style="color:${{pnlColor}};font-weight:600">${{pnlSign}}$${{Math.round(d.pnl).toLocaleString()}}</span></div>
      <div class="tt-row"><span class="tt-label">Trades:</span> <span>${{d.trades}} (${{d.wins}}W / ${{d.trades - d.wins}}L)</span></div>
      <div class="tt-row"><span class="tt-label">Win Rate:</span> <span>${{wr}}%</span></div>
    `;
    tooltip.style.display = 'block';

    // Position tooltip
    let tx = e.clientX - rect.left + 16;
    let ty = e.clientY - rect.top - 40;
    if (tx + 220 > rect.width) tx = e.clientX - rect.left - 230;
    if (ty < 0) ty = 10;
    tooltip.style.left = tx + 'px';
    tooltip.style.top = ty + 'px';
  }});

  canvas.addEventListener('mouseleave', function() {{
    tooltip.style.display = 'none';
  }});
}}

// ─── Tab switching ─────────────────────────────
function showTab(name) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}}

// ─── Trade log filters ─────────────────────────
function onYearChange() {{
  const monthSel = document.getElementById('filterMonth');
  const yearVal = document.getElementById('filterYear').value;
  if (yearVal === 'all') {{ monthSel.value = 'all'; monthSel.disabled = true; }}
  else {{ monthSel.disabled = false; }}
  filterTrades();
}}

function fmtMoney(v) {{
  if (v >= 0) return '+$' + Math.round(v).toLocaleString();
  return '-$' + Math.round(Math.abs(v)).toLocaleString();
}}
function pnlCls(v) {{ return v > 0 ? 'positive' : (v < 0 ? 'negative' : ''); }}

function filterTrades() {{
  const year = document.getElementById('filterYear').value;
  const month = document.getElementById('filterMonth').value;
  const dir = document.getElementById('filterDir').value;
  const outcome = document.getElementById('filterOutcome').value;
  const pnlFilter = document.getElementById('filterPnl').value;
  const rows = document.querySelectorAll('#tradeBody tr');
  let count = 0, pnls = [];
  rows.forEach(row => {{
    let show = true;
    if (year !== 'all' && row.dataset.year !== year) show = false;
    if (month !== 'all' && row.dataset.month !== month) show = false;
    if (dir !== 'all' && row.dataset.dir !== dir) show = false;
    if (outcome !== 'all' && !row.dataset.outcome.includes(outcome)) show = false;
    if (pnlFilter === 'winners' && parseFloat(row.dataset.pnl) <= 0) show = false;
    if (pnlFilter === 'losers' && parseFloat(row.dataset.pnl) > 0) show = false;
    row.style.display = show ? '' : 'none';
    if (show) {{ count++; pnls.push(parseFloat(row.dataset.pnl)); }}
  }});
  document.getElementById('tradeCount').textContent = count.toLocaleString() + ' trades';
  updateSummary(pnls);
}}

function updateSummary(pnls) {{
  const n = pnls.length;
  if (n === 0) {{
    ['sumTrades','sumWins','sumLosses','sumWR','sumPnl','sumAvg','sumAvgWin','sumAvgLoss','sumPF','sumBest','sumWorst','sumDD'].forEach(id => {{
      document.getElementById(id).textContent = '-';
      document.getElementById(id).className = 'summary-value';
    }});
    return;
  }}
  const wins = pnls.filter(p => p > 0);
  const losses = pnls.filter(p => p <= 0);
  const totalPnl = pnls.reduce((a,b) => a+b, 0);
  const avgPnl = totalPnl / n;
  const avgWin = wins.length > 0 ? wins.reduce((a,b) => a+b, 0) / wins.length : 0;
  const avgLoss = losses.length > 0 ? losses.reduce((a,b) => a+b, 0) / losses.length : 0;
  const grossProfit = wins.reduce((a,b) => a+b, 0);
  const grossLoss = Math.abs(losses.reduce((a,b) => a+b, 0));
  const pf = grossLoss > 0 ? (grossProfit / grossLoss) : 0;
  const best = Math.max(...pnls);
  const worst = Math.min(...pnls);
  let cum = 0, peak = 0, maxDD = 0;
  for (let i = 0; i < pnls.length; i++) {{
    cum += pnls[i]; if (cum > peak) peak = cum;
    let dd = cum - peak; if (dd < maxDD) maxDD = dd;
  }}
  const wr = (wins.length / n * 100);
  const el = (id, text, cls) => {{ const e = document.getElementById(id); e.textContent = text; e.className = 'summary-value ' + (cls||''); }};
  el('sumTrades', n.toLocaleString(), 'neutral');
  el('sumWins', wins.length.toLocaleString(), 'positive');
  el('sumLosses', losses.length.toLocaleString(), 'negative');
  el('sumWR', wr.toFixed(1) + '%', wr >= 40 ? 'positive' : (wr < 30 ? 'negative' : ''));
  el('sumPnl', fmtMoney(totalPnl), pnlCls(totalPnl));
  el('sumAvg', fmtMoney(avgPnl), pnlCls(avgPnl));
  el('sumAvgWin', fmtMoney(avgWin), 'positive');
  el('sumAvgLoss', fmtMoney(avgLoss), 'negative');
  el('sumPF', pf > 0 ? pf.toFixed(2) : '-', pf >= 1 ? 'positive' : 'negative');
  el('sumBest', fmtMoney(best), 'positive');
  el('sumWorst', fmtMoney(worst), 'negative');
  el('sumDD', fmtMoney(maxDD), 'negative');
}}

let sortDir = {{}};
function sortTable(col) {{
  const table = document.getElementById('tradeTable');
  const tbody = table.querySelector('tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  const dir = sortDir[col] = !sortDir[col];
  rows.sort((a, b) => {{
    let aVal = a.cells[col].textContent.replace(/[$,+]/g, '').trim();
    let bVal = b.cells[col].textContent.replace(/[$,+]/g, '').trim();
    let aNum = parseFloat(aVal); let bNum = parseFloat(bVal);
    if (!isNaN(aNum) && !isNaN(bNum)) return dir ? aNum - bNum : bNum - aNum;
    return dir ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
  }});
  rows.forEach(row => tbody.appendChild(row));
}}

// Init
document.addEventListener('DOMContentLoaded', function() {{
  drawChart();
  setupChartHover();
  filterTrades();
}});
window.addEventListener('resize', drawChart);
</script>
"""

page += '</div></body></html>'

# ─── Write file ───────────────────────────────────────────────────────────
output_path = project_root / "reports" / "strategy_smartdl_v3.html"
output_path.parent.mkdir(exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    f.write(page)

print(f"\nHTML report written to: {output_path}")
print(f"  {len(tdf):,} trades, P&L ${tdf['pnl_dollars'].sum():+,.0f}")
