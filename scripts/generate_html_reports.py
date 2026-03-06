"""Generate detailed HTML reports for each trading configuration.

Creates one HTML file per config with full stats, breakdowns, and trade log.
"""
import sys, sqlite3, time, html
from pathlib import Path

project_root = Path(r"c:\Ohad\ohad\אפליקציות\ftmo")
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from src.utils.config import load_config
from src.utils.timeutils import add_session_flags
from src.engines.mtf_trend import compute_mtf_trend
from src.engines.volume import compute_volume
from src.engines.key_levels import compute_key_levels
from src.engines.vwap import compute_vwap
from src.engines.composite import compute_composite
from src.backtest.simulator import simulate_trades, trades_to_dataframe
from src.backtest.analysis import compute_stats

# ─── Data Loading ───────────────────────────────────────────────────────────
print("Loading 1m data from DB...")
t0 = time.time()
db_path = project_root / "nq_data.db"
conn = sqlite3.connect(str(db_path))
df_1m = pd.read_sql_query(
    "SELECT datetime, open, high, low, close, volume FROM ohlcv_1m "
    "WHERE datetime >= '2021-02-01' ORDER BY datetime", conn)
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
print(f"  Loaded {len(df_1m)} 1m bars in {time.time()-t0:.1f}s")

# Resample to 5m
df = df_1m.resample("5min").agg({
    "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
}).dropna(subset=["open", "high", "low", "close"])
print(f"  {len(df)} 5m bars")

# ─── Pipeline ───────────────────────────────────────────────────────────────
print("Running indicator pipeline...")
t0 = time.time()
cfg = load_config(project_root / "config" / "settings.yaml")
ecfg = cfg.engines

df = add_session_flags(df, tz=cfg.timezone,
    premarket_start_hour=ecfg.key_levels.premarket_start_hour,
    premarket_end_hour=ecfg.key_levels.premarket_end_hour,
    rth_start_hour=ecfg.vwap.rth_start_hour,
    rth_start_minute=ecfg.vwap.rth_start_minute)
df = compute_mtf_trend(df, ecfg.mtf_trend, cfg.mtf_timeframes)
df = compute_volume(df, ecfg.volume)
df = compute_key_levels(df, ecfg.key_levels)
df = compute_vwap(df, ecfg.vwap)
df = compute_composite(df, cfg.scoring, atr_period=cfg.backtest.atr_period)
print(f"  Pipeline done in {time.time()-t0:.1f}s")

# ─── Simulate 3 configs ────────────────────────────────────────────────────
bt = cfg.backtest
capital = 100_000.0

configs = [
    ("2c (1TP+1R)", True, 1, 1),
    ("3c (2TP+1R)", True, 2, 1),
    ("4c Simple",   False, 0, 0),
]

all_results = {}
for label, use_runner, tp_c, run_c in configs:
    trades = simulate_trades(
        df, starting_capital=capital,
        point_value=bt.point_value, sl_atr_mult=bt.sl_atr_mult,
        rr_ratio=bt.rr_ratio,
        risk_pct=bt.risk_pct / 100 if bt.risk_pct > 1 else bt.risk_pct,
        max_bars_held=bt.max_bars_held,
        use_runner=use_runner, tp_contracts=tp_c, runner_contracts=run_c,
        trail_atr_mult=bt.trail_atr_mult, runner_max_bars=bt.runner_max_bars)
    tdf = trades_to_dataframe(trades)
    tdf["year"] = tdf["entry_time"].dt.year
    tdf["month"] = tdf["entry_time"].dt.month
    all_results[label] = tdf
    print(f"  {label}: {len(tdf)} trades, P&L ${tdf['pnl_dollars'].sum():+,.0f}")


# ─── Helper functions ───────────────────────────────────────────────────────

def calc_stats(group):
    """Compute stats for a group of trades."""
    n = len(group)
    if n == 0:
        return None
    wins = (group["pnl_dollars"] > 0).sum()
    losses = (group["pnl_dollars"] <= 0).sum()
    wr = wins / n * 100
    total_pnl = group["pnl_dollars"].sum()
    avg_pnl = group["pnl_dollars"].mean()
    avg_win = group.loc[group["pnl_dollars"] > 0, "pnl_dollars"].mean() if wins > 0 else 0
    avg_loss = group.loc[group["pnl_dollars"] <= 0, "pnl_dollars"].mean() if losses > 0 else 0
    best = group["pnl_dollars"].max()
    worst = group["pnl_dollars"].min()
    gross_profit = group.loc[group["pnl_dollars"] > 0, "pnl_dollars"].sum() if wins > 0 else 0
    gross_loss = abs(group.loc[group["pnl_dollars"] <= 0, "pnl_dollars"].sum()) if losses > 0 else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    expectancy = (wr / 100 * avg_win) + ((1 - wr / 100) * avg_loss)

    # Equity & drawdown
    equity = [capital]
    for _, row in group.iterrows():
        equity.append(equity[-1] + row["pnl_dollars"])
    eq = pd.Series(equity)
    peak = eq.expanding().max()
    dd = (eq - peak)
    max_dd = dd.min()
    peak_val = eq[:dd.idxmin() + 1].max()
    max_dd_pct = (max_dd / peak_val * 100) if peak_val > 0 else 0

    # Consecutive
    signs = (group["pnl_dollars"] > 0).astype(int).values
    max_w = max_l = cur_w = cur_l = 0
    for s in signs:
        if s == 1:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)

    avg_bars = group["bars_held"].mean()
    final_account = group["account_after"].iloc[-1]
    total_return = (final_account - capital) / capital * 100

    return {
        "trades": n, "wins": int(wins), "losses": int(losses), "wr": wr,
        "total_pnl": total_pnl, "avg_pnl": avg_pnl,
        "avg_win": avg_win if not pd.isna(avg_win) else 0,
        "avg_loss": avg_loss if not pd.isna(avg_loss) else 0,
        "best": best, "worst": worst,
        "gross_profit": gross_profit, "gross_loss": gross_loss,
        "profit_factor": pf, "expectancy": expectancy,
        "max_dd": max_dd, "max_dd_pct": max_dd_pct,
        "max_consec_w": max_w, "max_consec_l": max_l,
        "avg_bars": avg_bars, "final_account": final_account,
        "total_return": total_return,
        "equity": equity,
    }


def pnl_class(val):
    """CSS class for positive/negative."""
    if val > 0:
        return "positive"
    elif val < 0:
        return "negative"
    return ""


def fmt_money(val):
    """Format dollar value."""
    if val >= 0:
        return f"+${val:,.0f}"
    return f"-${abs(val):,.0f}"


def fmt_pct(val):
    return f"{val:.1f}%"


# ─── HTML Generation ────────────────────────────────────────────────────────

CSS = """
:root {
    --bg: #0d1117;
    --card-bg: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --text-muted: #8b949e;
    --green: #3fb950;
    --red: #f85149;
    --blue: #58a6ff;
    --orange: #d29922;
    --purple: #bc8cff;
    --accent: #1f6feb;
}

* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
    padding: 20px;
}

.container { max-width: 1400px; margin: 0 auto; }

h1 {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 8px;
    color: var(--blue);
}
h2 {
    font-size: 20px;
    font-weight: 600;
    margin: 30px 0 16px 0;
    color: var(--text);
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
}
h3 {
    font-size: 16px;
    font-weight: 600;
    margin: 20px 0 10px 0;
    color: var(--text-muted);
}

.subtitle {
    color: var(--text-muted);
    font-size: 14px;
    margin-bottom: 24px;
}

/* Stats cards */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 24px;
}
.stat-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
}
.stat-card .label {
    font-size: 12px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.stat-card .value {
    font-size: 24px;
    font-weight: 700;
}
.stat-card .value.positive { color: var(--green); }
.stat-card .value.negative { color: var(--red); }
.stat-card .value.neutral { color: var(--blue); }

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 24px;
    font-size: 13px;
}
thead {
    background: #1c2129;
}
th {
    padding: 10px 12px;
    text-align: left;
    font-weight: 600;
    color: var(--text-muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
}
td {
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
}
tr:last-child td { border-bottom: none; }
tr:hover { background: #1c2129; }

.positive { color: var(--green); }
.negative { color: var(--red); }
.neutral { color: var(--blue); }
.muted { color: var(--text-muted); }

.text-right { text-align: right; }
.text-center { text-align: center; }
.font-mono { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px; }

/* Monthly heatmap */
.heatmap-cell {
    font-weight: 600;
    font-size: 12px;
    padding: 6px 8px;
}
.heatmap-green-1 { background: rgba(63, 185, 80, 0.15); }
.heatmap-green-2 { background: rgba(63, 185, 80, 0.3); }
.heatmap-green-3 { background: rgba(63, 185, 80, 0.45); }
.heatmap-red-1 { background: rgba(248, 81, 73, 0.15); }
.heatmap-red-2 { background: rgba(248, 81, 73, 0.3); }
.heatmap-red-3 { background: rgba(248, 81, 73, 0.45); }

/* Equity chart (SVG) */
.equity-chart {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 24px;
}

/* Trade log */
.trade-log-section {
    margin-top: 30px;
}
.trade-log-section table {
    font-size: 12px;
}
.trade-log-section td, .trade-log-section th {
    padding: 6px 10px;
}

/* Direction badges */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
}
.badge-buy { background: rgba(63, 185, 80, 0.2); color: var(--green); }
.badge-sell { background: rgba(248, 81, 73, 0.2); color: var(--red); }
.badge-tp { background: rgba(63, 185, 80, 0.2); color: var(--green); }
.badge-sl { background: rgba(248, 81, 73, 0.2); color: var(--red); }
.badge-timeout { background: rgba(210, 153, 34, 0.2); color: var(--orange); }
.badge-runner { background: rgba(88, 166, 255, 0.2); color: var(--blue); }

/* Tabs */
.tab-nav {
    display: flex;
    gap: 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
}
.tab-btn {
    padding: 10px 20px;
    cursor: pointer;
    border: none;
    background: none;
    color: var(--text-muted);
    font-size: 14px;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
}
.tab-btn:hover { color: var(--text); }
.tab-btn.active { color: var(--blue); border-bottom-color: var(--blue); }
.tab-content { display: none; }
.tab-content.active { display: block; }

/* Footer */
.footer {
    margin-top: 40px;
    padding: 20px 0;
    border-top: 1px solid var(--border);
    color: var(--text-muted);
    font-size: 12px;
    text-align: center;
}

@media print {
    body { background: white; color: black; }
    .stat-card { border: 1px solid #ddd; }
    table { border: 1px solid #ddd; }
}
"""


def build_equity_svg(equity, width=1200, height=250):
    """Build an SVG equity curve chart."""
    n = len(equity)
    if n < 2:
        return ""
    min_eq = min(equity)
    max_eq = max(equity)
    rng = max_eq - min_eq if max_eq != min_eq else 1
    pad = 60

    points = []
    for i, val in enumerate(equity):
        x = pad + (i / (n - 1)) * (width - 2 * pad)
        y = height - pad - ((val - min_eq) / rng) * (height - 2 * pad)
        points.append(f"{x:.1f},{y:.1f}")

    # Fill area
    fill_points = [f"{pad:.1f},{height - pad:.1f}"] + points + [f"{width - pad:.1f},{height - pad:.1f}"]

    # Grid lines
    grid_lines = ""
    for i in range(5):
        gy = pad + i * (height - 2 * pad) / 4
        val = max_eq - i * rng / 4
        grid_lines += f'<line x1="{pad}" y1="{gy:.1f}" x2="{width - pad}" y2="{gy:.1f}" stroke="#30363d" stroke-width="0.5"/>'
        grid_lines += f'<text x="{pad - 5}" y="{gy + 4:.1f}" text-anchor="end" fill="#8b949e" font-size="11">${val:,.0f}</text>'

    # Starting capital line
    start_y = height - pad - ((capital - min_eq) / rng) * (height - 2 * pad)

    svg = f"""<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;">
  <rect width="{width}" height="{height}" fill="#161b22" rx="8"/>
  {grid_lines}
  <line x1="{pad}" y1="{start_y:.1f}" x2="{width - pad}" y2="{start_y:.1f}" stroke="#d29922" stroke-width="1" stroke-dasharray="5,5"/>
  <text x="{width - pad + 5}" y="{start_y + 4:.1f}" fill="#d29922" font-size="10">$100K</text>
  <polygon points="{' '.join(fill_points)}" fill="rgba(88,166,255,0.1)"/>
  <polyline points="{' '.join(points)}" fill="none" stroke="#58a6ff" stroke-width="2"/>
  <circle cx="{points[0].split(',')[0]}" cy="{points[0].split(',')[1]}" r="3" fill="#58a6ff"/>
  <circle cx="{points[-1].split(',')[0]}" cy="{points[-1].split(',')[1]}" r="3" fill="#58a6ff"/>
</svg>"""
    return svg


def outcome_badge(outcome):
    """Return HTML badge for trade outcome."""
    o = outcome.upper()
    if "SL" in o:
        cls = "badge-sl"
    elif "TP" in o:
        if "trail" in outcome.lower() or "BE" in outcome or "timeout" in outcome.lower():
            cls = "badge-runner"
        else:
            cls = "badge-tp"
    elif "timeout" in o.lower():
        cls = "badge-timeout"
    else:
        cls = "badge-tp"
    return f'<span class="badge {cls}">{html.escape(outcome)}</span>'


def direction_badge(direction):
    """Return HTML badge for direction."""
    if direction == "buy":
        return '<span class="badge badge-buy">LONG</span>'
    return '<span class="badge badge-sell">SHORT</span>'


def heatmap_class(val, max_abs):
    """Get heatmap intensity class."""
    if max_abs == 0:
        return ""
    ratio = abs(val) / max_abs
    if val > 0:
        if ratio > 0.66:
            return "heatmap-green-3"
        elif ratio > 0.33:
            return "heatmap-green-2"
        return "heatmap-green-1"
    elif val < 0:
        if ratio > 0.66:
            return "heatmap-red-3"
        elif ratio > 0.33:
            return "heatmap-red-2"
        return "heatmap-red-1"
    return ""


def generate_html(label, tdf, config_desc):
    """Generate a full HTML report for one config."""
    s = calc_stats(tdf)
    if s is None:
        return ""

    # ── Header ──
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NQ Strategy Report - {html.escape(label)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">

<h1>NQ Futures Strategy Report</h1>
<p class="subtitle">{html.escape(label)} &mdash; {html.escape(config_desc)} &mdash; Feb 2021 &ndash; Feb 2026 (5 Years)</p>

<!-- Tab Navigation -->
<div class="tab-nav">
  <button class="tab-btn active" onclick="showTab('overview')">Overview</button>
  <button class="tab-btn" onclick="showTab('breakdowns')">Breakdowns</button>
  <button class="tab-btn" onclick="showTab('monthly')">Monthly Heatmap</button>
  <button class="tab-btn" onclick="showTab('tradelog')">Trade Log ({s['trades']})</button>
</div>
"""

    # ━━━━━━━━━━ TAB 1: OVERVIEW ━━━━━━━━━━
    page += '<div id="tab-overview" class="tab-content active">\n'

    # Key stats cards
    page += '<h2>Key Performance Metrics</h2>\n<div class="stats-grid">\n'

    cards = [
        ("Total Trades", str(s["trades"]), "neutral"),
        ("Win Rate", fmt_pct(s["wr"]), "positive" if s["wr"] > 50 else "negative"),
        ("Total P&L", fmt_money(s["total_pnl"]), pnl_class(s["total_pnl"])),
        ("Total Return", f"{s['total_return']:+.1f}%", pnl_class(s["total_return"])),
        ("Final Account", f"${s['final_account']:,.0f}", "neutral"),
        ("Profit Factor", f"{s['profit_factor']:.2f}" if s['profit_factor'] < 100 else "∞", "positive" if s['profit_factor'] > 1 else "negative"),
        ("Avg Trade", fmt_money(s["avg_pnl"]), pnl_class(s["avg_pnl"])),
        ("Expectancy", fmt_money(s["expectancy"]), pnl_class(s["expectancy"])),
        ("Max Drawdown", fmt_money(s["max_dd"]), "negative"),
        ("Max DD %", f"{s['max_dd_pct']:.1f}%", "negative"),
        ("Avg Win", fmt_money(s["avg_win"]), "positive"),
        ("Avg Loss", fmt_money(s["avg_loss"]), "negative"),
        ("Best Trade", fmt_money(s["best"]), "positive"),
        ("Worst Trade", fmt_money(s["worst"]), "negative"),
        ("Avg Bars Held", f"{s['avg_bars']:.1f}", "neutral"),
        ("Consec W/L", f"{s['max_consec_w']}W / {s['max_consec_l']}L", "neutral"),
    ]
    for lbl, val, cls in cards:
        page += f'''  <div class="stat-card">
    <div class="label">{lbl}</div>
    <div class="value {cls}">{val}</div>
  </div>\n'''
    page += '</div>\n'

    # Win/Loss summary
    page += f"""<h2>Win / Loss Summary</h2>
<table>
<thead><tr><th></th><th class="text-right">Count</th><th class="text-right">%</th><th class="text-right">Total P&L</th><th class="text-right">Average</th></tr></thead>
<tbody>
<tr><td class="positive">Winners</td><td class="text-right">{s['wins']}</td><td class="text-right">{s['wr']:.1f}%</td><td class="text-right positive">{fmt_money(s['gross_profit'])}</td><td class="text-right positive">{fmt_money(s['avg_win'])}</td></tr>
<tr><td class="negative">Losers</td><td class="text-right">{s['losses']}</td><td class="text-right">{100-s['wr']:.1f}%</td><td class="text-right negative">{fmt_money(-s['gross_loss'])}</td><td class="text-right negative">{fmt_money(s['avg_loss'])}</td></tr>
<tr style="font-weight:600"><td>Total</td><td class="text-right">{s['trades']}</td><td class="text-right">100%</td><td class="text-right {pnl_class(s['total_pnl'])}">{fmt_money(s['total_pnl'])}</td><td class="text-right {pnl_class(s['avg_pnl'])}">{fmt_money(s['avg_pnl'])}</td></tr>
</tbody></table>
"""

    # Equity curve
    page += '<h2>Equity Curve</h2>\n'
    page += '<div class="equity-chart">\n'
    page += build_equity_svg(s["equity"])
    page += '</div>\n'

    page += '</div>\n'  # end overview tab

    # ━━━━━━━━━━ TAB 2: BREAKDOWNS ━━━━━━━━━━
    page += '<div id="tab-breakdowns" class="tab-content">\n'

    # Direction breakdown
    page += '<h2>Breakdown by Direction</h2>\n'
    page += '<table><thead><tr><th>Direction</th><th class="text-right">Trades</th><th class="text-right">Wins</th><th class="text-right">Losses</th><th class="text-right">Win Rate</th><th class="text-right">Total P&L</th><th class="text-right">Avg Trade</th><th class="text-right">Avg Win</th><th class="text-right">Avg Loss</th></tr></thead><tbody>\n'
    for direction, tag in [("buy", "LONG"), ("sell", "SHORT")]:
        ddf = tdf[tdf["direction"] == direction]
        if ddf.empty:
            continue
        ds = calc_stats(ddf)
        page += f'<tr><td>{direction_badge(direction)}</td>'
        page += f'<td class="text-right">{ds["trades"]}</td>'
        page += f'<td class="text-right">{ds["wins"]}</td>'
        page += f'<td class="text-right">{ds["losses"]}</td>'
        page += f'<td class="text-right">{ds["wr"]:.1f}%</td>'
        page += f'<td class="text-right {pnl_class(ds["total_pnl"])}">{fmt_money(ds["total_pnl"])}</td>'
        page += f'<td class="text-right {pnl_class(ds["avg_pnl"])}">{fmt_money(ds["avg_pnl"])}</td>'
        page += f'<td class="text-right positive">{fmt_money(ds["avg_win"])}</td>'
        page += f'<td class="text-right negative">{fmt_money(ds["avg_loss"])}</td>'
        page += '</tr>\n'
    page += '</tbody></table>\n'

    # Outcome breakdown
    page += '<h2>Breakdown by Outcome</h2>\n'
    page += '<table><thead><tr><th>Outcome</th><th class="text-right">Trades</th><th class="text-right">% of Total</th><th class="text-right">Win Rate</th><th class="text-right">Total P&L</th><th class="text-right">Avg Trade</th></tr></thead><tbody>\n'
    for outcome in sorted(tdf["outcome"].unique()):
        sub = tdf[tdf["outcome"] == outcome]
        n_sub = len(sub)
        wr_sub = (sub["pnl_dollars"] > 0).mean() * 100
        pnl_sub = sub["pnl_dollars"].sum()
        avg_sub = sub["pnl_dollars"].mean()
        page += f'<tr><td>{outcome_badge(outcome)}</td>'
        page += f'<td class="text-right">{n_sub}</td>'
        page += f'<td class="text-right">{n_sub/s["trades"]*100:.1f}%</td>'
        page += f'<td class="text-right">{wr_sub:.1f}%</td>'
        page += f'<td class="text-right {pnl_class(pnl_sub)}">{fmt_money(pnl_sub)}</td>'
        page += f'<td class="text-right {pnl_class(avg_sub)}">{fmt_money(avg_sub)}</td>'
        page += '</tr>\n'
    page += '</tbody></table>\n'

    # Yearly breakdown
    page += '<h2>Yearly Performance</h2>\n'
    page += '<table><thead><tr><th>Year</th><th class="text-right">Trades</th><th class="text-right">Long</th><th class="text-right">Short</th><th class="text-right">Win Rate</th><th class="text-right">Long WR</th><th class="text-right">Short WR</th><th class="text-right">P&L</th><th class="text-right">Long P&L</th><th class="text-right">Short P&L</th><th class="text-right">Avg Trade</th><th class="text-right">Max DD</th></tr></thead><tbody>\n'

    years = sorted(tdf["year"].unique())
    for yr in years:
        ydf = tdf[tdf["year"] == yr]
        ys = calc_stats(ydf)
        buy_y = ydf[ydf["direction"] == "buy"]
        sell_y = ydf[ydf["direction"] == "sell"]
        buy_wr = (buy_y["pnl_dollars"] > 0).mean() * 100 if len(buy_y) > 0 else 0
        sell_wr = (sell_y["pnl_dollars"] > 0).mean() * 100 if len(sell_y) > 0 else 0
        buy_pnl = buy_y["pnl_dollars"].sum() if len(buy_y) > 0 else 0
        sell_pnl = sell_y["pnl_dollars"].sum() if len(sell_y) > 0 else 0
        page += f'<tr><td><strong>{yr}</strong></td>'
        page += f'<td class="text-right">{ys["trades"]}</td>'
        page += f'<td class="text-right">{len(buy_y)}</td>'
        page += f'<td class="text-right">{len(sell_y)}</td>'
        page += f'<td class="text-right">{ys["wr"]:.1f}%</td>'
        page += f'<td class="text-right">{buy_wr:.1f}%</td>'
        page += f'<td class="text-right">{sell_wr:.1f}%</td>'
        page += f'<td class="text-right {pnl_class(ys["total_pnl"])}">{fmt_money(ys["total_pnl"])}</td>'
        page += f'<td class="text-right {pnl_class(buy_pnl)}">{fmt_money(buy_pnl)}</td>'
        page += f'<td class="text-right {pnl_class(sell_pnl)}">{fmt_money(sell_pnl)}</td>'
        page += f'<td class="text-right {pnl_class(ys["avg_pnl"])}">{fmt_money(ys["avg_pnl"])}</td>'
        page += f'<td class="text-right negative">{fmt_money(ys["max_dd"])}</td>'
        page += '</tr>\n'
    page += '</tbody></table>\n'

    # Score distribution
    page += '<h2>Performance by Score Range</h2>\n'
    page += '<table><thead><tr><th>Score Range</th><th class="text-right">Trades</th><th class="text-right">Win Rate</th><th class="text-right">Total P&L</th><th class="text-right">Avg Trade</th></tr></thead><tbody>\n'
    bins = [7.5, 8.0, 8.5, 9.0, 9.5, 10.1]
    labels_b = ["7.5-8.0", "8.0-8.5", "8.5-9.0", "9.0-9.5", "9.5+"]
    tdf_copy = tdf.copy()
    tdf_copy["score_bin"] = pd.cut(tdf_copy["score"], bins=bins, labels=labels_b, right=False)
    for b in labels_b:
        bdf = tdf_copy[tdf_copy["score_bin"] == b]
        if bdf.empty:
            continue
        wr_b = (bdf["pnl_dollars"] > 0).mean() * 100
        pnl_b = bdf["pnl_dollars"].sum()
        avg_b = bdf["pnl_dollars"].mean()
        page += f'<tr><td>{b}</td>'
        page += f'<td class="text-right">{len(bdf)}</td>'
        page += f'<td class="text-right">{wr_b:.1f}%</td>'
        page += f'<td class="text-right {pnl_class(pnl_b)}">{fmt_money(pnl_b)}</td>'
        page += f'<td class="text-right {pnl_class(avg_b)}">{fmt_money(avg_b)}</td>'
        page += '</tr>\n'
    page += '</tbody></table>\n'

    # Top 10 best / worst
    for worst, title in [(False, "Top 10 Best Trades"), (True, "Top 10 Worst Trades")]:
        sorted_df = tdf.sort_values("pnl_dollars", ascending=worst).head(10)
        page += f'<h2>{title}</h2>\n'
        page += '<table><thead><tr><th>Entry Date</th><th>Direction</th><th class="text-right">Score</th><th class="text-right">Entry</th><th class="text-right">Exit</th><th>Outcome</th><th class="text-right">P&L</th><th class="text-right">Bars</th></tr></thead><tbody>\n'
        for _, row in sorted_df.iterrows():
            page += f'<tr>'
            page += f'<td class="font-mono">{row["entry_time"].strftime("%Y-%m-%d %H:%M")}</td>'
            page += f'<td>{direction_badge(row["direction"])}</td>'
            page += f'<td class="text-right">{row["score"]:.1f}</td>'
            page += f'<td class="text-right font-mono">{row["entry_price"]:,.2f}</td>'
            page += f'<td class="text-right font-mono">{row["exit_price"]:,.2f}</td>'
            page += f'<td>{outcome_badge(row["outcome"])}</td>'
            page += f'<td class="text-right font-mono {pnl_class(row["pnl_dollars"])}">{fmt_money(row["pnl_dollars"])}</td>'
            page += f'<td class="text-right">{row["bars_held"]}</td>'
            page += '</tr>\n'
        page += '</tbody></table>\n'

    page += '</div>\n'  # end breakdowns tab

    # ━━━━━━━━━━ TAB 3: MONTHLY HEATMAP ━━━━━━━━━━
    page += '<div id="tab-monthly" class="tab-content">\n'
    page += '<h2>Monthly P&L Heatmap</h2>\n'

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Build monthly data
    monthly_data = {}
    for yr in years:
        ydf = tdf[tdf["year"] == yr]
        for m in range(1, 13):
            mdf = ydf[ydf["month"] == m]
            if not mdf.empty:
                monthly_data[(yr, m)] = mdf["pnl_dollars"].sum()

    max_abs_monthly = max(abs(v) for v in monthly_data.values()) if monthly_data else 1

    page += '<table><thead><tr><th>Year</th>'
    for mn in month_names:
        page += f'<th class="text-center">{mn}</th>'
    page += '<th class="text-right">Total</th></tr></thead><tbody>\n'

    for yr in years:
        page += f'<tr><td><strong>{yr}</strong></td>'
        yr_total = 0
        for m in range(1, 13):
            val = monthly_data.get((yr, m))
            if val is None:
                page += '<td class="text-center muted">—</td>'
            else:
                yr_total += val
                cls = heatmap_class(val, max_abs_monthly)
                page += f'<td class="text-right heatmap-cell {cls} {pnl_class(val)}">{fmt_money(val)}</td>'
        page += f'<td class="text-right font-mono" style="font-weight:700;{" color:var(--green)" if yr_total > 0 else " color:var(--red)"}">{fmt_money(yr_total)}</td>'
        page += '</tr>\n'
    page += '</tbody></table>\n'

    # Monthly trade count
    page += '<h2>Monthly Trade Count</h2>\n'
    page += '<table><thead><tr><th>Year</th>'
    for mn in month_names:
        page += f'<th class="text-center">{mn}</th>'
    page += '<th class="text-right">Total</th></tr></thead><tbody>\n'
    for yr in years:
        ydf = tdf[tdf["year"] == yr]
        page += f'<tr><td><strong>{yr}</strong></td>'
        yr_total_trades = 0
        for m in range(1, 13):
            mdf = ydf[ydf["month"] == m]
            cnt = len(mdf)
            yr_total_trades += cnt
            if cnt == 0:
                page += '<td class="text-center muted">—</td>'
            else:
                page += f'<td class="text-center">{cnt}</td>'
        page += f'<td class="text-right font-mono" style="font-weight:700">{yr_total_trades}</td>'
        page += '</tr>\n'
    page += '</tbody></table>\n'

    # Monthly win rate
    page += '<h2>Monthly Win Rate</h2>\n'
    page += '<table><thead><tr><th>Year</th>'
    for mn in month_names:
        page += f'<th class="text-center">{mn}</th>'
    page += '</tr></thead><tbody>\n'
    for yr in years:
        ydf = tdf[tdf["year"] == yr]
        page += f'<tr><td><strong>{yr}</strong></td>'
        for m in range(1, 13):
            mdf = ydf[ydf["month"] == m]
            if mdf.empty:
                page += '<td class="text-center muted">—</td>'
            else:
                mwr = (mdf["pnl_dollars"] > 0).mean() * 100
                cls = "positive" if mwr >= 50 else "negative"
                page += f'<td class="text-center {cls}">{mwr:.0f}%</td>'
        page += '</tr>\n'
    page += '</tbody></table>\n'

    page += '</div>\n'  # end monthly tab

    # ━━━━━━━━━━ TAB 4: TRADE LOG ━━━━━━━━━━
    page += '<div id="tab-tradelog" class="tab-content">\n'
    page += f'<h2>Complete Trade Log ({s["trades"]} Trades)</h2>\n'
    page += '<div class="trade-log-section">\n'
    page += '<table><thead><tr>'
    page += '<th>#</th><th>Entry Date</th><th>Exit Date</th><th>Dir</th><th class="text-right">Score</th>'
    page += '<th class="text-right">Entry</th><th class="text-right">TP</th><th class="text-right">SL</th>'
    page += '<th class="text-right">Exit</th><th>Outcome</th><th class="text-right">Contracts</th>'
    page += '<th class="text-right">P&L Pts</th><th class="text-right">P&L $</th>'
    page += '<th class="text-right">Account</th><th class="text-right">Bars</th>'

    # Runner columns if applicable
    has_runners = tdf["runner_exit_price"].notna().any()
    if has_runners:
        page += '<th class="text-right">Runner Exit</th><th class="text-right">Runner P&L</th><th>Runner Out</th>'

    page += '</tr></thead><tbody>\n'

    for idx, (_, row) in enumerate(tdf.iterrows(), 1):
        row_class = pnl_class(row["pnl_dollars"])
        page += '<tr>'
        page += f'<td class="muted">{idx}</td>'
        page += f'<td class="font-mono">{row["entry_time"].strftime("%Y-%m-%d %H:%M")}</td>'
        page += f'<td class="font-mono">{row["exit_time"].strftime("%Y-%m-%d %H:%M")}</td>'
        page += f'<td>{direction_badge(row["direction"])}</td>'
        page += f'<td class="text-right">{row["score"]:.1f}</td>'
        page += f'<td class="text-right font-mono">{row["entry_price"]:,.2f}</td>'
        page += f'<td class="text-right font-mono">{row["tp_price"]:,.2f}</td>'
        page += f'<td class="text-right font-mono">{row["sl_price"]:,.2f}</td>'
        page += f'<td class="text-right font-mono">{row["exit_price"]:,.2f}</td>'
        page += f'<td>{outcome_badge(row["outcome"])}</td>'
        page += f'<td class="text-right">{row["contracts"]}</td>'
        page += f'<td class="text-right font-mono {row_class}">{row["pnl_points"]:+.1f}</td>'
        page += f'<td class="text-right font-mono {row_class}">{fmt_money(row["pnl_dollars"])}</td>'
        page += f'<td class="text-right font-mono">${row["account_after"]:,.0f}</td>'
        page += f'<td class="text-right">{row["bars_held"]}</td>'

        if has_runners:
            if pd.notna(row["runner_exit_price"]):
                r_cls = pnl_class(row["runner_pnl"]) if pd.notna(row["runner_pnl"]) else ""
                page += f'<td class="text-right font-mono">{row["runner_exit_price"]:,.2f}</td>'
                page += f'<td class="text-right font-mono {r_cls}">{fmt_money(row["runner_pnl"])}</td>'
                page += f'<td>{outcome_badge(row["runner_outcome"])}</td>'
            else:
                page += '<td class="muted text-center">—</td><td class="muted text-center">—</td><td class="muted text-center">—</td>'

        page += '</tr>\n'

    page += '</tbody></table>\n'
    page += '</div>\n'  # end trade-log-section
    page += '</div>\n'  # end tradelog tab

    # ── Footer & JS ──
    page += f"""
<div class="footer">
  Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} &mdash; NQ Futures Backtest System &mdash; {html.escape(label)}
</div>

</div><!-- end container -->

<script>
function showTab(name) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}}
</script>
</body>
</html>"""

    return page


# ─── Generate all reports ───────────────────────────────────────────────────
output_dir = project_root / "reports"
output_dir.mkdir(exist_ok=True)

config_descs = {
    "2c (1TP+1R)": "2 Contracts: 1 takes profit at TP, 1 runner with trailing stop",
    "3c (2TP+1R)": "3 Contracts: 2 take profit at TP, 1 runner with trailing stop",
    "4c Simple":   "4 Contracts: all exit at TP/SL (dynamic sizing)",
}

for label, tdf in all_results.items():
    print(f"\nGenerating HTML for {label}...")
    html_content = generate_html(label, tdf, config_descs[label])
    safe_name = label.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
    filename = f"nq_report_{safe_name}.html"
    filepath = output_dir / filename
    filepath.write_text(html_content, encoding="utf-8")
    print(f"  Saved: {filepath}")

print(f"\nAll reports saved to: {output_dir}")
print("Done!")
