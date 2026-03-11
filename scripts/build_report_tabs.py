"""Build the 3 new HTML tabs from report_data.json and inject into the report."""

import json
import statistics

with open("reports/report_data.json") as f:
    d = json.load(f)

monthly = d["monthly"]
yearly = d["yearly"]
trades = d["trades"]
years = sorted(yearly.keys())
months = ["01","02","03","04","05","06","07","08","09","10","11","12"]
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
all_months = sorted(monthly.keys())

# ═══════════════════════════════════════════════
# TAB 1: MONTHLY HEATMAP
# ═══════════════════════════════════════════════
h = []
h.append('<h2>Monthly P&L Heatmap (Surge x1.2)</h2>')
h.append('<div class="info-box"><strong>Color coding:</strong> Green = profitable month, Red = losing month. Intensity reflects magnitude. Hover for details.</div>')

h.append('<div style="overflow-x:auto"><table style="text-align:center;min-width:1200px">')
h.append('<thead><tr><th>Year</th>')
for mn in month_names:
    h.append(f'<th class="text-center">{mn}</th>')
h.append('<th class="text-center" style="border-left:2px solid var(--border)">TOTAL</th>')
h.append('<th class="text-center">Trades</th>')
h.append('<th class="text-center">WR</th>')
h.append('</tr></thead><tbody>')

for year in years:
    yr = yearly[year]
    h.append(f'<tr><td style="font-weight:700">{year}</td>')
    for m in months:
        ym = f"{year}-{m}"
        if ym in monthly:
            v = monthly[ym]
            pnl = v["pnl"]
            if pnl > 0:
                intensity = min(pnl / 50000, 1.0)
                bg = f"rgba(63,185,80,{0.1 + intensity * 0.5:.2f})"
                cls = "positive"
            elif pnl < 0:
                intensity = min(abs(pnl) / 30000, 1.0)
                bg = f"rgba(248,81,73,{0.1 + intensity * 0.5:.2f})"
                cls = "negative"
            else:
                bg = "transparent"
                cls = "muted"
            h.append(f'<td class="text-center font-mono {cls}" style="background:{bg}" title="{v["trades"]} trades, WR {v["wr"]}%, PF {v["pf"]}">${pnl:+,.0f}</td>')
        else:
            h.append('<td class="text-center muted">-</td>')
    yr_cls = "positive" if yr["pnl"] > 0 else "negative"
    h.append(f'<td class="text-center font-mono {yr_cls}" style="border-left:2px solid var(--border);font-weight:700">${yr["pnl"]:+,.0f}</td>')
    h.append(f'<td class="text-center">{yr["trades"]}</td>')
    h.append(f'<td class="text-center">{yr["wr"]}%</td>')
    h.append('</tr>')

# Column totals
h.append('<tr style="border-top:2px solid var(--border);font-weight:700"><td>TOTAL</td>')
for m in months:
    total_m = sum(monthly.get(f"{y}-{m}", {}).get("pnl", 0) for y in years)
    cls = "positive" if total_m > 0 else "negative" if total_m < 0 else "muted"
    h.append(f'<td class="text-center font-mono {cls}">${total_m:+,.0f}</td>')
total_all = sum(yearly[y]["pnl"] for y in years)
total_trades_all = sum(yearly[y]["trades"] for y in years)
h.append(f'<td class="text-center font-mono positive" style="border-left:2px solid var(--border)">${total_all:+,.0f}</td>')
h.append(f'<td class="text-center">{total_trades_all}</td>')
h.append('<td class="text-center">-</td></tr></tbody></table></div>')

# Distribution stats
pnls_m = [monthly[ym]["pnl"] for ym in all_months]
pos_m = [p for p in pnls_m if p > 0]
neg_m = [p for p in pnls_m if p < 0]
h.append('<h3>Monthly Performance Distribution</h3>')
h.append('<div class="stats-grid">')
h.append(f'<div class="stat-card"><div class="label">Profitable Months</div><div class="value positive">{len(pos_m)}/{len(pnls_m)}</div><div class="delta">{len(pos_m)/len(pnls_m)*100:.0f}%</div></div>')
h.append(f'<div class="stat-card"><div class="label">Best Month</div><div class="value positive">${max(pnls_m):+,.0f}</div></div>')
h.append(f'<div class="stat-card"><div class="label">Worst Month</div><div class="value negative">${min(pnls_m):+,.0f}</div></div>')
h.append(f'<div class="stat-card"><div class="label">Avg Month</div><div class="value positive">${statistics.mean(pnls_m):+,.0f}</div></div>')
h.append(f'<div class="stat-card"><div class="label">Median Month</div><div class="value positive">${statistics.median(pnls_m):+,.0f}</div></div>')
h.append(f'<div class="stat-card"><div class="label">Avg Profitable</div><div class="value positive">${statistics.mean(pos_m):+,.0f}</div></div>')
h.append(f'<div class="stat-card"><div class="label">Avg Losing</div><div class="value negative">${statistics.mean(neg_m):+,.0f}</div></div>')
max_cp = max_cn = cp = cn = 0
for p in pnls_m:
    if p > 0: cp += 1; cn = 0; max_cp = max(max_cp, cp)
    else: cn += 1; cp = 0; max_cn = max(max_cn, cn)
h.append(f'<div class="stat-card"><div class="label">Max Consec Profit Months</div><div class="value positive">{max_cp}</div></div>')
h.append('</div>')

# Best/worst months
h.append('<div class="two-col">')
h.append('<div><h3>Top 10 Best Months</h3><table><thead><tr><th>Month</th><th class="text-right">P&L</th><th class="text-right">Trades</th><th class="text-right">WR</th><th class="text-right">PF</th></tr></thead><tbody>')
sorted_m = sorted(all_months, key=lambda x: monthly[x]["pnl"], reverse=True)
for ym in sorted_m[:10]:
    v = monthly[ym]
    h.append(f'<tr><td>{ym}</td><td class="text-right positive font-mono">${v["pnl"]:+,.0f}</td><td class="text-right">{v["trades"]}</td><td class="text-right">{v["wr"]}%</td><td class="text-right">{v["pf"]}</td></tr>')
h.append('</tbody></table></div>')

h.append('<div><h3>Top 10 Worst Months</h3><table><thead><tr><th>Month</th><th class="text-right">P&L</th><th class="text-right">Trades</th><th class="text-right">WR</th><th class="text-right">PF</th></tr></thead><tbody>')
for ym in sorted_m[-10:]:
    v = monthly[ym]
    h.append(f'<tr><td>{ym}</td><td class="text-right negative font-mono">${v["pnl"]:+,.0f}</td><td class="text-right">{v["trades"]}</td><td class="text-right">{v["wr"]}%</td><td class="text-right">{v["pf"]}</td></tr>')
h.append('</tbody></table></div></div>')

tab_heatmap = "\n".join(h)

# ═══════════════════════════════════════════════
# TAB 2: YEARLY STATS
# ═══════════════════════════════════════════════
h = []
h.append('<h2>Yearly Performance Breakdown (Surge x1.2)</h2>')

best_yr = max(years, key=lambda y: yearly[y]["pnl"])
worst_yr = min(years, key=lambda y: yearly[y]["pnl"])
best_pf = max(years, key=lambda y: yearly[y]["pf"])
all_positive = all(yearly[y]["pnl"] > 0 for y in years)
avg_pnl_yr = sum(yearly[y]["pnl"] for y in years) / len(years)

h.append('<div class="stats-grid">')
h.append(f'<div class="stat-card"><div class="label">Best Year</div><div class="value positive">{best_yr}</div><div class="delta">${yearly[best_yr]["pnl"]:+,.0f}</div></div>')
h.append(f'<div class="stat-card"><div class="label">Worst Year</div><div class="value neutral">{worst_yr}</div><div class="delta">${yearly[worst_yr]["pnl"]:+,.0f}</div></div>')
h.append(f'<div class="stat-card"><div class="label">Profitable Years</div><div class="value positive">{sum(1 for y in years if yearly[y]["pnl"] > 0)}/10</div><div class="delta">{"ALL profitable" if all_positive else ""}</div></div>')
h.append(f'<div class="stat-card"><div class="label">Best PF Year</div><div class="value positive">{best_pf}</div><div class="delta">PF {yearly[best_pf]["pf"]}</div></div>')
h.append(f'<div class="stat-card"><div class="label">Avg Yearly P&L</div><div class="value positive">${avg_pnl_yr:+,.0f}</div></div>')
h.append('</div>')

h.append('<table>')
h.append('<thead><tr><th>Year</th><th class="text-right">Trades</th><th class="text-right">Win Rate</th>')
h.append('<th class="text-right">Total P&L</th><th class="text-right">PF</th>')
h.append('<th class="text-right">Max DD</th><th class="text-right">Expectancy</th>')
h.append('<th class="text-right">Gross Profit</th><th class="text-right">Gross Loss</th>')
h.append('<th class="text-right">T1</th><th class="text-right">T2</th>')
h.append('<th class="text-right">BUY WR</th><th class="text-right">SELL WR</th>')
h.append('</tr></thead><tbody>')

for y in years:
    v = yearly[y]
    pnl_cls = "positive" if v["pnl"] > 0 else "negative"
    pf_cls = "positive" if v["pf"] >= 1.5 else "neutral" if v["pf"] >= 1.0 else "negative"
    wr_cls = "positive" if v["wr"] >= 35 else "neutral" if v["wr"] >= 30 else "negative"
    h.append(f'<tr><td style="font-weight:700">{y}</td>')
    h.append(f'<td class="text-right">{v["trades"]}</td>')
    h.append(f'<td class="text-right {wr_cls}">{v["wr"]}%</td>')
    h.append(f'<td class="text-right font-mono {pnl_cls}">${v["pnl"]:+,.0f}</td>')
    h.append(f'<td class="text-right {pf_cls}">{v["pf"]}</td>')
    h.append(f'<td class="text-right negative font-mono">${v["max_dd"]:+,.0f}</td>')
    h.append(f'<td class="text-right font-mono">${v["expectancy"]:+,.0f}</td>')
    h.append(f'<td class="text-right positive font-mono">${v["gross_profit"]:+,.0f}</td>')
    h.append(f'<td class="text-right negative font-mono">${v["gross_loss"]:,.0f}</td>')
    h.append(f'<td class="text-right">{v["t1"]}</td>')
    h.append(f'<td class="text-right">{v["t2"]}</td>')
    h.append(f'<td class="text-right">{v["buy_wr"]}%</td>')
    h.append(f'<td class="text-right">{v["sell_wr"]}%</td></tr>')

# Totals
tp = sum(yearly[y]["pnl"] for y in years)
tt = sum(yearly[y]["trades"] for y in years)
tgp = sum(yearly[y]["gross_profit"] for y in years)
tgl = sum(yearly[y]["gross_loss"] for y in years)
tw = sum(yearly[y]["wins"] for y in years)
tt1 = sum(yearly[y]["t1"] for y in years)
tt2 = sum(yearly[y]["t2"] for y in years)
h.append(f'<tr style="border-top:2px solid var(--border);font-weight:700"><td>TOTAL</td>')
h.append(f'<td class="text-right">{tt}</td><td class="text-right">{tw/tt*100:.1f}%</td>')
h.append(f'<td class="text-right font-mono positive">${tp:+,.0f}</td><td class="text-right">{tgp/tgl:.2f}</td>')
h.append(f'<td class="text-right">-</td><td class="text-right font-mono">${tp/tt:+,.0f}</td>')
h.append(f'<td class="text-right positive font-mono">${tgp:+,.0f}</td><td class="text-right negative font-mono">${tgl:,.0f}</td>')
h.append(f'<td class="text-right">{tt1}</td><td class="text-right">{tt2}</td>')
h.append('<td class="text-right">-</td><td class="text-right">-</td></tr></tbody></table>')

# Equity growth
h.append('<h3>Cumulative Equity by Year</h3>')
h.append('<table><thead><tr><th>Year</th><th class="text-right">Start Balance</th><th class="text-right">Year P&L</th><th class="text-right">End Balance</th><th class="text-right">Annual Return</th></tr></thead><tbody>')
balance = 100000
for y in years:
    start = balance
    pnl = yearly[y]["pnl"]
    balance += pnl
    ret = pnl / start * 100
    h.append(f'<tr><td style="font-weight:700">{y}</td>')
    h.append(f'<td class="text-right font-mono">${start:,.0f}</td>')
    h.append(f'<td class="text-right font-mono positive">${pnl:+,.0f}</td>')
    h.append(f'<td class="text-right font-mono">${balance:,.0f}</td>')
    h.append(f'<td class="text-right positive">{ret:+.1f}%</td></tr>')
h.append('</tbody></table>')

tab_yearly = "\n".join(h)

# ═══════════════════════════════════════════════
# TAB 3: TRADE LOG
# ═══════════════════════════════════════════════
h = []
h.append('<h2>Full Trade Log (Surge x1.2) &mdash; 11,687 Trades</h2>')

# Filters
h.append('''<div style="margin-bottom:16px;display:flex;gap:12px;flex-wrap:wrap;align-items:center">
<label style="color:var(--text-muted);font-size:13px">Filter:</label>
<select id="flt-dir" onchange="filterTrades()" style="background:var(--card-bg);color:var(--text);border:1px solid var(--border);padding:6px 10px;border-radius:6px;font-size:13px">
<option value="all">All Directions</option><option value="buy">BUY only</option><option value="sell">SELL only</option></select>
<select id="flt-outcome" onchange="filterTrades()" style="background:var(--card-bg);color:var(--text);border:1px solid var(--border);padding:6px 10px;border-radius:6px;font-size:13px">
<option value="all">All Outcomes</option><option value="TP">TP</option><option value="SL">SL</option><option value="timeout">Timeout</option><option value="TP+trail">TP+Trail</option><option value="TP+BE">TP+BE</option></select>
<select id="flt-tier" onchange="filterTrades()" style="background:var(--card-bg);color:var(--text);border:1px solid var(--border);padding:6px 10px;border-radius:6px;font-size:13px">
<option value="all">All Tiers</option><option value="1">T1 only</option><option value="2">T2 only</option></select>
<select id="flt-year" onchange="filterTrades()" style="background:var(--card-bg);color:var(--text);border:1px solid var(--border);padding:6px 10px;border-radius:6px;font-size:13px">
<option value="all">All Years</option>''')
for y in range(2016, 2026):
    h.append(f'<option value="{y}">{y}</option>')
h.append('''</select>
<select id="flt-pnl" onchange="filterTrades()" style="background:var(--card-bg);color:var(--text);border:1px solid var(--border);padding:6px 10px;border-radius:6px;font-size:13px">
<option value="all">All P&L</option><option value="pos">Winners only</option><option value="neg">Losers only</option></select>
<span id="flt-count" style="color:var(--blue);font-size:13px;margin-left:auto"></span>
<span id="flt-stats" style="color:var(--text-muted);font-size:12px;margin-left:8px"></span>
</div>''')

# Pagination
h.append('''<div style="margin-bottom:12px;display:flex;gap:8px;align-items:center">
<button onclick="prevPage()" style="background:var(--card-bg);color:var(--text);border:1px solid var(--border);padding:6px 14px;border-radius:6px;cursor:pointer;font-size:13px">&lt; Prev</button>
<span id="page-info" style="color:var(--text-muted);font-size:13px"></span>
<button onclick="nextPage()" style="background:var(--card-bg);color:var(--text);border:1px solid var(--border);padding:6px 14px;border-radius:6px;cursor:pointer;font-size:13px">Next &gt;</button>
<select id="page-size" onchange="filterTrades()" style="background:var(--card-bg);color:var(--text);border:1px solid var(--border);padding:6px 10px;border-radius:6px;font-size:13px">
<option value="50">50/page</option><option value="100" selected>100/page</option><option value="250">250/page</option><option value="500">500/page</option></select>
</div>''')

# Table
h.append('''<table id="trade-table">
<thead><tr>
<th>#</th><th>Entry Time</th><th>Exit Time</th><th>Dir</th><th>Tier</th>
<th class="text-right">Score</th><th class="text-right">Entry</th><th class="text-right">Exit</th>
<th class="text-right">SL Dist</th><th class="text-right">ATR</th>
<th class="text-right">P&L</th><th>Outcome</th><th class="text-right">Bars</th>
<th class="text-right">Ctrs</th><th class="text-right">Account</th>
</tr></thead>
<tbody id="trade-tbody"></tbody>
</table>''')

tab_tradelog = "\n".join(h)

# Trade data as JSON (separate file to keep HTML manageable)
trades_json = json.dumps(trades)

# ═══════════════════════════════════════════════
# NOW: Read the existing HTML and inject tabs
# ═══════════════════════════════════════════════
with open("reports/surge_optimization_1.5_vs_1.2.html", "r", encoding="utf-8") as f:
    html = f.read()

# Add new tab buttons
old_tabs = '''<button class="tab-btn" onclick="showTab('config')">Configuration</button>'''
new_tabs = '''<button class="tab-btn" onclick="showTab('config')">Configuration</button>
  <button class="tab-btn" onclick="showTab('heatmap')">Monthly Heatmap</button>
  <button class="tab-btn" onclick="showTab('yearly')">Yearly Stats</button>
  <button class="tab-btn" onclick="showTab('tradelog')">Trade Log</button>'''
html = html.replace(old_tabs, new_tabs)

# Insert new tab content before the footer
footer_marker = '<div class="footer">'
new_content = f'''
<!-- =================== MONTHLY HEATMAP TAB =================== -->
<div id="tab-heatmap" class="tab-content">
{tab_heatmap}
</div>

<!-- =================== YEARLY STATS TAB =================== -->
<div id="tab-yearly" class="tab-content">
{tab_yearly}
</div>

<!-- =================== TRADE LOG TAB =================== -->
<div id="tab-tradelog" class="tab-content">
{tab_tradelog}
</div>

<div class="footer">'''

html = html.replace(footer_marker, new_content)

# Replace the script section with enhanced version that includes trade log JS
old_script = """<script>
function showTab(name) {
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}
</script>"""

new_script = f"""<script>
var ALL_TRADES = {trades_json};
var currentPage = 0;
var filteredTrades = ALL_TRADES;

function showTab(name) {{
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
  if (name === 'tradelog' && document.getElementById('trade-tbody').innerHTML === '') {{
    filterTrades();
  }}
}}

function filterTrades() {{
  var dir = document.getElementById('flt-dir').value;
  var outcome = document.getElementById('flt-outcome').value;
  var tier = document.getElementById('flt-tier').value;
  var year = document.getElementById('flt-year').value;
  var pnl = document.getElementById('flt-pnl').value;

  filteredTrades = ALL_TRADES.filter(function(t) {{
    if (dir !== 'all' && t.direction !== dir) return false;
    if (outcome !== 'all' && t.outcome !== outcome) return false;
    if (tier !== 'all' && String(t.tier) !== tier) return false;
    if (year !== 'all' && !t.time.startsWith(year)) return false;
    if (pnl === 'pos' && t.pnl <= 0) return false;
    if (pnl === 'neg' && t.pnl >= 0) return false;
    return true;
  }});

  currentPage = 0;
  renderPage();
}}

function renderPage() {{
  var pageSize = parseInt(document.getElementById('page-size').value);
  var start = currentPage * pageSize;
  var end = Math.min(start + pageSize, filteredTrades.length);
  var totalPages = Math.max(1, Math.ceil(filteredTrades.length / pageSize));

  document.getElementById('flt-count').textContent = filteredTrades.length + ' trades matched';
  document.getElementById('page-info').textContent = 'Page ' + (currentPage + 1) + ' of ' + totalPages + ' (' + (start+1) + '-' + end + ')';

  var totalPnl = 0, wins = 0;
  filteredTrades.forEach(function(t) {{ totalPnl += t.pnl; if (t.pnl > 0) wins++; }});
  var wr = filteredTrades.length > 0 ? (wins / filteredTrades.length * 100).toFixed(1) : 0;
  document.getElementById('flt-stats').textContent = 'P&L: $' + Math.round(totalPnl).toLocaleString() + ' | WR: ' + wr + '%';

  var tbody = document.getElementById('trade-tbody');
  var rows = '';
  for (var i = start; i < end; i++) {{
    var t = filteredTrades[i];
    var pnlCls = t.pnl > 0 ? 'positive' : t.pnl < 0 ? 'negative' : 'muted';
    var dirStyle = t.direction === 'buy' ? 'color:var(--green)' : 'color:var(--red)';
    var ocStyle = t.outcome === 'SL' ? 'background:rgba(248,81,73,0.2);color:var(--red)' :
                  t.outcome.startsWith('TP') ? 'background:rgba(63,185,80,0.2);color:var(--green)' :
                  'background:rgba(210,153,34,0.2);color:var(--orange)';
    var fmt = function(n) {{ return n.toLocaleString('en-US', {{minimumFractionDigits:2, maximumFractionDigits:2}}); }};
    var fmtI = function(n) {{ return (n >= 0 ? '+' : '') + Math.round(n).toLocaleString(); }};

    rows += '<tr>' +
      '<td class="muted">' + (i+1) + '</td>' +
      '<td class="font-mono" style="font-size:11px">' + t.time.substring(0,19) + '</td>' +
      '<td class="font-mono" style="font-size:11px">' + t.exit_time.substring(0,19) + '</td>' +
      '<td style="font-weight:700;' + dirStyle + '">' + t.direction.toUpperCase() + '</td>' +
      '<td class="text-center">T' + t.tier + '</td>' +
      '<td class="text-right">' + t.score.toFixed(1) + '</td>' +
      '<td class="text-right font-mono">' + fmt(t.entry) + '</td>' +
      '<td class="text-right font-mono">' + fmt(t.exit) + '</td>' +
      '<td class="text-right font-mono">' + t.sl_dist.toFixed(1) + '</td>' +
      '<td class="text-right font-mono">' + t.atr.toFixed(1) + '</td>' +
      '<td class="text-right font-mono ' + pnlCls + '" style="font-weight:700">$' + fmtI(t.pnl) + '</td>' +
      '<td><span class="badge" style="' + ocStyle + '">' + t.outcome + '</span></td>' +
      '<td class="text-right">' + t.bars + '</td>' +
      '<td class="text-center">' + t.contracts + '</td>' +
      '<td class="text-right font-mono">$' + Math.round(t.account).toLocaleString() + '</td>' +
      '</tr>';
  }}
  tbody.innerHTML = rows;
}}

function nextPage() {{
  var pageSize = parseInt(document.getElementById('page-size').value);
  var totalPages = Math.ceil(filteredTrades.length / pageSize);
  if (currentPage < totalPages - 1) {{ currentPage++; renderPage(); }}
}}
function prevPage() {{
  if (currentPage > 0) {{ currentPage--; renderPage(); }}
}}
</script>"""

html = html.replace(old_script, new_script)

with open("reports/surge_optimization_1.5_vs_1.2.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Report updated with 3 new tabs!")
print(f"  - Monthly Heatmap: {len(all_months)} months")
print(f"  - Yearly Stats: {len(years)} years")
print(f"  - Trade Log: {len(trades)} trades (with filters + pagination)")
