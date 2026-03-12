"""Check the paper trading database state."""
import sqlite3, os, json

db = os.path.join(os.environ.get("DATA_DIR", "."), "paper_trading.db")
print(f"DB path: {db}")
print(f"DB exists: {os.path.exists(db)}")

if not os.path.exists(db):
    print("No database found!")
    exit()

conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row

# Check trades
trades = conn.execute("SELECT * FROM trades ORDER BY entry_time DESC").fetchall()
print(f"\nTrades: {len(trades)}")
for t in trades:
    d = dict(t)
    pnl = d.get("pnl_dollars", 0) or 0
    print(f"  {d['direction']} @ {d['entry_price']} | {d['outcome']} | PnL: ${pnl:+,.0f} | Tier: {d.get('tier')} | {d['entry_time']}")

# Check positions
pos = conn.execute("SELECT data FROM positions LIMIT 1").fetchone()
if pos:
    p = json.loads(pos["data"])
    print(f"\nOpen position: {p['direction']} @ {p['entry_price']} | phase: {p['phase']} | bars: {p.get('bars_held', 0)}")
    print(f"  SL: {p['sl_price']} | TP: {p['tp_price']} | Score: {p.get('score')}")
else:
    print("\nNo open position")

# Check state
states = conn.execute("SELECT * FROM bot_state").fetchall()
print(f"\nBot state entries: {len(states)}")
for s in states:
    d = dict(s)
    val = d["value"]
    if len(val) > 100:
        val = val[:100] + "..."
    print(f"  {d['key']} = {val}")

conn.close()
