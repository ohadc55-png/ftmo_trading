"""SQLite database operations for the Paper Trading Bot."""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

# Railway volume mount or local fallback
DB_DIR = os.environ.get("DATA_DIR", str(Path(__file__).parent))
DB_PATH = os.path.join(DB_DIR, "paper_trading.db")


@contextmanager
def get_db():
    """Thread-safe database connection context manager."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                sl_price REAL NOT NULL,
                tp_price REAL NOT NULL,
                sl_distance REAL NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                score REAL NOT NULL,
                outcome TEXT,
                pnl_dollars REAL DEFAULT 0,
                pnl_points REAL DEFAULT 0,
                bars_held INTEGER DEFAULT 0,
                contracts INTEGER DEFAULT 2,
                runner_exit_price REAL,
                runner_pnl REAL,
                runner_outcome TEXT,
                account_after REAL,
                tier INTEGER DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS positions (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS bot_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS blocked_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                sl_distance REAL NOT NULL,
                bar_time TEXT NOT NULL,
                score REAL,
                tier INTEGER DEFAULT 1,
                reason TEXT NOT NULL,
                daily_pnl REAL DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_blocked_bar_time
                ON blocked_signals(bar_time);

            CREATE INDEX IF NOT EXISTS idx_trades_entry_time
                ON trades(entry_time);
            CREATE INDEX IF NOT EXISTS idx_trades_outcome
                ON trades(outcome);
        """)

        # Migration: add tier column to existing tables
        try:
            conn.execute("ALTER TABLE trades ADD COLUMN tier INTEGER DEFAULT 1")
        except sqlite3.OperationalError:
            pass  # Column already exists


# ── Trade Operations ────────────────────────────────────────────────────

def save_trade(trade: dict):
    """Insert a closed trade."""
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO trades
            (id, direction, entry_price, exit_price, sl_price, tp_price,
             sl_distance, entry_time, exit_time, score, outcome,
             pnl_dollars, pnl_points, bars_held, contracts,
             runner_exit_price, runner_pnl, runner_outcome, account_after, tier)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade["id"], trade["direction"], trade["entry_price"],
            trade.get("exit_price"), trade["sl_price"], trade["tp_price"],
            trade["sl_distance"], trade["entry_time"], trade.get("exit_time"),
            trade["score"], trade.get("outcome"),
            trade.get("pnl_dollars", 0), trade.get("pnl_points", 0),
            trade.get("bars_held", 0), trade.get("contracts", 2),
            trade.get("runner_exit_price"), trade.get("runner_pnl"),
            trade.get("runner_outcome"), trade.get("account_after"),
            trade.get("tier", 1),
        ))


def save_blocked_signal(signal: dict, reason: str, daily_pnl: float = 0):
    """Persist a blocked signal to the DB for historical analysis."""
    with get_db() as conn:
        conn.execute("""
            INSERT INTO blocked_signals
            (direction, entry_price, sl_distance, bar_time, score, tier, reason, daily_pnl)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.get("direction", ""),
            signal.get("entry_price", 0),
            signal.get("sl_distance", 0),
            signal.get("bar_time", ""),
            signal.get("score", 0),
            signal.get("tier", 1),
            reason,
            daily_pnl,
        ))


def get_trades(start_date: str = None, end_date: str = None) -> list[dict]:
    """Query trades with optional date filters."""
    with get_db() as conn:
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        if start_date:
            query += " AND entry_time >= ?"
            params.append(start_date)
        if end_date:
            query += " AND entry_time <= ?"
            params.append(end_date)
        query += " ORDER BY entry_time DESC"
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def get_today_trades(today_str: str) -> list[dict]:
    """Get all trades for a specific date (YYYY-MM-DD)."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE entry_time LIKE ? ORDER BY entry_time",
            (f"{today_str}%",)
        ).fetchall()
        return [dict(row) for row in rows]


def get_weekly_trades(week_start: str, week_end: str) -> list[dict]:
    """Get trades for a week range."""
    return get_trades(start_date=week_start, end_date=week_end)


# ── Position Operations ─────────────────────────────────────────────────

def save_position(position: dict | None):
    """Save (upsert) a single position by ID. Pass None to clear all."""
    with get_db() as conn:
        if position is None:
            conn.execute("DELETE FROM positions")
        else:
            conn.execute(
                "INSERT OR REPLACE INTO positions (id, data) VALUES (?, ?)",
                (position["id"], json.dumps(position))
            )


def delete_position(pos_id: str):
    """Delete a single position by ID."""
    with get_db() as conn:
        conn.execute("DELETE FROM positions WHERE id = ?", (pos_id,))


def get_position() -> dict | None:
    """Get first open position, or None. (backward compat)"""
    with get_db() as conn:
        row = conn.execute("SELECT data FROM positions LIMIT 1").fetchone()
        if row:
            return json.loads(row["data"])
        return None


def get_positions() -> list[dict]:
    """Get all open positions."""
    with get_db() as conn:
        rows = conn.execute("SELECT data FROM positions").fetchall()
        return [json.loads(row["data"]) for row in rows]


# ── Bot State Operations ────────────────────────────────────────────────

def save_state(key: str, value):
    """Save a key-value state entry."""
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO bot_state (key, value) VALUES (?, ?)",
            (key, json.dumps(value))
        )


def get_state(key: str, default=None):
    """Get a state value by key."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT value FROM bot_state WHERE key = ?", (key,)
        ).fetchone()
        if row:
            return json.loads(row["value"])
        return default


# ── Stats & Analytics ────────────────────────────────────────────────────

def get_all_stats() -> dict:
    """Compute comprehensive stats from all trades."""
    trades = get_trades()
    return _compute_stats(trades)


def get_stats_since(start_date: str) -> dict:
    """Compute stats from a start date."""
    trades = get_trades(start_date=start_date)
    return _compute_stats(trades)


def _compute_stats(trades: list[dict]) -> dict:
    """Internal stats computation."""
    if not trades:
        return {
            "total_trades": 0, "winners": 0, "losers": 0,
            "win_rate": 0, "total_pnl": 0, "profit_factor": 0,
            "avg_win": 0, "avg_loss": 0, "best_trade": 0, "worst_trade": 0,
            "max_drawdown": 0, "expectancy": 0,
            "max_consec_wins": 0, "max_consec_losses": 0,
            "avg_bars_held": 0, "trades": [],
        }

    pnls = [t["pnl_dollars"] for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    win_rate = len(winners) / len(pnls) * 100 if pnls else 0
    gross_profit = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_win = sum(winners) / len(winners) if winners else 0
    avg_loss = sum(losers) / len(losers) if losers else 0
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

    # Max drawdown
    capital = 100_000
    equity = [capital]
    for p in reversed(pnls):  # trades are DESC, reverse for chronological
        equity.append(equity[-1] + p)
    peak = equity[0]
    max_dd = 0
    for e in equity:
        if e > peak:
            peak = e
        dd = e - peak
        if dd < max_dd:
            max_dd = dd

    # Consecutive wins/losses
    max_w = max_l = cur_w = cur_l = 0
    for p in reversed(pnls):
        if p > 0:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)

    bars = [t["bars_held"] for t in trades if t["bars_held"]]
    avg_bars = sum(bars) / len(bars) if bars else 0

    return {
        "total_trades": len(pnls),
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "profit_factor": round(pf, 2) if pf != float("inf") else 999,
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "best_trade": max(pnls) if pnls else 0,
        "worst_trade": min(pnls) if pnls else 0,
        "max_drawdown": round(max_dd, 2),
        "expectancy": round(expectancy, 2),
        "max_consec_wins": max_w,
        "max_consec_losses": max_l,
        "avg_bars_held": round(avg_bars, 1),
        "trades": list(reversed(trades)),  # chronological
    }
