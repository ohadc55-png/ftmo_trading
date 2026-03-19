"""NQ Futures Paper Trading Bot — Flask Web App.

Runs on Railway (or locally). Polls Yahoo Finance every 5 minutes,
runs the V3 strategy pipeline, manages virtual positions, and serves
a live dashboard accessible from any device.
"""

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from flask import Flask, jsonify, render_template, request

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from webapp.models import (
    init_db, save_trade, get_trades, get_today_trades,
    save_position, get_position, get_positions, delete_position,
    save_state, get_state,
    get_all_stats, save_blocked_signal, get_blocked_signals_today,
)
from webapp.strategy_runner import (
    StrategyRunner, PositionManager, DailyPnLTracker,
    POINT_VALUE, TIER1_MAX_SL, TIER2_MAX_SL, SMART_DL, STARTING_CAPITAL,
)
from webapp.email_service import send_weekly_email, send_test_email

# ── Logging ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("paper_bot")

ET = ZoneInfo("America/New_York")

# ── Flask App ────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))

# ── Bot State ────────────────────────────────────────────────────────────

runner = StrategyRunner()
pos_mgr = PositionManager()
pnl_tracker = DailyPnLTracker()
blocked_signals: list[dict] = []  # Signals blocked by SL cap / SmartDL (today only)
bot_status = {
    "running": False,
    "last_cycle": None,
    "last_error": None,
    "cycles_count": 0,
    "start_time": None,
}

# Thread lock for shared state (pos_mgr, pnl_tracker, blocked_signals, bot_status)
state_lock = threading.Lock()

# Admin secret from env var (fallback to hardcoded for backward compat)
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "nq_seed_2026")


def _check_admin(data) -> bool:
    """Verify admin secret from request data."""
    return data and data.get("secret") == ADMIN_SECRET


# ═══════════════════════════════════════════════════════════════════════════
# Background Strategy Loop
# ═══════════════════════════════════════════════════════════════════════════

WEEKEND_MAX_POSITIONS = 1  # Max positions allowed over the weekend
FRIDAY_TRIM_HOUR = 16     # Hour ET to start trimming excess positions on Friday
FRIDAY_TRIM_MINUTE = 50   # Minute to start trimming


def is_friday_weekend_block() -> bool:
    """Check if we should block new signals due to weekend position limit."""
    now = datetime.now(ET)
    return now.weekday() == 4  # Friday


def is_friday_trim_time() -> bool:
    """Check if it's time to trim excess positions before Friday close."""
    now = datetime.now(ET)
    if now.weekday() != 4:
        return False
    return (now.hour > FRIDAY_TRIM_HOUR or
            (now.hour == FRIDAY_TRIM_HOUR and now.minute >= FRIDAY_TRIM_MINUTE))


def is_market_hours() -> bool:
    """Check if within NQ futures trading hours (ET).

    NQ futures trade Sun 18:00 - Fri 17:00 ET (with daily halt 17:00-18:00).
    We use 07:00-23:00 for active strategy but need data flowing outside that.
    """
    now = datetime.now(ET)
    wd = now.weekday()
    if wd == 5:  # Saturday — fully closed
        return False
    if wd == 6:  # Sunday — futures open at 18:00 ET
        return now.hour >= 18
    return 7 <= now.hour < 23


def is_futures_open() -> bool:
    """Check if NQ futures are actually trading (nearly 24h on weekdays).

    Used for position management — SL/TP must be checked whenever futures trade.
    Futures halt daily 17:00-18:00 ET and are closed Sat.
    """
    now = datetime.now(ET)
    wd = now.weekday()
    if wd == 5:  # Saturday
        return False
    if wd == 6:  # Sunday — opens 18:00 ET
        return now.hour >= 18
    if wd == 4:  # Friday — closes 17:00 ET
        return now.hour < 17
    # Mon-Thu: halt 17:00-18:00 ET
    return not (now.hour == 17)


def strategy_loop():
    """Background thread: polls every 5 minutes during market hours."""
    bot_status["running"] = True
    bot_status["start_time"] = datetime.now(ET).isoformat()
    logger.info("Strategy loop started")

    # Restore state from DB
    _restore_state()

    while True:
        try:
            _wait_for_next_bar()

            if not is_futures_open():
                logger.debug("Futures closed, sleeping...")
                time.sleep(60)
                continue

            _run_cycle()

        except Exception as e:
            logger.error(f"Strategy loop error: {e}", exc_info=True)
            bot_status["last_error"] = str(e)
            time.sleep(60)


def _wait_for_next_bar():
    """Sleep until 5 seconds after the next 5-minute boundary."""
    now = datetime.now(ET)
    seconds_into_period = (now.minute % 5) * 60 + now.second
    seconds_remaining = 300 - seconds_into_period
    sleep_time = max(seconds_remaining + 5, 1)

    if sleep_time > 10:
        logger.debug(f"Waiting {sleep_time}s for next bar...")

    # Sleep in 1-second increments for responsiveness
    for _ in range(int(sleep_time)):
        time.sleep(1)


def _run_cycle():
    """Execute one strategy cycle."""
    with state_lock:
        _run_cycle_locked()


def _run_cycle_locked():
    """Execute one strategy cycle (called under state_lock)."""
    logger.info("=" * 50)
    logger.info("Running strategy cycle...")

    # Check for new trading day
    today = datetime.now(ET).strftime("%Y-%m-%d")
    last_day = get_state("last_trading_day")
    if last_day != today:
        logger.info(f"New trading day: {today}")
        save_state("last_trading_day", today)

    # 1. Fetch data and run pipeline
    df = None
    for attempt in range(3):
        df = runner.fetch_and_run_pipeline()
        if df is not None:
            break
        logger.warning(f"Fetch attempt {attempt + 1}/3 failed, retrying in 30s...")
        time.sleep(30)

    if df is None:
        logger.error("All fetch attempts failed, skipping cycle")
        bot_status["last_error"] = "Data fetch failed"
        return

    current_price = runner.get_current_price()
    last_bar = runner.get_last_bar()

    # 2. Check stale data
    if last_bar:
        bar_time = datetime.fromisoformat(last_bar["time"])
        age = (datetime.now(ET) - bar_time).total_seconds()
        if age > 900:
            logger.warning(f"Data stale: latest bar is {age/60:.0f} min old")

    # 3. Update ALL open positions on new bar
    if pos_mgr.has_open_position() and last_bar:
        closed_positions = pos_mgr.update_all_on_bar(last_bar)

        # Process all positions that just closed
        for closed_pos, event in closed_positions:
            _save_closed_trade(closed_pos)

    # 3b. Friday weekend trim: close excess positions before market close
    if is_friday_trim_time() and len(pos_mgr.open_positions) > WEEKEND_MAX_POSITIONS and current_price:
        _trim_weekend_positions(current_price, last_bar)

    # 4. Save all open position states
    for pos in pos_mgr.get_open_positions():
        save_position(pos)

    # 5. Check for new signal (only during strategy hours 07-23 ET)
    if not is_market_hours():
        logger.info("Outside strategy hours (07-23 ET) — skipping signal check, positions managed")
        # Still update bot status below
        signal = None
    else:
        signal = runner.check_last_bar_signal(df)

    last_bar_time = get_state("last_signal_bar_time")

    if signal and signal["bar_time"] != last_bar_time:
        # Reset blocked list on new day
        today_str = datetime.now(ET).strftime("%Y-%m-%d")
        blocked_signals[:] = [b for b in blocked_signals if b.get("date") == today_str]

        # Signal blocked by SL cap / tier rules
        if signal.get("blocked"):
            signal["date"] = today_str
            blocked_signals.append(signal)
            save_blocked_signal(signal, signal["blocked"])
            save_state("last_signal_bar_time", signal["bar_time"])
            logger.info(f"Signal BLOCKED ({signal['blocked']}): {signal['direction'].upper()} @ {signal['entry_price']:.2f}")
        # Weekend limit: block new trades on Friday if already at max positions
        elif is_friday_weekend_block() and len(pos_mgr.open_positions) >= WEEKEND_MAX_POSITIONS:
            n_open = len(pos_mgr.open_positions)
            signal["blocked"] = f"Weekend limit ({n_open} pos open, max {WEEKEND_MAX_POSITIONS})"
            signal["date"] = today_str
            blocked_signals.append(signal)
            save_blocked_signal(signal, signal["blocked"])
            save_state("last_signal_bar_time", signal["bar_time"])
            logger.info(f"Signal BLOCKED: Weekend limit ({n_open} positions open, max {WEEKEND_MAX_POSITIONS} on Friday)")
        # Smart DL: block new trades if (realized + unrealized) daily P&L <= -$1,100
        # Open positions always run to completion. If P&L recovers, trading resumes.
        else:
            unrealized = pos_mgr.get_total_unrealized_pnl(current_price) if current_price and pos_mgr.has_open_position() else 0
            if not pnl_tracker.can_take_trade(unrealized):
                realized = pnl_tracker.get_today_pnl()
                total = realized + unrealized
                signal["blocked"] = f"SmartDL (realized ${realized:+,.0f} + unrealized ${unrealized:+,.0f} = ${total:+,.0f})"
                signal["date"] = today_str
                blocked_signals.append(signal)
                save_blocked_signal(signal, signal["blocked"], total)
                save_state("last_signal_bar_time", signal["bar_time"])
                logger.info(f"Signal BLOCKED: Smart DL (realized: ${realized:+,.0f} + unrealized: ${unrealized:+,.0f} = ${total:+,.0f})")
            else:
                # Get current account
                stats = get_all_stats()
                account = STARTING_CAPITAL + stats["total_pnl"]

                pos = pos_mgr.open_position(signal, account=account)
                save_position(pos)
                save_state("last_signal_bar_time", signal["bar_time"])
                logger.info(f"Signal TAKEN: {signal['direction'].upper()} @ {signal['entry_price']:.2f} | Open positions: {len(pos_mgr.open_positions)}")

    # 6. Update bot status
    bot_status["last_cycle"] = datetime.now(ET).isoformat()
    bot_status["cycles_count"] += 1
    bot_status["last_error"] = None

    # Save state
    save_state("pnl_tracker", pnl_tracker.to_dict())

    price_str = f"{current_price:,.2f}" if current_price else "N/A"
    pos_str = "FLAT"
    if pos_mgr.has_open_position():
        n_pos = len(pos_mgr.open_positions)
        unr = pos_mgr.get_total_unrealized_pnl(current_price) if current_price else 0
        pos_str = f"{n_pos} positions open, unrealized: ${unr:+,.0f}, exposure: ${pos_mgr.get_worst_case_loss():,.0f}"

    unr_total = pos_mgr.get_total_unrealized_pnl(current_price) if current_price and pos_mgr.has_open_position() else 0
    logger.info(
        f"Cycle #{bot_status['cycles_count']} | NQ {price_str} | {pos_str} | "
        f"Daily: ${pnl_tracker.get_today_pnl():+,.0f} realized + ${unr_total:+,.0f} unrealized | "
        f"Status: {pnl_tracker.get_today_status(unr_total)}"
    )


def _trim_weekend_positions(current_price: float, last_bar: dict):
    """Trim excess positions before Friday close. Keep earliest entry only."""
    positions = pos_mgr.get_open_positions()
    if len(positions) <= WEEKEND_MAX_POSITIONS:
        return

    # Sort by entry time — keep the earliest
    sorted_pos = sorted(positions, key=lambda p: p.get("entry_time", ""))
    trim = sorted_pos[WEEKEND_MAX_POSITIONS:]

    bar_time = last_bar.get("time", str(datetime.now(ET))) if last_bar else str(datetime.now(ET))

    logger.info(
        f"WEEKEND TRIM: {len(trim)} excess position(s) to close "
        f"(keeping {WEEKEND_MAX_POSITIONS} earliest)"
    )

    for pos in trim:
        closed = pos_mgr.force_close_at_market(pos["id"], current_price, bar_time)
        if closed:
            _save_closed_trade(closed)


def _save_closed_trade(closed_pos: dict):
    """Record a closed position to DB and update P&L tracker."""
    pnl_tracker.record_pnl(closed_pos["total_pnl"], closed_pos.get("entry_time"))
    save_trade({
        "id": closed_pos["id"],
        "direction": closed_pos["direction"],
        "entry_price": closed_pos["entry_price"],
        "exit_price": closed_pos.get("exit_price"),
        "sl_price": closed_pos["sl_price"],
        "tp_price": closed_pos["tp_price"],
        "sl_distance": closed_pos["sl_distance"],
        "entry_time": closed_pos["entry_time"],
        "exit_time": closed_pos.get("exit_time"),
        "score": closed_pos["score"],
        "outcome": closed_pos.get("outcome", ""),
        "pnl_dollars": closed_pos["total_pnl"],
        "pnl_points": closed_pos.get("pnl_points", 0),
        "bars_held": closed_pos.get("bars_held", 0),
        "contracts": closed_pos.get("contracts", 2),
        "runner_exit_price": closed_pos.get("runner_exit_price"),
        "runner_pnl": closed_pos.get("runner_pnl"),
        "runner_outcome": closed_pos.get("runner_outcome"),
        "account_after": closed_pos.get("account_after"),
        "tier": closed_pos.get("tier", 1),
    })
    delete_position(closed_pos["id"])
    logger.info(f"Trade saved to DB: {closed_pos['outcome']} ${closed_pos['total_pnl']:+,.0f}")


def _restore_state():
    """Restore bot state from DB on startup."""
    # Restore all open positions
    all_positions = get_positions()
    for pos_data in all_positions:
        if pos_data.get("phase") != "closed":
            pos_mgr.open_positions[pos_data["id"]] = pos_data
            logger.info(f"Restored open position: {pos_data['direction']} @ {pos_data['entry_price']}")
    if all_positions:
        logger.info(f"Restored {len(pos_mgr.open_positions)} open position(s)")

    # Restore daily P&L tracker
    pnl_data = get_state("pnl_tracker")
    if pnl_data:
        pnl_tracker.from_dict(pnl_data)
        logger.info(f"Restored daily P&L: {pnl_tracker.daily_pnl}")

    # Restore today's blocked signals
    today_blocked = get_blocked_signals_today()
    for b in today_blocked:
        blocked_signals.append({
            "direction": b["direction"],
            "entry_price": b["entry_price"],
            "sl_distance": b["sl_distance"],
            "bar_time": b["bar_time"],
            "score": b.get("score", 0),
            "tier": b.get("tier", 1),
            "blocked": b["reason"],
            "date": b["bar_time"][:10] if b.get("bar_time") else "",
        })
    if today_blocked:
        logger.info(f"Restored {len(today_blocked)} blocked signal(s) from DB")

    # Initial data fetch so dashboard has chart data immediately
    logger.info("Initial data fetch for dashboard...")
    df = runner.fetch_and_run_pipeline()
    if df is not None:
        logger.info("Initial fetch complete — chart data available")
    else:
        logger.warning("Initial fetch failed — chart will load on next cycle")


def _weekly_email_job():
    """Background thread for weekly email. Runs Friday 17:00 ET."""
    while True:
        now = datetime.now(ET)
        # Check if Friday after 17:00
        if now.weekday() == 4 and now.hour == 17 and now.minute < 5:
            try:
                send_weekly_email()
            except Exception as e:
                logger.error(f"Weekly email failed: {e}", exc_info=True)
            time.sleep(3600)  # Wait 1 hour to avoid duplicate
        else:
            time.sleep(60)  # Check every minute


# ═══════════════════════════════════════════════════════════════════════════
# Flask Routes
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/")
def dashboard():
    """Main dashboard page."""
    stats = get_all_stats()
    today = datetime.now(ET).strftime("%Y-%m-%d")
    today_trades = get_today_trades(today)

    with state_lock:
        positions = pos_mgr.get_open_positions()
        position = positions[0] if positions else None  # backward compat for template
        current_price = runner.get_current_price()
        unrealized = pos_mgr.get_unrealized_pnl(current_price) if current_price and positions else None
        today_blocked = [b for b in blocked_signals if b.get("date") == today]

    # Equity curve data
    equity_data = _build_equity_data(stats.get("trades", []))
    candle_data = runner.get_recent_candles(500, tf_minutes=1)

    return render_template("dashboard.html",
        stats=stats,
        position=position,
        positions=positions,
        current_price=current_price,
        unrealized=unrealized,
        today_trades=today_trades,
        daily_pnl=pnl_tracker.get_today_pnl(),
        daily_status=pnl_tracker.get_today_status(unrealized or 0),
        budget_remaining=pnl_tracker.get_budget_remaining(unrealized or 0),
        bot_status=dict(bot_status),
        is_market_open=is_market_hours(),
        equity_data=json.dumps(equity_data),
        candle_data=json.dumps(candle_data),
        blocked_signals=json.dumps(today_blocked),
        now=datetime.now(ET),
    )


@app.route("/api/status")
def api_status():
    """JSON endpoint for AJAX auto-refresh."""
    current_price = runner.get_live_price()

    with state_lock:
        positions = pos_mgr.get_open_positions()
        unrealized = pos_mgr.get_total_unrealized_pnl(current_price) if current_price and positions else None

        # Per-position unrealized
        positions_with_pnl = []
        for pos in positions:
            pos_copy = dict(pos)
            if current_price:
                pos_copy["unrealized"] = pos_mgr.get_position_unrealized_pnl(pos, current_price)
            positions_with_pnl.append(pos_copy)

        worst_case = pos_mgr.get_worst_case_loss()
        daily_pnl = pnl_tracker.get_today_pnl()
        unr_for_dl = unrealized or 0
        daily_status = pnl_tracker.get_today_status(unr_for_dl)
        budget_remaining = pnl_tracker.get_budget_remaining(unr_for_dl)

        today_str = datetime.now(ET).strftime("%Y-%m-%d")
        today_blocked = [b for b in blocked_signals if b.get("date") == today_str]
        status_snapshot = dict(bot_status)

    stats = get_all_stats()

    return jsonify({
        "price": current_price,
        "position": positions_with_pnl[0] if positions_with_pnl else None,  # backward compat
        "positions": positions_with_pnl,
        "positions_count": len(positions_with_pnl),
        "unrealized": unrealized,
        "worst_case_exposure": worst_case,
        "daily_pnl": daily_pnl,
        "daily_status": daily_status,
        "budget_remaining": budget_remaining,
        "total_pnl": stats["total_pnl"],
        "total_trades": stats["total_trades"],
        "win_rate": stats["win_rate"],
        "profit_factor": stats["profit_factor"],
        "expectancy": stats["expectancy"],
        "max_drawdown": stats["max_drawdown"],
        "avg_pnl": round(stats["total_pnl"] / stats["total_trades"], 0) if stats["total_trades"] > 0 else 0,
        "max_consec_wins": stats["max_consec_wins"],
        "max_consec_losses": stats["max_consec_losses"],
        "trades": stats["trades"],
        "bot": status_snapshot,
        "is_market_open": is_market_hours(),
        "blocked_signals": today_blocked,
        "data_source": runner._data_source,
        "timestamp": datetime.now(ET).isoformat(),
    })


@app.route("/api/candles")
def api_candles():
    """Return recent candle data for the price chart."""
    count = request.args.get("count", 120, type=int)
    tf = request.args.get("tf", 1, type=int)
    if tf not in (1, 5, 15, 60):
        tf = 1
    candles = runner.get_recent_candles(min(count, 1000), tf_minutes=tf)
    return jsonify(candles)


@app.route("/trades")
def trades_page():
    """Full trade history."""
    trades = get_trades()
    stats = get_all_stats()
    return render_template("dashboard.html",
        stats=stats,
        position=None,
        positions=[],
        current_price=runner.get_current_price(),
        unrealized=None,
        today_trades=[],
        daily_pnl=pnl_tracker.get_today_pnl(),
        daily_status=pnl_tracker.get_today_status(0),
        budget_remaining=pnl_tracker.get_budget_remaining(0),
        bot_status=bot_status,
        is_market_open=is_market_hours(),
        equity_data=json.dumps(_build_equity_data(stats.get("trades", []))),
        candle_data=json.dumps(runner.get_recent_candles(120)),
        blocked_signals=json.dumps([]),
        now=datetime.now(ET),
        show_all_trades=True,
    )


@app.route("/test-email")
def test_email_route():
    """Send a test email to verify configuration."""
    success, message = send_test_email()
    return jsonify({"success": success, "message": message})


@app.route("/health")
def health():
    """Health check for Railway."""
    return jsonify({"status": "ok", "timestamp": datetime.now(ET).isoformat()})


def _build_equity_data(trades: list[dict]) -> list[dict]:
    """Build equity curve data for the chart."""
    capital = STARTING_CAPITAL
    data = [{"date": "Start", "equity": capital, "pnl": 0}]

    for t in trades:
        capital += t.get("pnl_dollars", 0)
        entry_time = t.get("entry_time", "")
        date_str = entry_time[:10] if entry_time else ""
        data.append({
            "date": date_str,
            "equity": round(capital, 2),
            "pnl": round(t.get("pnl_dollars", 0), 2),
            "direction": t.get("direction", ""),
            "outcome": t.get("outcome", ""),
        })
    return data


@app.route("/api/admin/seed-trade", methods=["POST"])
def seed_trade():
    """Seed a historical trade into the database (admin only)."""
    data = request.get_json()
    if not _check_admin(data):
        return jsonify({"error": "unauthorized"}), 403

    trade = data.get("trade")
    if not trade:
        return jsonify({"error": "missing trade data"}), 400

    save_trade(trade)
    logger.info(f"Seeded trade: {trade['direction']} @ {trade['entry_price']} | {trade.get('outcome')}")
    return jsonify({"ok": True, "trade_id": trade["id"]})


@app.route("/api/admin/cleanup-positions", methods=["POST"])
def cleanup_positions():
    """Remove invalid positions from DB and memory (admin only)."""
    data = request.get_json()
    if not _check_admin(data):
        return jsonify({"error": "unauthorized"}), 403
    removed = []
    with state_lock:
        for pos_id in list(pos_mgr.open_positions.keys()):
            pos = pos_mgr.open_positions[pos_id]
            if not pos.get("entry_price") or not pos.get("direction"):
                del pos_mgr.open_positions[pos_id]
                delete_position(pos_id)
                removed.append(pos_id)
    return jsonify({"ok": True, "removed": removed})


@app.route("/api/admin/delete-trade", methods=["POST"])
def delete_trade():
    """Delete a trade by ID (admin only)."""
    data = request.get_json()
    if not _check_admin(data):
        return jsonify({"error": "unauthorized"}), 403
    trade_id = data.get("trade_id")
    if not trade_id:
        return jsonify({"error": "missing trade_id"}), 400
    from webapp.models import get_db
    import webapp.models as _models
    with get_db() as conn:
        conn.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
    _models._stats_cache = None  # Invalidate stats cache
    logger.info(f"Deleted trade: {trade_id}")
    return jsonify({"ok": True})


@app.route("/api/admin/reset", methods=["POST"])
def reset_bot():
    """Full reset: clear all trades, positions, state. Fresh start."""
    data = request.get_json()
    if not _check_admin(data):
        return jsonify({"error": "unauthorized"}), 403
    from webapp.models import get_db
    import webapp.models as _models
    with get_db() as conn:
        conn.execute("DELETE FROM trades")
        conn.execute("DELETE FROM positions")
        conn.execute("DELETE FROM bot_state")
        conn.execute("DELETE FROM blocked_signals")
    _models._stats_cache = None  # Invalidate stats cache
    with state_lock:
        pos_mgr.open_positions.clear()
        pnl_tracker.daily_pnl.clear()
        blocked_signals.clear()
    logger.info("FULL RESET: All data cleared")
    return jsonify({"ok": True, "message": "Full reset complete"})


@app.route("/api/admin/seed-position", methods=["POST"])
def seed_position():
    """Seed an active position into the bot (admin only)."""
    data = request.get_json()
    if not _check_admin(data):
        return jsonify({"error": "unauthorized"}), 403

    position = data.get("position")
    if not position:
        return jsonify({"error": "missing position data"}), 400

    # Add to in-memory position manager AND persist to DB
    with state_lock:
        pos_mgr.open_positions[position["id"]] = position
    save_position(position)
    logger.info(f"Seeded position: {position['direction']} @ {position['entry_price']} | ID: {position['id']}")
    return jsonify({"ok": True, "position_id": position["id"]})


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Start the bot and web server."""
    logger.info("=" * 60)
    logger.info("  NQ FUTURES PAPER TRADING BOT v5.0 (Config A)")
    logger.info("  Strategy: BRK+MTF+VOL+VWAP [2,1,5,2] | RR 5.0 | Tiered T1+T2")
    logger.info("  T1: SL<=25, 2c (1TP+1R) trail ATR*0.5, runner SL*3 | T2: SL<=50, 1c")
    logger.info("  Smart DL: $1,100 (realized+unrealized) | Hours: 07-23 ET")
    logger.info("  Trades run to natural exit (SL/TP/timeout) — no force close")
    logger.info("  Data: Databento (CME) primary, YF fallback")
    logger.info("  Multi-position: ENABLED (unlimited concurrent)")
    logger.info("=" * 60)

    # Initialize database
    init_db()
    logger.info("Database initialized")

    # Start strategy loop in background thread
    strategy_thread = threading.Thread(target=strategy_loop, daemon=True)
    strategy_thread.start()
    logger.info("Strategy loop thread started")

    # Start weekly email job in background thread
    email_thread = threading.Thread(target=_weekly_email_job, daemon=True)
    email_thread.start()
    logger.info("Weekly email thread started")

    # Start Flask server
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting web server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
