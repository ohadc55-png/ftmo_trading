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
    get_all_stats,
)
from webapp.strategy_runner import (
    StrategyRunner, PositionManager, DailyPnLTracker,
    POINT_VALUE, TIER1_MAX_SL, TIER2_MAX_SL, SMART_DL,
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


# ═══════════════════════════════════════════════════════════════════════════
# Background Strategy Loop
# ═══════════════════════════════════════════════════════════════════════════

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

            if not is_market_hours():
                logger.debug("Market closed, sleeping...")
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
            pnl_tracker.record_pnl(closed_pos["total_pnl"])
            tier = closed_pos.get("tier", 1)
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
                "tier": tier,
            })
            delete_position(closed_pos["id"])
            logger.info(f"Trade saved to DB: {closed_pos['outcome']} ${closed_pos['total_pnl']:+,.0f}")

    # 4. Save all open position states
    for pos in pos_mgr.get_open_positions():
        save_position(pos)

    # 5. Check for new signal (always, regardless of open positions)
    last_bar_time = get_state("last_signal_bar_time")
    signal = runner.check_last_bar_signal(df)

    if signal and signal["bar_time"] != last_bar_time:
        # Reset blocked list on new day
        today_str = datetime.now(ET).strftime("%Y-%m-%d")
        blocked_signals[:] = [b for b in blocked_signals if b.get("date") == today_str]

        # Signal blocked by SL cap / tier rules
        if signal.get("blocked"):
            signal["date"] = today_str
            blocked_signals.append(signal)
            save_state("last_signal_bar_time", signal["bar_time"])
            logger.info(f"Signal BLOCKED ({signal['blocked']}): {signal['direction'].upper()} @ {signal['entry_price']:.2f}")
        # Smart DL: block new trades if realized daily P&L <= -$1,100
        # Open positions always run to completion. If P&L recovers, trading resumes.
        elif not pnl_tracker.can_take_trade():
            signal["blocked"] = f"SmartDL (daily realized ${pnl_tracker.get_today_pnl():+,.0f})"
            signal["date"] = today_str
            blocked_signals.append(signal)
            save_state("last_signal_bar_time", signal["bar_time"])
            logger.info(f"Signal BLOCKED: Smart DL (daily realized: ${pnl_tracker.get_today_pnl():+,.0f})")
        else:
            # Get current account
            stats = get_all_stats()
            account = 100_000 + stats["total_pnl"]

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

    logger.info(
        f"Cycle #{bot_status['cycles_count']} | NQ {price_str} | {pos_str} | "
        f"Daily: ${pnl_tracker.get_today_pnl():+,.0f} | "
        f"Status: {pnl_tracker.get_today_status()}"
    )


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
    positions = pos_mgr.get_open_positions()
    position = positions[0] if positions else None  # backward compat for template
    current_price = runner.get_current_price()
    unrealized = pos_mgr.get_unrealized_pnl(current_price) if current_price and positions else None

    # Equity curve data
    equity_data = _build_equity_data(stats.get("trades", []))
    candle_data = runner.get_recent_candles(120)

    # Filter blocked signals for today only
    today_str = datetime.now(ET).strftime("%Y-%m-%d")
    today_blocked = [b for b in blocked_signals if b.get("date") == today_str]

    return render_template("dashboard.html",
        stats=stats,
        position=position,
        positions=positions,
        current_price=current_price,
        unrealized=unrealized,
        today_trades=today_trades,
        daily_pnl=pnl_tracker.get_today_pnl(),
        daily_status=pnl_tracker.get_today_status(),
        budget_remaining=pnl_tracker.get_budget_remaining(),
        bot_status=bot_status,
        is_market_open=is_market_hours(),
        equity_data=json.dumps(equity_data),
        candle_data=json.dumps(candle_data),
        blocked_signals=json.dumps(today_blocked),
        now=datetime.now(ET),
    )


@app.route("/api/status")
def api_status():
    """JSON endpoint for AJAX auto-refresh."""
    # Refresh data if stale (>10 min)
    if runner._last_fetch_time:
        age = (datetime.now(ET) - runner._last_fetch_time).total_seconds()
        if age > 600:
            runner.fetch_and_run_pipeline()
    current_price = runner.get_current_price()
    positions = pos_mgr.get_open_positions()
    unrealized = pos_mgr.get_total_unrealized_pnl(current_price) if current_price and positions else None

    # Per-position unrealized
    positions_with_pnl = []
    for pos in positions:
        pos_copy = dict(pos)
        if current_price:
            pos_copy["unrealized"] = pos_mgr.get_position_unrealized_pnl(pos, current_price)
        positions_with_pnl.append(pos_copy)

    stats = get_all_stats()

    today_str = datetime.now(ET).strftime("%Y-%m-%d")
    today_blocked = [b for b in blocked_signals if b.get("date") == today_str]

    return jsonify({
        "price": current_price,
        "position": positions_with_pnl[0] if positions_with_pnl else None,  # backward compat
        "positions": positions_with_pnl,
        "positions_count": len(positions),
        "unrealized": unrealized,
        "worst_case_exposure": pos_mgr.get_worst_case_loss(),
        "daily_pnl": pnl_tracker.get_today_pnl(),
        "daily_status": pnl_tracker.get_today_status(),
        "budget_remaining": pnl_tracker.get_budget_remaining(),
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
        "bot": bot_status,
        "is_market_open": is_market_hours(),
        "blocked_signals": today_blocked,
        "timestamp": datetime.now(ET).isoformat(),
    })


@app.route("/api/candles")
def api_candles():
    """Return recent candle data for the price chart."""
    # Refresh data if stale (>10 min) so chart stays alive even outside strategy loop
    if runner._last_fetch_time:
        age = (datetime.now(ET) - runner._last_fetch_time).total_seconds()
        if age > 600:
            runner.fetch_and_run_pipeline()
    elif runner._last_df is None:
        runner.fetch_and_run_pipeline()

    count = request.args.get("count", 120, type=int)
    tf = request.args.get("tf", 5, type=int)
    if tf not in (5, 15, 60):
        tf = 5
    candles = runner.get_recent_candles(min(count, 500), tf_minutes=tf)
    return jsonify(candles)


@app.route("/trades")
def trades_page():
    """Full trade history."""
    trades = get_trades()
    stats = get_all_stats()
    return render_template("dashboard.html",
        stats=stats,
        position=None,
        current_price=runner.get_current_price(),
        unrealized=None,
        today_trades=[],
        daily_pnl=pnl_tracker.get_today_pnl(),
        daily_status=pnl_tracker.get_today_status(),
        budget_remaining=pnl_tracker.get_budget_remaining(),
        bot_status=bot_status,
        is_market_open=is_market_hours(),
        equity_data=json.dumps(_build_equity_data(stats.get("trades", []))),
        candle_data=json.dumps(runner.get_recent_candles(120)),
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
    capital = 100_000
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
    if not data or "secret" not in data or data["secret"] != "nq_seed_2026":
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
    if not data or data.get("secret") != "nq_seed_2026":
        return jsonify({"error": "unauthorized"}), 403
    removed = []
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
    if not data or data.get("secret") != "nq_seed_2026":
        return jsonify({"error": "unauthorized"}), 403
    trade_id = data.get("trade_id")
    if not trade_id:
        return jsonify({"error": "missing trade_id"}), 400
    from webapp.models import get_db
    with get_db() as conn:
        conn.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
    logger.info(f"Deleted trade: {trade_id}")
    return jsonify({"ok": True})


@app.route("/api/admin/seed-position", methods=["POST"])
def seed_position():
    """Seed an active position into the bot (admin only)."""
    data = request.get_json()
    if not data or "secret" not in data or data["secret"] != "nq_seed_2026":
        return jsonify({"error": "unauthorized"}), 403

    position = data.get("position")
    if not position:
        return jsonify({"error": "missing position data"}), 400

    # Add to in-memory position manager AND persist to DB
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
    logger.info("  NQ FUTURES PAPER TRADING BOT v3.0 (Multi-Position)")
    logger.info("  Strategy: BRK+MTF+VOL | RR 5.0 | Tiered T1+T2")
    logger.info("  T1: SL<=25, 2c (1TP+1R) | T2: SL<=50, 1c (no runner)")
    logger.info("  Smart DL: $1,100 (realized P&L only, recoverable) | Hours: 07-23 ET")
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
