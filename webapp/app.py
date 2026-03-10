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
    save_position, get_position, save_state, get_state,
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

    # 3. If open position, check SL/TP/runner
    if pos_mgr.has_open_position() and last_bar:
        event = pos_mgr.update_on_bar(last_bar)
        if event == "CLOSED":
            pos = pos_mgr.open_pos  # None now, get from last closed
            # The position was stored in open_pos before close, get the closed data
            # Actually, after close, open_pos is None. We need to retrieve the closed position.
            # Let's get it from the position we had
            pass

        # Check if position just closed
        position = get_position()
        if position and position.get("phase") == "closed":
            # Record the trade
            pnl_tracker.record_pnl(position["total_pnl"])
            tier = position.get("tier", 1)
            save_trade({
                "id": position["id"],
                "direction": position["direction"],
                "entry_price": position["entry_price"],
                "exit_price": position.get("exit_price"),
                "sl_price": position["sl_price"],
                "tp_price": position["tp_price"],
                "sl_distance": position["sl_distance"],
                "entry_time": position["entry_time"],
                "exit_time": position.get("exit_time"),
                "score": position["score"],
                "outcome": position.get("outcome", ""),
                "pnl_dollars": position["total_pnl"],
                "pnl_points": position.get("pnl_points", 0),
                "bars_held": position.get("bars_held", 0),
                "contracts": position.get("contracts", 2),
                "runner_exit_price": position.get("runner_exit_price"),
                "runner_pnl": position.get("runner_pnl"),
                "runner_outcome": position.get("runner_outcome"),
                "account_after": position.get("account_after"),
                "tier": tier,
            })
            save_position(None)
            logger.info(f"Trade saved to DB: {position['outcome']} ${position['total_pnl']:+,.0f}")

    # 4. Save current position state
    if pos_mgr.has_open_position():
        save_position(pos_mgr.open_pos)

    # 5. If flat, check for new signal
    if not pos_mgr.has_open_position():
        last_bar_time = get_state("last_signal_bar_time")
        signal = runner.check_last_bar_signal(df)

        if signal and signal["bar_time"] != last_bar_time:
            # Check Smart DL (pass tier-specific contracts)
            if not pnl_tracker.can_take_trade(signal["sl_distance"], signal.get("contracts", 2)):
                logger.info(f"Signal SKIPPED: Smart DL limit (daily: ${pnl_tracker.get_today_pnl():+,.0f})")
            else:
                # Get current account
                stats = get_all_stats()
                account = 100_000 + stats["total_pnl"]

                pos = pos_mgr.open_position(signal, account=account)
                save_position(pos)
                save_state("last_signal_bar_time", signal["bar_time"])
                logger.info(f"Signal TAKEN: {signal['direction'].upper()} @ {signal['entry_price']:.2f}")

    # 6. Update bot status
    bot_status["last_cycle"] = datetime.now(ET).isoformat()
    bot_status["cycles_count"] += 1
    bot_status["last_error"] = None

    # Save state
    save_state("pnl_tracker", pnl_tracker.to_dict())

    price_str = f"{current_price:,.2f}" if current_price else "N/A"
    pos_str = "FLAT"
    if pos_mgr.has_open_position():
        p = pos_mgr.open_pos
        unr = pos_mgr.get_unrealized_pnl(current_price) if current_price else 0
        pos_str = f"{p['direction'].upper()} @ {p['entry_price']:,.2f} ({p['phase']}) unrealized: ${unr:+,.0f}"

    logger.info(
        f"Cycle #{bot_status['cycles_count']} | NQ {price_str} | {pos_str} | "
        f"Daily: ${pnl_tracker.get_today_pnl():+,.0f} | "
        f"Status: {pnl_tracker.get_today_status()}"
    )


def _restore_state():
    """Restore bot state from DB on startup."""
    # Restore position
    pos_data = get_position()
    if pos_data and pos_data.get("phase") != "closed":
        pos_mgr.open_pos = pos_data
        logger.info(f"Restored open position: {pos_data['direction']} @ {pos_data['entry_price']}")

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
    position = pos_mgr.open_pos
    current_price = runner.get_current_price()
    unrealized = pos_mgr.get_unrealized_pnl(current_price) if current_price and position else None

    # Equity curve data
    equity_data = _build_equity_data(stats.get("trades", []))
    candle_data = runner.get_recent_candles(120)

    return render_template("dashboard.html",
        stats=stats,
        position=position,
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
    position = pos_mgr.open_pos
    unrealized = pos_mgr.get_unrealized_pnl(current_price) if current_price and position else None
    stats = get_all_stats()

    return jsonify({
        "price": current_price,
        "position": position,
        "unrealized": unrealized,
        "daily_pnl": pnl_tracker.get_today_pnl(),
        "daily_status": pnl_tracker.get_today_status(),
        "budget_remaining": pnl_tracker.get_budget_remaining(),
        "total_pnl": stats["total_pnl"],
        "total_trades": stats["total_trades"],
        "win_rate": stats["win_rate"],
        "profit_factor": stats["profit_factor"],
        "bot": bot_status,
        "is_market_open": is_market_hours(),
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


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Start the bot and web server."""
    logger.info("=" * 60)
    logger.info("  NQ FUTURES PAPER TRADING BOT v2.0")
    logger.info("  Strategy: BRK+MTF+VOL | RR 5.0 | Tiered T1+T2")
    logger.info("  T1: SL<=25, 2c (1TP+1R) | T2: SL<=50, 1c (no runner)")
    logger.info("  Smart DL: $1,100 | Hours: 07-23 ET")
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
