"""Strategy pipeline runner and position manager for paper trading.

Adapted from forward_test_2026.py — same 3-engine composite scoring,
cooldown state machine, and simulator.py runner logic.
"""

import logging
import sys
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf

# Add project root to path for engine imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.timeutils import add_session_flags
from src.engines.mtf_trend import compute_mtf_trend
from src.engines.volume import compute_volume
from src.engines.key_levels import compute_key_levels
from src.engines.vwap import compute_vwap

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# ── Strategy Parameters (matching V3) ────────────────────────────────────

RR_RATIO = 5.0
MAX_SL = 25
THRESHOLD = 5.0
SMART_DL = 750
POINT_VALUE = 20.0
CONTRACTS = 2  # 1 TP + 1 Runner
EXCLUDE_HOURS = [0, 1, 2, 3, 4, 5, 6]
SELL_WEIGHTS = np.array([3, 2, 3], dtype=float)  # MTF, VOL, BRK
BUY_WEIGHTS = np.array([3, 1, 4], dtype=float)   # MTF, VOL, BRK
MAX_BARS_HELD = 60
RUNNER_MAX_BARS = 120
SL_ATR_MULT = 1.5
TRAIL_ATR_MULT = 1.5


# ═══════════════════════════════════════════════════════════════════════════
# Strategy Runner
# ═══════════════════════════════════════════════════════════════════════════

class StrategyRunner:
    """Fetches data from Yahoo Finance and runs the full strategy pipeline."""

    def __init__(self):
        self.cfg = load_config(PROJECT_ROOT / "config" / "settings.yaml")
        self._last_df: pd.DataFrame | None = None
        self._last_fetch_time: datetime | None = None

    def fetch_and_run_pipeline(self) -> pd.DataFrame | None:
        """Fetch 60d of 5m data from Yahoo Finance and run the full pipeline.

        Returns the processed DataFrame with signals, or None on failure.
        """
        try:
            logger.info("Fetching Yahoo Finance data (NQ=F, 60d, 5m)...")
            ticker = yf.Ticker("NQ=F")
            df = ticker.history(period="60d", interval="5m")

            if df is None or len(df) == 0:
                logger.error("Yahoo Finance returned empty data")
                return None

            # Normalize columns
            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            df = df[["open", "high", "low", "close", "volume"]].copy()

            # Timezone handling
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df.index = df.index.tz_convert("America/New_York")

            # Clean data
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["open", "high", "low", "close"])
            df["volume"] = df["volume"].fillna(0)

            logger.info(f"  Got {len(df):,} bars: {df.index[0]} to {df.index[-1]}")

            # Run pipeline
            ecfg = self.cfg.engines
            bt = self.cfg.backtest

            df = add_session_flags(
                df, tz=self.cfg.timezone,
                premarket_start_hour=ecfg.key_levels.premarket_start_hour,
                premarket_end_hour=ecfg.key_levels.premarket_end_hour,
                rth_start_hour=ecfg.vwap.rth_start_hour,
                rth_start_minute=ecfg.vwap.rth_start_minute,
            )
            df = compute_mtf_trend(df, ecfg.mtf_trend, self.cfg.mtf_timeframes)
            df = compute_volume(df, ecfg.volume)
            df = compute_key_levels(df, ecfg.key_levels)
            df = compute_vwap(df, ecfg.vwap)
            df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=bt.atr_period)

            # 3-engine composite scoring (same as V3 / forward_test_2026.py)
            mtf_arr = df["mtf_score"].fillna(0).values
            vol_arr = df["vol_score"].fillna(0).values
            lvl_arr = df["levels_score"].fillna(0).values
            direction_arr = df["mtf_direction"].fillna("neutral").values
            bull_mask = (direction_arr == "bull")
            bear_mask = (direction_arr == "bear")
            hour_arr = df.index.hour.values

            scores_map = {"MTF": mtf_arr, "VOL": vol_arr, "BRK": lvl_arr}
            engines_list = ["MTF", "VOL", "BRK"]

            sell_score = sum(scores_map[e] * SELL_WEIGHTS[i] for i, e in enumerate(engines_list)) / SELL_WEIGHTS.sum()
            sell_score = np.clip(sell_score, 0, 10)
            buy_score = sum(scores_map[e] * BUY_WEIGHTS[i] for i, e in enumerate(engines_list)) / BUY_WEIGHTS.sum()
            buy_score = np.clip(buy_score, 0, 10)

            buy_elig = (buy_score >= THRESHOLD) & bull_mask
            sell_elig = (sell_score >= THRESHOLD) & bear_mask
            hour_mask = np.ones(len(df), dtype=bool)
            for h in EXCLUDE_HOURS:
                hour_mask &= (hour_arr != h)
            buy_elig = buy_elig & hour_mask
            sell_elig = sell_elig & hour_mask

            sig_buy = _apply_cooldown(buy_elig, buy_score, THRESHOLD)
            sig_sell = _apply_cooldown(sell_elig, sell_score, THRESHOLD)

            df["composite_score"] = sell_score
            bm = sig_buy.astype(bool)
            if bm.any():
                df.loc[df.index[bm], "composite_score"] = buy_score[bm]
            df["signal_buy"] = sig_buy
            df["signal_sell"] = sig_sell

            self._last_df = df
            self._last_fetch_time = datetime.now(ET)
            logger.info(f"  Pipeline complete. Signals: {sig_buy.sum()} buy, {sig_sell.sum()} sell")
            return df

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return None

    def check_last_bar_signal(self, df: pd.DataFrame) -> dict | None:
        """Check if the most recent completed bar has a new signal.

        Returns signal dict or None.
        """
        if df is None or len(df) < 2:
            return None

        # Check the second-to-last bar (last completed bar)
        # The very last bar might still be forming
        idx = -2 if len(df) > 1 else -1
        row = df.iloc[idx]
        bar_time = df.index[idx]

        direction = None
        if row.get("signal_buy", False):
            direction = "buy"
        elif row.get("signal_sell", False):
            direction = "sell"

        if direction is None:
            return None

        atr = row.get("atr", 0)
        if pd.isna(atr) or atr <= 0:
            return None

        entry_price = row["close"]
        sl_distance = max(atr * SL_ATR_MULT, 10.0)

        if sl_distance > MAX_SL:
            logger.info(f"  Signal skipped: SL distance {sl_distance:.1f} > {MAX_SL}")
            return None

        tp_distance = sl_distance * RR_RATIO

        if direction == "buy":
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance

        # Individual engine scores for display
        mtf_s = row.get("mtf_score", 0)
        vol_s = row.get("vol_score", 0)
        brk_s = row.get("levels_score", 0)

        return {
            "direction": direction,
            "score": round(float(row.get("composite_score", 0)), 2),
            "entry_price": round(entry_price, 2),
            "sl_price": round(sl_price, 2),
            "tp_price": round(tp_price, 2),
            "sl_distance": round(sl_distance, 2),
            "atr": round(float(atr), 2),
            "bar_time": str(bar_time),
            "mtf_score": round(float(mtf_s), 1),
            "vol_score": round(float(vol_s), 1),
            "brk_score": round(float(brk_s), 1),
        }

    def get_current_price(self) -> float | None:
        """Get latest price from cached data."""
        if self._last_df is not None and len(self._last_df) > 0:
            return float(self._last_df.iloc[-1]["close"])
        return None

    def get_last_bar(self) -> dict | None:
        """Get the latest bar as a dict."""
        if self._last_df is None or len(self._last_df) == 0:
            return None
        row = self._last_df.iloc[-1]
        return {
            "time": str(self._last_df.index[-1]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }

    def get_recent_candles(self, count: int = 120, tf_minutes: int = 5) -> list[dict]:
        """Get the last N candles for the price chart.

        Args:
            count: Number of candles to return.
            tf_minutes: Timeframe in minutes (5, 15, or 60).
        """
        if self._last_df is None or len(self._last_df) == 0:
            return []
        df = self._last_df
        if tf_minutes > 5:
            df = df.resample(f"{tf_minutes}min").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna()
        df = df.tail(count)
        candles = []
        for ts, row in df.iterrows():
            # Send UTC epoch seconds — the browser converts to local TZ
            epoch = int(ts.timestamp())
            candles.append({
                "time": epoch,
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
            })
        return candles


def _apply_cooldown(eligible, scores, threshold):
    """Cooldown state machine to prevent signal spam."""
    READY, FIRED, COOLING = 0, 1, 2
    n = len(eligible)
    signals = np.zeros(n, dtype=bool)
    state = READY
    for i in range(n):
        if state == READY:
            if eligible[i]:
                state = FIRED
                signals[i] = True
        elif state == FIRED:
            if scores[i] < threshold:
                state = COOLING
        elif state == COOLING:
            if eligible[i]:
                state = FIRED
                signals[i] = True
    return signals


# ═══════════════════════════════════════════════════════════════════════════
# Position Manager
# ═══════════════════════════════════════════════════════════════════════════

class PositionManager:
    """Manages virtual positions with runner mode.

    Mirrors simulator.py logic exactly:
    - Phase 1 (active): Check SL/TP each bar, timeout at max_bars_held
    - Phase 2 (runner): After TP1 hit, trail with ATR*1.5, timeout at runner_max_bars
    - Conservative: if both SL and TP could trigger on same bar, SL wins
    """

    def __init__(self):
        self.open_pos: dict | None = None
        self.starting_capital = 100_000.0

    def has_open_position(self) -> bool:
        return self.open_pos is not None

    def open_position(self, signal: dict, account: float = None) -> dict:
        """Create a new virtual position from a signal."""
        pos = {
            "id": str(uuid.uuid4())[:8],
            "direction": signal["direction"],
            "entry_price": signal["entry_price"],
            "entry_time": signal["bar_time"],
            "sl_price": signal["sl_price"],
            "tp_price": signal["tp_price"],
            "sl_distance": signal["sl_distance"],
            "score": signal["score"],
            "atr_at_entry": signal["atr"],
            "contracts": CONTRACTS,

            # Phase tracking
            "phase": "active",
            "tp1_hit": False,
            "tp1_exit_price": None,
            "tp1_exit_time": None,
            "tp1_pnl": 0.0,

            # Runner tracking
            "runner_sl": None,
            "runner_extreme": None,
            "runner_trail_dist": None,
            "runner_exit_price": None,
            "runner_exit_time": None,
            "runner_pnl": 0.0,
            "runner_outcome": None,

            # Timing
            "bars_held": 0,
            "runner_bars": 0,

            # P&L
            "total_pnl": 0.0,
            "outcome": "",
            "exit_price": None,
            "exit_time": None,
            "account_before": account or self.starting_capital,

            # Score details
            "mtf_score": signal.get("mtf_score", 0),
            "vol_score": signal.get("vol_score", 0),
            "brk_score": signal.get("brk_score", 0),
        }
        self.open_pos = pos
        logger.info(
            f"POSITION OPENED: {signal['direction'].upper()} @ {signal['entry_price']:.2f} "
            f"| SL: {signal['sl_price']:.2f} | TP: {signal['tp_price']:.2f} "
            f"| Score: {signal['score']}"
        )
        return pos

    def update_on_bar(self, bar: dict) -> str | None:
        """Update position with new bar data.

        Args:
            bar: dict with keys: high, low, close

        Returns:
            Event string ('SL', 'TP1_HIT', 'RUNNER_TRAIL', 'RUNNER_BE',
            'RUNNER_TIMEOUT', 'TIMEOUT', 'CLOSED') or None
        """
        pos = self.open_pos
        if pos is None:
            return None

        bar_high = bar["high"]
        bar_low = bar["low"]
        bar_close = bar["close"]
        bar_time = bar.get("time", str(datetime.now(ET)))

        if pos["phase"] == "active":
            return self._update_active_phase(pos, bar_high, bar_low, bar_close, bar_time)
        elif pos["phase"] == "runner":
            return self._update_runner_phase(pos, bar_high, bar_low, bar_close, bar_time)
        return None

    def _update_active_phase(self, pos, bar_high, bar_low, bar_close, bar_time) -> str | None:
        """Phase 1: Check SL/TP, manage timeout."""
        pos["bars_held"] += 1
        direction = pos["direction"]
        entry = pos["entry_price"]
        sl = pos["sl_price"]
        tp = pos["tp_price"]

        # Check SL/TP
        if direction == "buy":
            sl_hit = bar_low <= sl
            tp_hit = bar_high >= tp
        else:
            sl_hit = bar_high >= sl
            tp_hit = bar_low <= tp

        # Conservative: SL wins if both hit
        if sl_hit:
            # Both contracts stopped out
            pnl_pts = (sl - entry) if direction == "buy" else (entry - sl)
            pnl_dollars = pnl_pts * POINT_VALUE * CONTRACTS
            self._close_position(pos, sl, bar_time, "SL", pnl_dollars, pnl_pts)
            return "CLOSED"

        if tp_hit:
            # TP1 hit — close contract 1, start runner on contract 2
            if direction == "buy":
                c1_pnl_pts = tp - entry
            else:
                c1_pnl_pts = entry - tp
            c1_pnl = c1_pnl_pts * POINT_VALUE * 1  # 1 contract at TP

            pos["tp1_hit"] = True
            pos["tp1_exit_price"] = tp
            pos["tp1_exit_time"] = bar_time
            pos["tp1_pnl"] = c1_pnl

            # Start runner phase
            pos["phase"] = "runner"
            pos["runner_sl"] = entry  # Breakeven
            pos["runner_extreme"] = tp  # Best price = TP level
            pos["runner_trail_dist"] = pos["atr_at_entry"] * TRAIL_ATR_MULT
            pos["runner_bars"] = 0

            logger.info(
                f"TP1 HIT! +{c1_pnl_pts:.2f} pts (${c1_pnl:+,.0f}) | "
                f"Runner active, trail dist: {pos['runner_trail_dist']:.2f}"
            )
            return "TP1_HIT"

        # Check timeout
        if pos["bars_held"] >= MAX_BARS_HELD:
            pnl_pts = (bar_close - entry) if direction == "buy" else (entry - bar_close)
            pnl_dollars = pnl_pts * POINT_VALUE * CONTRACTS
            self._close_position(pos, bar_close, bar_time, "timeout", pnl_dollars, pnl_pts)
            return "CLOSED"

        return None

    def _update_runner_phase(self, pos, bar_high, bar_low, bar_close, bar_time) -> str | None:
        """Phase 2: Runner with trailing stop."""
        pos["runner_bars"] += 1
        pos["bars_held"] += 1
        direction = pos["direction"]
        entry = pos["entry_price"]
        trail_dist = pos["runner_trail_dist"]

        if direction == "buy":
            # Update extreme high
            if bar_high > pos["runner_extreme"]:
                pos["runner_extreme"] = bar_high
                pos["runner_sl"] = max(entry, pos["runner_extreme"] - trail_dist)

            # Check trailing stop
            if bar_low <= pos["runner_sl"]:
                runner_exit = pos["runner_sl"]
                runner_outcome = "trail" if runner_exit > entry else "BE"
                return self._close_runner(pos, runner_exit, bar_time, runner_outcome)

        else:  # sell
            # Update extreme low
            if bar_low < pos["runner_extreme"]:
                pos["runner_extreme"] = bar_low
                pos["runner_sl"] = min(entry, pos["runner_extreme"] + trail_dist)

            # Check trailing stop
            if bar_high >= pos["runner_sl"]:
                runner_exit = pos["runner_sl"]
                runner_outcome = "trail" if runner_exit < entry else "BE"
                return self._close_runner(pos, runner_exit, bar_time, runner_outcome)

        # Check runner timeout
        if pos["runner_bars"] >= RUNNER_MAX_BARS:
            return self._close_runner(pos, bar_close, bar_time, "timeout")

        return None

    def _close_runner(self, pos, runner_exit, bar_time, runner_outcome) -> str:
        """Close the runner contract and finalize the trade."""
        direction = pos["direction"]
        entry = pos["entry_price"]

        if direction == "buy":
            c2_pnl_pts = runner_exit - entry
        else:
            c2_pnl_pts = entry - runner_exit

        c2_pnl = c2_pnl_pts * POINT_VALUE * 1  # 1 runner contract

        pos["runner_exit_price"] = runner_exit
        pos["runner_exit_time"] = bar_time
        pos["runner_pnl"] = c2_pnl
        pos["runner_outcome"] = runner_outcome

        total_pnl = pos["tp1_pnl"] + c2_pnl
        outcome = f"TP+{runner_outcome}"

        self._close_position(pos, runner_exit, bar_time, outcome, total_pnl)

        logger.info(
            f"RUNNER CLOSED: {runner_outcome.upper()} @ {runner_exit:.2f} | "
            f"Runner PnL: ${c2_pnl:+,.0f} | Total: ${total_pnl:+,.0f}"
        )
        return "CLOSED"

    def _close_position(self, pos, exit_price, exit_time, outcome, total_pnl, pnl_pts=None):
        """Finalize and close the position."""
        pos["exit_price"] = exit_price
        pos["exit_time"] = exit_time
        pos["outcome"] = outcome
        pos["total_pnl"] = total_pnl
        if pnl_pts is not None:
            pos["pnl_points"] = pnl_pts
        pos["account_after"] = pos["account_before"] + total_pnl
        pos["phase"] = "closed"
        self.open_pos = None

        logger.info(
            f"TRADE CLOSED: {outcome} | {pos['direction'].upper()} "
            f"@ {pos['entry_price']:.2f} → {exit_price:.2f} | "
            f"P&L: ${total_pnl:+,.0f} | Bars: {pos['bars_held']}"
        )

    def get_unrealized_pnl(self, current_price: float) -> float | None:
        """Calculate unrealized P&L for the open position."""
        if self.open_pos is None:
            return None
        pos = self.open_pos
        entry = pos["entry_price"]
        direction = pos["direction"]

        if pos["phase"] == "active":
            pts = (current_price - entry) if direction == "buy" else (entry - current_price)
            return pts * POINT_VALUE * CONTRACTS
        elif pos["phase"] == "runner":
            pts = (current_price - entry) if direction == "buy" else (entry - current_price)
            return pos["tp1_pnl"] + (pts * POINT_VALUE * 1)  # TP1 realized + runner unrealized
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Daily P&L Tracker
# ═══════════════════════════════════════════════════════════════════════════

class DailyPnLTracker:
    """Enforces Smart Daily Loss limit ($750)."""

    def __init__(self, daily_loss_limit: float = SMART_DL):
        self.daily_loss_limit = daily_loss_limit
        self.daily_pnl: dict[str, float] = {}

    def get_today_key(self) -> str:
        return datetime.now(ET).strftime("%Y-%m-%d")

    def record_pnl(self, pnl: float):
        key = self.get_today_key()
        self.daily_pnl[key] = self.daily_pnl.get(key, 0) + pnl

    def can_take_trade(self, sl_distance: float) -> bool:
        """Check if a new trade is allowed under Smart Daily Loss."""
        key = self.get_today_key()
        current = self.daily_pnl.get(key, 0)

        # Already at limit
        if current <= -self.daily_loss_limit:
            return False

        # Check if potential loss would exceed budget
        potential_loss = sl_distance * POINT_VALUE * CONTRACTS
        remaining = self.daily_loss_limit + current
        if potential_loss > remaining:
            return False

        return True

    def get_today_pnl(self) -> float:
        return self.daily_pnl.get(self.get_today_key(), 0)

    def get_today_status(self) -> str:
        pnl = self.get_today_pnl()
        if pnl <= -self.daily_loss_limit:
            return "LOCKED"
        elif pnl <= -(self.daily_loss_limit - 200):
            return "WARNING"
        return "OK"

    def get_budget_remaining(self) -> float:
        return self.daily_loss_limit + self.get_today_pnl()

    def to_dict(self) -> dict:
        return {"daily_pnl": self.daily_pnl, "limit": self.daily_loss_limit}

    def from_dict(self, data: dict):
        self.daily_pnl = data.get("daily_pnl", {})
        self.daily_loss_limit = data.get("limit", SMART_DL)
