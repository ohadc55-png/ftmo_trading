"""Strategy pipeline runner and position manager for paper trading.

Adapted from forward_test_2026.py — same 3-engine composite scoring,
cooldown state machine, and simulator.py runner logic.

Data source: Databento (CME/GLBX) for accurate tick-level OHLCV bars.
Fallback: Yahoo Finance if Databento is unavailable.
"""

import logging
import os
import sys
import uuid
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore", category=DeprecationWarning)

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

# ── Strategy Parameters (Tiered T1+T2) ──────────────────────────────────

RR_RATIO = 5.0
TIER1_MAX_SL = 25       # Tier 1: 2 contracts (1 TP + 1 Runner)
TIER2_MAX_SL = 50       # Tier 2: 1 contract (no runner)
TIER2_MAX_LOSS = 1000.0 # Max potential loss per Tier 2 trade ($)
THRESHOLD = 5.0
SMART_DL = 1100
POINT_VALUE = 20.0
STARTING_CAPITAL = 100_000
EXCLUDE_HOURS = [0, 1, 2, 3, 4, 5, 6]
SELL_WEIGHTS = np.array([2, 1, 5, 2], dtype=float)  # MTF, VOL, BRK, VWAP
BUY_WEIGHTS = np.array([2, 1, 5, 2], dtype=float)   # MTF, VOL, BRK, VWAP
MAX_BARS_HELD = 60
RUNNER_MAX_BARS = 120
SL_ATR_MULT = 1.5
TRAIL_ATR_MULT = 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Strategy Runner
# ═══════════════════════════════════════════════════════════════════════════

class StrategyRunner:
    """Fetches data and runs the full strategy pipeline.

    Data source priority:
    1. Databento (CME/GLBX) — accurate exchange data, if API key is set
    2. Yahoo Finance — fallback
    """

    def __init__(self):
        self.cfg = load_config(PROJECT_ROOT / "config" / "settings.yaml")
        self._last_df: pd.DataFrame | None = None       # 5m pipeline data
        self._last_df_1m: pd.DataFrame | None = None     # 1m raw candles
        self._last_fetch_time: datetime | None = None
        self._databento_key = os.environ.get("DATABENTO_API_KEY")
        self._data_source = "databento" if self._databento_key else "yfinance"

    def _fetch_databento(self) -> pd.DataFrame | None:
        """Fetch recent 1m bars from Databento and resample to 5m.

        Databento free plan has a ~1 day delay. We fetch up to yesterday
        from Databento (accurate CME data) and the most recent data from YF.

        COST OPTIMIZATION: Databento historical data is cached for the day.
        Since it only covers up to yesterday, there's no reason to re-fetch
        every 5 minutes. Only the YF gap-fill is refreshed each cycle.
        """
        try:
            import databento as db

            now = datetime.now(ET)
            # End = yesterday to avoid subscription-only data
            end_dt = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0)
            start_dt = end_dt - timedelta(days=5)

            # Cache Databento data — only fetch once per day
            cache_key = end_dt.strftime("%Y-%m-%d")
            if (hasattr(self, "_db_cache_key")
                    and self._db_cache_key == cache_key
                    and self._db_cache_df is not None):
                df_db = self._db_cache_df.copy()
                logger.info(f"Using cached Databento data ({len(df_db):,} 1m bars)")
            else:
                logger.info(f"Fetching Databento data (NQ, {start_dt.strftime('%m/%d')}-{end_dt.strftime('%m/%d')}, 1m)...")
                client = db.Historical(key=self._databento_key)

                data = client.timeseries.get_range(
                    dataset="GLBX.MDP3",
                    symbols=["NQ.v.0"],
                    schema="ohlcv-1m",
                    start=start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                    end=end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                    stype_in="continuous",
                )

                df_db = data.to_df()
                if df_db.empty:
                    logger.warning("Databento returned empty data")
                    return None

                df_db.index = df_db.index.tz_convert("America/New_York")
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df_db.columns:
                        df_db[col] = pd.to_numeric(df_db[col], errors="coerce")

                if df_db["close"].iloc[-1] > 100000:
                    for col in ["open", "high", "low", "close"]:
                        df_db[col] = df_db[col] / 1e9

                df_db = df_db[["open", "high", "low", "close", "volume"]].copy()
                df_db = df_db.dropna(subset=["open", "high", "low", "close"])
                df_db["volume"] = df_db["volume"].fillna(0)

                # Cache for the rest of the day
                self._db_cache_key = cache_key
                self._db_cache_df = df_db.copy()
                logger.info(f"  Databento: {len(df_db):,} 1m bars (cached)")

            # Fetch latest day from YF to fill the gap
            try:
                ticker = yf.Ticker("NQ=F")
                yf_df = ticker.history(period="5d", interval="1m")
                if yf_df is not None and len(yf_df) > 0:
                    yf_df = yf_df.rename(columns={
                        "Open": "open", "High": "high", "Low": "low",
                        "Close": "close", "Volume": "volume",
                    })
                    yf_df = yf_df[["open", "high", "low", "close", "volume"]].copy()
                    if yf_df.index.tz is None:
                        yf_df.index = yf_df.index.tz_localize("UTC")
                    yf_df.index = yf_df.index.tz_convert("America/New_York")
                    for col in ["open", "high", "low", "close", "volume"]:
                        yf_df[col] = pd.to_numeric(yf_df[col], errors="coerce")
                    yf_df = yf_df.dropna(subset=["open", "high", "low", "close"])
                    yf_df["volume"] = yf_df["volume"].fillna(0)

                    # Only use YF data that's newer than Databento
                    yf_new = yf_df[yf_df.index > df_db.index.max()]
                    if len(yf_new) > 0:
                        df_db = pd.concat([df_db, yf_new])
                        df_db = df_db[~df_db.index.duplicated(keep="last")].sort_index()
                        logger.info(f"  + {len(yf_new):,} YF 1m bars (gap fill)")
            except Exception as e:
                logger.warning(f"YF gap fill failed: {e}")

            df_db = _detect_and_adjust_rollover_gaps(df_db)
            self._last_df_1m = df_db.copy()
            logger.info(f"  Total: {len(df_db):,} 1m bars, {df_db.index[0]} to {df_db.index[-1]}")

            # Resample to 5m for pipeline
            df_5m = df_db.resample("5min").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna(subset=["open", "high", "low", "close"])

            logger.info(f"  Resampled to {len(df_5m):,} 5m bars")
            return df_5m

        except Exception as e:
            logger.warning(f"Databento fetch failed: {e}")
            return None

    def _fetch_yfinance(self) -> pd.DataFrame | None:
        """Fetch 5m bars from Yahoo Finance (fallback)."""
        try:
            logger.info("Fetching Yahoo Finance data (NQ=F, 60d, 5m)...")
            ticker = yf.Ticker("NQ=F")
            df = ticker.history(period="60d", interval="5m")

            if df is None or len(df) == 0:
                logger.error("Yahoo Finance returned empty data")
                return None

            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            df = df[["open", "high", "low", "close", "volume"]].copy()

            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df.index = df.index.tz_convert("America/New_York")

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["open", "high", "low", "close"])
            df["volume"] = df["volume"].fillna(0)

            logger.info(f"  YF: {len(df):,} bars, {df.index[0]} to {df.index[-1]}")

            # Also store as 1m-equivalent for chart (YF 5m bars displayed as-is)
            df = _detect_and_adjust_rollover_gaps(df)
            self._last_df_1m = df.copy()
            return df

        except Exception as e:
            logger.error(f"Yahoo Finance fetch failed: {e}", exc_info=True)
            return None

    def fetch_and_run_pipeline(self) -> pd.DataFrame | None:
        """Fetch data and run the full strategy pipeline.

        Returns the processed 5m DataFrame with signals, or None on failure.
        """
        try:
            # Try Databento first, fall back to YF
            df = None
            if self._databento_key:
                df = self._fetch_databento()
                if df is not None:
                    self._data_source = "databento"

            if df is None:
                df = self._fetch_yfinance()
                if df is not None:
                    self._data_source = "yfinance"

            if df is None:
                return None

            # Run pipeline on 5m data
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

            # 4-engine composite scoring (MTF + VOL + BRK + VWAP)
            mtf_arr = df["mtf_score"].fillna(0).values
            vol_arr = df["vol_score"].fillna(0).values
            lvl_arr = df["levels_score"].fillna(0).values
            vwap_arr = df["vwap_score"].fillna(0).values
            direction_arr = df["mtf_direction"].fillna("neutral").values
            bull_mask = (direction_arr == "bull")
            bear_mask = (direction_arr == "bear")
            hour_arr = df.index.hour.values

            scores_map = {"MTF": mtf_arr, "VOL": vol_arr, "BRK": lvl_arr, "VWAP": vwap_arr}
            engines_list = ["MTF", "VOL", "BRK", "VWAP"]

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
            logger.info(
                f"  Pipeline complete [{self._data_source}]. "
                f"Signals: {sig_buy.sum()} buy, {sig_sell.sum()} sell"
            )
            return df

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return None

    def check_last_bar_signal(self, df: pd.DataFrame) -> dict | None:
        """Check if the most recent completed bar has a new signal.

        Returns signal dict or None.  If the signal exists but was blocked
        by SL-cap / tier rules, a dict is still returned with an extra
        ``blocked`` key describing the reason (so the UI can show a gray marker).
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

        # Common fields used by both accepted and blocked signals
        mtf_s = row.get("mtf_score", 0)
        vol_s = row.get("vol_score", 0)
        brk_s = row.get("levels_score", 0)

        def _make_signal(tier, contracts, blocked=None):
            tp_distance = sl_distance * RR_RATIO
            if direction == "buy":
                sl_price = entry_price - sl_distance
                tp_price = entry_price + tp_distance
            else:
                sl_price = entry_price + sl_distance
                tp_price = entry_price - tp_distance
            sig = {
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
                "tier": tier,
                "contracts": contracts,
            }
            if blocked:
                sig["blocked"] = blocked
            return sig

        # Determine tier based on SL distance
        if sl_distance <= TIER1_MAX_SL:
            tier = 1
            contracts = 2  # 1 TP + 1 Runner
        elif sl_distance <= TIER2_MAX_SL:
            potential_loss = sl_distance * POINT_VALUE * 1  # 1 contract
            if potential_loss > TIER2_MAX_LOSS:
                reason = f"T2 risk ${potential_loss:.0f} > ${TIER2_MAX_LOSS:.0f}"
                logger.info(f"  Signal BLOCKED: {reason}")
                return _make_signal(2, 1, blocked=reason)
            tier = 2
            contracts = 1  # No runner
        else:
            reason = f"SL {sl_distance:.1f} > {TIER2_MAX_SL} cap"
            logger.info(f"  Signal BLOCKED: {reason}")
            return _make_signal(0, 0, blocked=reason)

        return _make_signal(tier, contracts)

    def get_current_price(self) -> float | None:
        """Get latest price from cached data."""
        if self._last_df is not None and len(self._last_df) > 0:
            return float(self._last_df.iloc[-1]["close"])
        return None

    def get_live_price(self) -> float | None:
        """Fast price fetch with 10-second cache. Used for real-time P&L."""
        now = datetime.now(ET)
        if (hasattr(self, "_live_price_cache")
                and self._live_price_cache is not None
                and hasattr(self, "_live_price_time")
                and (now - self._live_price_time).total_seconds() < 30):
            return self._live_price_cache

        try:
            ticker = yf.Ticker("NQ=F")
            price = ticker.fast_info.get("lastPrice")
            if price and price > 0:
                self._live_price_cache = float(price)
                self._live_price_time = now
                return self._live_price_cache
        except Exception:
            pass

        # Fallback to cached data
        return self.get_current_price()

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

    def get_recent_candles(self, count: int = 120, tf_minutes: int = 1) -> list[dict]:
        """Get the last N candles for the price chart.

        Args:
            count: Number of candles to return.
            tf_minutes: Timeframe in minutes (1, 5, 15, or 60).
        """
        # For 1m candles, use the raw 1m data if available
        if tf_minutes == 1 and self._last_df_1m is not None and len(self._last_df_1m) > 0:
            df = self._last_df_1m.tail(count)
        elif self._last_df is not None and len(self._last_df) > 0:
            df = self._last_df
            if tf_minutes > 5:
                df = df.resample(f"{tf_minutes}min").agg({
                    "open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum",
                }).dropna()
            df = df.tail(count)
        else:
            return []

        candles = []
        for ts, row in df.iterrows():
            epoch = int(ts.timestamp())
            candles.append({
                "time": epoch,
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
                "volume": int(row["volume"]) if "volume" in row.index else 0,
            })
        return candles



def _is_rollover_window(dt) -> bool:
    """Check if date falls within a quarterly futures rollover window.

    NQ rolls on the 2nd Thursday before the 3rd Friday of Mar/Jun/Sep/Dec.
    We use a wide window (day 8-22) to be safe.
    """
    return dt.month in (3, 6, 9, 12) and 8 <= dt.day <= 22


def _detect_and_adjust_rollover_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and back-adjust rollover price gaps in continuous futures data.

    When Yahoo Finance switches from an expiring contract to the next one
    (quarterly), a sudden price gap appears. This detects such gaps and shifts
    all pre-gap bars so the series is smooth. Post-gap (current) prices stay real.

    Handles two scenarios:
    1. Gap within a session (< 10 min between bars) - uses ATR*3 threshold
    2. Gap at session boundary (overnight/merge point) during rollover window -
       uses higher ATR*5 threshold to distinguish from normal overnight gaps
    """
    if len(df) < 20:
        return df

    df = df.copy()
    closes = df["close"].values
    opens = df["open"].values
    times = df.index

    # Rolling ATR for adaptive threshold
    atr_series = ta.atr(df["high"], df["low"], df["close"], length=14)

    # Scan for gaps between consecutive bars (newest-to-oldest)
    adjustments = []
    for i in range(len(df) - 1, 0, -1):
        gap = opens[i] - closes[i - 1]

        minutes_between = (times[i] - times[i - 1]).total_seconds() / 60

        current_atr = atr_series.iloc[i - 1]
        if pd.isna(current_atr):
            current_atr = 30  # fallback for early bars

        if minutes_between > 10:
            # Session boundary — only check during rollover windows
            if not _is_rollover_window(times[i]):
                continue
            threshold = max(current_atr * 5, 150)
        else:
            threshold = max(current_atr * 3, 100)

        if abs(gap) > threshold:
            adjustments.append((i, gap))
            logger.info(
                f"ROLLOVER GAP (inter-bar): {times[i]} | "
                f"Gap: {gap:+.2f} pts | ATR: {current_atr:.1f} | "
                f"Threshold: {threshold:.1f}"
            )

    # Also detect intra-bar rollover: YF may switch contract WITHIN a bar,
    # creating a single candle with an abnormally large body (open on old
    # contract, close on new contract).
    if not adjustments:
        highs = df["high"].values
        lows = df["low"].values
        for i in range(14, len(df)):
            body = abs(closes[i] - opens[i])
            bar_range = highs[i] - lows[i]
            current_atr = atr_series.iloc[i]
            if pd.isna(current_atr):
                continue
            # Flag if bar body > 4x ATR and we're in a rollover window
            if body > current_atr * 4 and _is_rollover_window(times[i]):
                gap = closes[i] - opens[i]
                adjustments.append((i, gap))
                logger.info(
                    f"ROLLOVER GAP (intra-bar): {times[i]} | "
                    f"Body: {body:.2f} pts | ATR: {current_atr:.1f} | "
                    f"Range: {bar_range:.2f}"
                )
                break  # only one rollover per fetch

    # Back-adjust: shift all pre-gap bars by the gap amount
    for gap_idx, gap_amount in adjustments:
        for col in ["open", "high", "low", "close"]:
            df.iloc[:gap_idx, df.columns.get_loc(col)] -= gap_amount
        logger.info(f"  Adjusted {gap_idx} bars backward by {gap_amount:+.2f} pts")

    return df


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
    """Manages multiple virtual positions with runner mode.

    Mirrors simulator.py logic exactly:
    - Phase 1 (active): Check SL/TP each bar, timeout at max_bars_held
    - Phase 2 (runner): After TP1 hit, trail with ATR*1.5, timeout at runner_max_bars
    - Conservative: if both SL and TP could trigger on same bar, SL wins
    """

    def __init__(self):
        self.open_positions: dict[str, dict] = {}  # keyed by position ID
        self.starting_capital = float(STARTING_CAPITAL)

    # Backward compat property
    @property
    def open_pos(self):
        if not self.open_positions:
            return None
        return next(iter(self.open_positions.values()))

    def has_open_position(self) -> bool:
        return len(self.open_positions) > 0

    def get_open_positions(self) -> list[dict]:
        return list(self.open_positions.values())

    def open_position(self, signal: dict, account: float = None) -> dict:
        """Create a new virtual position from a signal."""
        tier = signal.get("tier", 1)
        contracts = signal.get("contracts", 2)

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
            "contracts": contracts,
            "tier": tier,

            # Phase tracking
            "phase": "active",
            "tp1_hit": False,
            "tp1_exit_price": None,
            "tp1_exit_time": None,
            "tp1_pnl": 0.0,

            # Runner tracking (only used for Tier 1)
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
        self.open_positions[pos["id"]] = pos
        tier_str = f"T{tier}" if tier else "T1"
        logger.info(
            f"POSITION OPENED [{tier_str}]: {signal['direction'].upper()} @ {signal['entry_price']:.2f} "
            f"| SL: {signal['sl_price']:.2f} | TP: {signal['tp_price']:.2f} "
            f"| {contracts}c | Score: {signal['score']} "
            f"| Open positions: {len(self.open_positions)}"
        )
        return pos

    def update_all_on_bar(self, bar: dict) -> list[tuple[dict, str]]:
        """Update ALL open positions with new bar data.

        Returns list of (closed_position_data, event) tuples for positions that closed.
        """
        closed = []
        # Iterate a copy since we may remove during iteration
        for pos_id in list(self.open_positions.keys()):
            pos = self.open_positions[pos_id]
            event = self._update_single_position(pos, bar)
            if event == "CLOSED":
                closed.append((dict(pos), event))  # copy before removing
                del self.open_positions[pos_id]
        return closed

    def _update_single_position(self, pos: dict, bar: dict) -> str | None:
        """Update a single position with new bar data."""
        bar_high = bar["high"]
        bar_low = bar["low"]
        bar_close = bar["close"]
        bar_time = bar.get("time", str(datetime.now(ET)))

        if pos["phase"] == "active":
            return self._update_active_phase(pos, bar_high, bar_low, bar_close, bar_time)
        elif pos["phase"] == "runner":
            return self._update_runner_phase(pos, bar_high, bar_low, bar_close, bar_time)
        return None

    # Keep old method for backward compat but it now calls update_all_on_bar
    def update_on_bar(self, bar: dict) -> str | None:
        closed = self.update_all_on_bar(bar)
        if closed:
            return "CLOSED"
        return None

    def _update_active_phase(self, pos, bar_high, bar_low, bar_close, bar_time) -> str | None:
        """Phase 1: Check SL/TP, manage timeout."""
        pos["bars_held"] += 1
        direction = pos["direction"]
        entry = pos["entry_price"]
        sl = pos["sl_price"]
        tp = pos["tp_price"]
        contracts = pos.get("contracts", 2)
        tier = pos.get("tier", 1)

        # Check SL/TP
        if direction == "buy":
            sl_hit = bar_low <= sl
            tp_hit = bar_high >= tp
        else:
            sl_hit = bar_high >= sl
            tp_hit = bar_low <= tp

        # Conservative: SL wins if both hit
        if sl_hit:
            pnl_pts = (sl - entry) if direction == "buy" else (entry - sl)
            pnl_dollars = pnl_pts * POINT_VALUE * contracts
            self._close_position(pos, sl, bar_time, "SL", pnl_dollars, pnl_pts)
            return "CLOSED"

        if tp_hit:
            if direction == "buy":
                tp_pnl_pts = tp - entry
            else:
                tp_pnl_pts = entry - tp

            if tier == 2:
                # Tier 2: 1 contract, no runner — full close at TP
                pnl_dollars = tp_pnl_pts * POINT_VALUE * 1
                self._close_position(pos, tp, bar_time, "TP", pnl_dollars, tp_pnl_pts)
                return "CLOSED"

            # Tier 1: TP1 hit — close contract 1, start runner on contract 2
            c1_pnl = tp_pnl_pts * POINT_VALUE * 1  # 1 contract at TP

            pos["tp1_hit"] = True
            pos["tp1_exit_price"] = tp
            pos["tp1_exit_time"] = bar_time
            pos["tp1_pnl"] = c1_pnl

            # Start runner phase
            pos["phase"] = "runner"
            sl_dist = pos.get("sl_distance", 0)
            if direction == "buy":
                pos["runner_sl"] = entry + sl_dist * 3.0  # Lock SL*3 profit
            else:
                pos["runner_sl"] = entry - sl_dist * 3.0  # Lock SL*3 profit
            pos["runner_extreme"] = tp  # Best price = TP level
            pos["runner_trail_dist"] = pos["atr_at_entry"] * TRAIL_ATR_MULT
            pos["runner_bars"] = 0

            logger.info(
                f"TP1 HIT! +{tp_pnl_pts:.2f} pts (${c1_pnl:+,.0f}) | "
                f"Runner active, trail dist: {pos['runner_trail_dist']:.2f}"
            )
            return "TP1_HIT"

        # Check timeout
        if pos["bars_held"] >= MAX_BARS_HELD:
            pnl_pts = (bar_close - entry) if direction == "buy" else (entry - bar_close)
            pnl_dollars = pnl_pts * POINT_VALUE * contracts
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
        """Finalize and close the position. Removal from dict happens in update_all_on_bar."""
        pos["exit_price"] = exit_price
        pos["exit_time"] = exit_time
        pos["outcome"] = outcome
        pos["total_pnl"] = total_pnl
        if pnl_pts is not None:
            pos["pnl_points"] = pnl_pts
        pos["account_after"] = pos["account_before"] + total_pnl
        pos["phase"] = "closed"

        logger.info(
            f"TRADE CLOSED: {outcome} | {pos['direction'].upper()} "
            f"@ {pos['entry_price']:.2f} → {exit_price:.2f} | "
            f"P&L: ${total_pnl:+,.0f} | Bars: {pos['bars_held']} "
            f"| Remaining open: {len(self.open_positions) - 1}"
        )

    def get_unrealized_pnl(self, current_price: float) -> float | None:
        """Calculate total unrealized P&L across all open positions."""
        if not self.open_positions:
            return None
        return self.get_total_unrealized_pnl(current_price)

    def get_position_unrealized_pnl(self, pos: dict, current_price: float) -> float:
        """Calculate unrealized P&L for a single position."""
        entry = pos.get("entry_price")
        direction = pos.get("direction")
        if not entry or not direction:
            return 0.0
        contracts = pos.get("contracts", 2)

        if pos["phase"] == "active":
            pts = (current_price - entry) if direction == "buy" else (entry - current_price)
            return pts * POINT_VALUE * contracts
        elif pos["phase"] == "runner":
            pts = (current_price - entry) if direction == "buy" else (entry - current_price)
            return pos["tp1_pnl"] + (pts * POINT_VALUE * 1)
        return 0.0

    def get_total_unrealized_pnl(self, current_price: float) -> float:
        """Sum unrealized P&L across all open positions."""
        total = 0.0
        for pos in self.open_positions.values():
            total += self.get_position_unrealized_pnl(pos, current_price)
        return total

    def force_close_at_market(self, pos_id: str, current_price: float, bar_time: str) -> dict | None:
        """Force-close a position at current market price (weekend trim)."""
        pos = self.open_positions.get(pos_id)
        if not pos:
            return None

        direction = pos["direction"]
        entry = pos["entry_price"]

        if pos["phase"] == "runner":
            # Runner: TP1 already banked, close runner contract at market
            runner_pts = (current_price - entry) if direction == "buy" else (entry - current_price)
            runner_pnl = runner_pts * POINT_VALUE * 1
            pos["runner_exit_price"] = current_price
            pos["runner_exit_time"] = bar_time
            pos["runner_pnl"] = runner_pnl
            pos["runner_outcome"] = "weekend_trim"
            total_pnl = pos["tp1_pnl"] + runner_pnl
            outcome = "TP+weekend_trim"
            self._close_position(pos, current_price, bar_time, outcome, total_pnl)
        else:
            # Active: close all contracts at market
            contracts = pos.get("contracts", 2)
            pnl_pts = (current_price - entry) if direction == "buy" else (entry - current_price)
            total_pnl = pnl_pts * POINT_VALUE * contracts
            outcome = "weekend_trim"
            self._close_position(pos, current_price, bar_time, outcome, total_pnl, pnl_pts)
        closed_pos = dict(pos)
        del self.open_positions[pos_id]

        logger.info(
            f"WEEKEND TRIM: Closed {direction.upper()} @ {entry:.2f} → {current_price:.2f} | "
            f"P&L: ${total_pnl:+,.0f} | Remaining: {len(self.open_positions)}"
        )
        return closed_pos

    def get_worst_case_loss(self) -> float:
        """Sum of max potential loss (SL hit) for all open positions."""
        total = 0.0
        for pos in self.open_positions.values():
            if pos["phase"] == "runner":
                # Runner SL is at breakeven or better — worst case is minimal
                entry = pos["entry_price"]
                runner_sl = pos.get("runner_sl", entry)
                if pos["direction"] == "buy":
                    worst_pts = max(0, entry - runner_sl)
                else:
                    worst_pts = max(0, runner_sl - entry)
                total += worst_pts * POINT_VALUE * 1
            else:
                sl_dist = pos.get("sl_distance", 0)
                contracts = pos.get("contracts", 2)
                total += sl_dist * POINT_VALUE * contracts
        return total


# ═══════════════════════════════════════════════════════════════════════════
# Daily P&L Tracker
# ═══════════════════════════════════════════════════════════════════════════

class DailyPnLTracker:
    """Enforces Smart Daily Loss limit ($1,100)."""

    def __init__(self, daily_loss_limit: float = SMART_DL):
        self.daily_loss_limit = daily_loss_limit
        self.daily_pnl: dict[str, float] = {}

    def get_today_key(self) -> str:
        return datetime.now(ET).strftime("%Y-%m-%d")

    def record_pnl(self, pnl: float, entry_time: str | None = None):
        """Record P&L against the trade's ENTRY date, not exit date.

        Each calendar day starts from $0. A trade that opened on day A
        and closed on day B counts toward day A's daily loss limit.
        """
        if entry_time:
            key = entry_time[:10]  # "2024-09-18T07:30:00" -> "2024-09-18"
        else:
            key = self.get_today_key()
        self.daily_pnl[key] = self.daily_pnl.get(key, 0) + pnl

    def can_take_trade(self, unrealized_pnl: float = 0) -> bool:
        """Check if a new trade is allowed under Smart Daily Loss.

        Uses realized + unrealized P&L. Open positions always run to completion.
        If total P&L recovers above the limit, trading resumes.
        """
        total = self.get_today_pnl() + unrealized_pnl
        return total > -self.daily_loss_limit

    def get_today_pnl(self) -> float:
        return self.daily_pnl.get(self.get_today_key(), 0)

    def get_today_status(self, unrealized_pnl: float = 0) -> str:
        pnl = self.get_today_pnl() + unrealized_pnl
        if pnl <= -self.daily_loss_limit:
            return "LOCKED"
        elif pnl <= -(self.daily_loss_limit - 200):
            return "WARNING"
        return "OK"

    def get_budget_remaining(self, unrealized_pnl: float = 0) -> float:
        return self.daily_loss_limit + self.get_today_pnl() + unrealized_pnl

    def to_dict(self) -> dict:
        # Prune entries older than 30 days to prevent unbounded growth
        cutoff = (datetime.now(ET) - timedelta(days=30)).strftime("%Y-%m-%d")
        self.daily_pnl = {k: v for k, v in self.daily_pnl.items() if k >= cutoff}
        return {"daily_pnl": self.daily_pnl, "limit": self.daily_loss_limit}

    def from_dict(self, data: dict):
        self.daily_pnl = data.get("daily_pnl", {})
        self.daily_loss_limit = data.get("limit", SMART_DL)
