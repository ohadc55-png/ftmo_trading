"""Engine 1: Multi-Timeframe Trend — EMA Stack + HH/HL Structure.

Analyzes H4, H1, M15 for trend direction.
Each TF contributes 0-2 points (EMA alignment + structure).
Raw score 0-6, normalized to 0-10.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils.config import MTFTrendConfig
from src.data.resampler import resample_ohlcv


def compute_mtf_trend(
    df: pd.DataFrame,
    cfg: MTFTrendConfig,
    mtf_timeframes: list[str] = None,
) -> pd.DataFrame:
    """Compute MTF trend score and direction.

    Args:
        df: Base timeframe DataFrame (M1 or M5).
        cfg: MTF trend configuration.
        mtf_timeframes: List of higher timeframes (e.g., ["15min", "1h", "4h"]).

    Returns:
        DataFrame with added columns: mtf_score, mtf_direction,
        plus per-TF debug columns.
    """
    if mtf_timeframes is None:
        mtf_timeframes = ["15min", "1h", "4h"]

    # Compute per-TF scores and merge onto base
    tf_bull_cols = []
    tf_bear_cols = []

    for tf in mtf_timeframes:
        df_htf = resample_ohlcv(df, tf)
        htf_flags = _compute_tf_flags(df_htf, cfg, tf)

        # Shift-before-merge: prevent lookahead
        htf_shifted = htf_flags.shift(1)
        merged = htf_shifted.reindex(df.index, method="ffill")

        for col in merged.columns:
            df[col] = merged[col]

        tf_label = tf.replace("min", "m")
        tf_bull_cols.append(f"tf_{tf_label}_bull")
        tf_bear_cols.append(f"tf_{tf_label}_bear")

    # Sum raw scores: each TF contributes 0-2 (EMA + structure)
    bull_score = pd.Series(0.0, index=df.index)
    bear_score = pd.Series(0.0, index=df.index)

    for tf in mtf_timeframes:
        tf_label = tf.replace("min", "m")
        bull_score += df[f"tf_{tf_label}_ema_bull"].fillna(0).astype(float)
        bull_score += df[f"tf_{tf_label}_struct_bull"].fillna(0).astype(float)
        bear_score += df[f"tf_{tf_label}_ema_bear"].fillna(0).astype(float)
        bear_score += df[f"tf_{tf_label}_struct_bear"].fillna(0).astype(float)

    # Normalize to 0-10 (max raw = 2 per TF)
    max_raw = len(mtf_timeframes) * 2.0
    df["mtf_bull_raw"] = bull_score
    df["mtf_bear_raw"] = bear_score
    df["mtf_score_bull"] = (bull_score / max_raw * 10.0).clip(0, 10)
    df["mtf_score_bear"] = (bear_score / max_raw * 10.0).clip(0, 10)

    # Direction: majority vote
    n_tf = len(mtf_timeframes)
    bull_count = pd.Series(0, index=df.index)
    bear_count = pd.Series(0, index=df.index)
    for tf in mtf_timeframes:
        tf_label = tf.replace("min", "m")
        col_b = f"tf_{tf_label}_bull"
        col_s = f"tf_{tf_label}_bear"
        if col_b in df.columns:
            bull_count += df[col_b].fillna(0).astype(int)
        if col_s in df.columns:
            bear_count += df[col_s].fillna(0).astype(int)

    df["mtf_direction"] = np.where(
        bull_count > bear_count, "bull",
        np.where(bear_count > bull_count, "bear", "neutral")
    )

    # Final mtf_score: use the relevant direction's score
    df["mtf_score"] = np.where(
        df["mtf_direction"] == "bull", df["mtf_score_bull"],
        np.where(df["mtf_direction"] == "bear", df["mtf_score_bear"], 0.0)
    )

    return df


def _compute_tf_flags(
    df_htf: pd.DataFrame,
    cfg: MTFTrendConfig,
    tf: str,
) -> pd.DataFrame:
    """Compute EMA stack + HH/HL flags for a single timeframe.

    Returns a DataFrame with bool flag columns.
    """
    tf_label = tf.replace("min", "m")
    out = pd.DataFrame(index=df_htf.index)

    # EMA computation
    ema_fast = df_htf["close"].ewm(span=cfg.ema_fast, adjust=False).mean()
    ema_mid = df_htf["close"].ewm(span=cfg.ema_mid, adjust=False).mean()
    ema_slow = df_htf["close"].ewm(span=cfg.ema_slow, adjust=False).mean()

    ema_bull = (ema_fast > ema_mid) & (ema_mid > ema_slow)
    ema_bear = (ema_fast < ema_mid) & (ema_mid < ema_slow)

    out[f"tf_{tf_label}_ema_bull"] = ema_bull
    out[f"tf_{tf_label}_ema_bear"] = ema_bear

    # Swing structure: HH/HL or LH/LL (left-side only)
    lb = cfg.swing_lookback
    highs = df_htf["high"]
    lows = df_htf["low"]

    swing_high = highs.rolling(lb, min_periods=lb).max()
    swing_low = lows.rolling(lb, min_periods=lb).min()

    # Previous swing extremes (shift by lookback)
    prev_swing_high = swing_high.shift(lb)
    prev_swing_low = swing_low.shift(lb)

    # HH + HL = bullish structure
    hh = swing_high > prev_swing_high
    hl = swing_low > prev_swing_low
    struct_bull = hh & hl

    # LH + LL = bearish structure
    lh = swing_high < prev_swing_high
    ll = swing_low < prev_swing_low
    struct_bear = lh & ll

    out[f"tf_{tf_label}_struct_bull"] = struct_bull.fillna(False)
    out[f"tf_{tf_label}_struct_bear"] = struct_bear.fillna(False)

    # Combined direction flag
    out[f"tf_{tf_label}_bull"] = ema_bull | struct_bull.fillna(False)
    out[f"tf_{tf_label}_bear"] = ema_bear | struct_bear.fillna(False)

    return out
