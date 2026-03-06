"""Engine 4: VWAP — Session-anchored Volume Weighted Average Price.

VWAP resets at RTH start. Scores based on price position relative to VWAP,
crossovers with volume confirmation, and overextension filter.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils.config import VWAPConfig


def compute_vwap(df: pd.DataFrame, cfg: VWAPConfig) -> pd.DataFrame:
    """Compute VWAP line and VWAP score.

    Expects df to have: is_rth, vol_surge, mtf_direction columns.
    Adds: vwap, vwap_dist_pct, vwap_cross_bull, vwap_cross_bear, vwap_score.
    """
    # --- VWAP Calculation (resets at RTH start) ---
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].astype(float)

    # Identify RTH session start: first bar where is_rth becomes True
    rth = df["is_rth"].astype(bool)
    rth_start = rth & (~rth.shift(1).fillna(False))

    # Create session IDs for cumulative grouping
    session_id = rth_start.cumsum()

    # Cumulative TP*Volume and Volume per session
    tp_vol = tp * vol
    cum_tp_vol = tp_vol.groupby(session_id).cumsum()
    cum_vol = vol.groupby(session_id).cumsum()

    df["vwap"] = np.where(cum_vol > 0, cum_tp_vol / cum_vol, tp)

    # --- Distance from VWAP as % of price ---
    df["vwap_dist_pct"] = ((df["close"] - df["vwap"]) / df["close"] * 100).abs()

    # --- Crossover detection ---
    above_vwap = df["close"] > df["vwap"]
    below_vwap = df["close"] < df["vwap"]
    prev_above = above_vwap.shift(1).fillna(False)
    prev_below = below_vwap.shift(1).fillna(False)

    surge = df["vol_surge"] if "vol_surge" in df.columns else pd.Series(True, index=df.index)

    df["vwap_cross_bull"] = above_vwap & prev_below & surge
    df["vwap_cross_bear"] = below_vwap & prev_above & surge

    # --- Scoring ---
    direction = df["mtf_direction"] if "mtf_direction" in df.columns else pd.Series("neutral", index=df.index)
    overext_pct = cfg.overextended_pct

    # Base score by condition
    score = pd.Series(0.0, index=df.index)

    # VWAP cross with volume = 10
    score = np.where(df["vwap_cross_bull"] | df["vwap_cross_bear"], 10.0, score)

    # Above/below VWAP aligned with trend = 6
    aligned_bull = above_vwap & (direction == "bull")
    aligned_bear = below_vwap & (direction == "bear")
    score = np.where(
        (score < 10) & (aligned_bull | aligned_bear), 6.0, score
    )

    # On VWAP (within 0.1%) = 3
    on_vwap = df["vwap_dist_pct"] < 0.1
    score = np.where((score < 6) & on_vwap, 3.0, score)

    # Against VWAP stays 0

    # Overextended filter: cap at 3
    overextended = df["vwap_dist_pct"] > overext_pct
    score = np.where(overextended & (score > 3), 3.0, score)

    df["vwap_score"] = pd.Series(score, index=df.index).clip(0, 10)

    return df
