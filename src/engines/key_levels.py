"""Engine 3: Key Levels Breakout — Daily/PrevDay/PreMarket levels.

Detects breakouts of key price levels with volume confirmation.
Scores 0-10 based on which levels were broken and whether retested.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils.config import KeyLevelsConfig


def compute_key_levels(df: pd.DataFrame, cfg: KeyLevelsConfig) -> pd.DataFrame:
    """Compute key levels, breakout detection, and score.

    Expects df to have: is_premarket, trading_day, vol_surge columns.
    Adds: daily_high, daily_low, prev_day_high, prev_day_low, prev_day_close,
          premarket_high, premarket_low, levels_score, plus breakout flags.
    """
    buf = cfg.breakout_buffer_points

    # --- Daily High / Low (rolling intraday, resets each day) ---
    _compute_daily_hl(df)

    # --- Previous Day H/L/C ---
    _compute_prev_day(df)

    # --- Pre-Market H/L ---
    _compute_premarket_hl(df)

    # --- Breakout detection ---
    close = df["close"]
    surge = df["vol_surge"] if "vol_surge" in df.columns else pd.Series(True, index=df.index)

    # Prev Day breakouts (3 points each)
    df["brk_prev_high"] = (close > df["prev_day_high"] + buf) & surge
    df["brk_prev_low"] = (close < df["prev_day_low"] - buf) & surge

    # Daily H/L breakouts (2 points each)
    df["brk_daily_high"] = (close > df["daily_high"].shift(1) + buf) & surge
    df["brk_daily_low"] = (close < df["daily_low"].shift(1) - buf) & surge

    # Pre-Market breakouts (2 points each)
    df["brk_premarket_high"] = (close > df["premarket_high"] + buf) & surge
    df["brk_premarket_low"] = (close < df["premarket_low"] - buf) & surge

    # Retest detection: price returns to a broken level within buffer
    df["retest_prev_high"] = (
        df["brk_prev_high"].shift(1).fillna(0).rolling(10, min_periods=1).max().astype(bool)
        & (abs(close - df["prev_day_high"]) <= buf * 2)
    )
    df["retest_prev_low"] = (
        df["brk_prev_low"].shift(1).fillna(0).rolling(10, min_periods=1).max().astype(bool)
        & (abs(close - df["prev_day_low"]) <= buf * 2)
    )

    # --- Scoring ---
    score = pd.Series(0.0, index=df.index)

    # Prev Day breakouts: 3 points
    score += df["brk_prev_high"].astype(float) * 3.0
    score += df["brk_prev_low"].astype(float) * 3.0

    # Daily H/L breakouts: 2 points
    score += df["brk_daily_high"].astype(float) * 2.0
    score += df["brk_daily_low"].astype(float) * 2.0

    # Pre-Market breakouts: 2 points
    score += df["brk_premarket_high"].astype(float) * 2.0
    score += df["brk_premarket_low"].astype(float) * 2.0

    # Retest bonus: 2 points
    score += df["retest_prev_high"].astype(float) * 2.0
    score += df["retest_prev_low"].astype(float) * 2.0

    df["levels_score"] = score.clip(0, 10)

    return df


def _compute_daily_hl(df: pd.DataFrame) -> None:
    """Compute rolling intraday high/low, resetting each trading day."""
    days = df["trading_day"]
    day_change = days != days.shift(1)

    # Expand high/low per day group
    daily_high = df["high"].copy()
    daily_low = df["low"].copy()

    # Use groupby cummax/cummin for efficiency
    day_groups = days.values
    df["daily_high"] = df.groupby(day_groups)["high"].cummax()
    df["daily_low"] = df.groupby(day_groups)["low"].cummin()


def _compute_prev_day(df: pd.DataFrame) -> None:
    """Compute previous day's high, low, close."""
    days = df["trading_day"]
    unique_days = days.unique()

    prev_high = pd.Series(np.nan, index=df.index)
    prev_low = pd.Series(np.nan, index=df.index)
    prev_close = pd.Series(np.nan, index=df.index)

    for i, day in enumerate(unique_days):
        if i == 0:
            continue
        prev = unique_days[i - 1]
        prev_mask = days == prev
        curr_mask = days == day

        if prev_mask.any():
            prev_data = df.loc[prev_mask]
            ph = prev_data["high"].max()
            pl = prev_data["low"].min()
            pc = prev_data["close"].iloc[-1]

            prev_high.loc[curr_mask] = ph
            prev_low.loc[curr_mask] = pl
            prev_close.loc[curr_mask] = pc

    df["prev_day_high"] = prev_high.ffill()
    df["prev_day_low"] = prev_low.ffill()
    df["prev_day_close"] = prev_close.ffill()


def _compute_premarket_hl(df: pd.DataFrame) -> None:
    """Compute pre-market high/low per day."""
    if "is_premarket" not in df.columns:
        df["premarket_high"] = np.nan
        df["premarket_low"] = np.nan
        return

    days = df["trading_day"]
    pm_mask = df["is_premarket"]

    pm_high = pd.Series(np.nan, index=df.index)
    pm_low = pd.Series(np.nan, index=df.index)

    for day in days.unique():
        day_mask = days == day
        day_pm = day_mask & pm_mask

        if day_pm.any():
            pm_data = df.loc[day_pm]
            ph = pm_data["high"].max()
            pl = pm_data["low"].min()
            pm_high.loc[day_mask] = ph
            pm_low.loc[day_mask] = pl

    df["premarket_high"] = pm_high.ffill()
    df["premarket_low"] = pm_low.ffill()
