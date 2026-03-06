"""Engine 5: Composite Score + Smart Cooldown → Buy/Sell signals.

Combines all 4 engine scores into a weighted average.
Applies direction filter and smart cooldown state machine.
Also computes ATR for backtest stop-loss calculation.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pandas_ta as ta

from src.utils.config import ScoringConfig


def compute_composite(
    df: pd.DataFrame,
    cfg: ScoringConfig,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Compute composite score and generate buy/sell signals.

    Expects df to have: mtf_score, vol_score, levels_score, vwap_score,
                        mtf_direction columns.
    Adds: composite_score, signal_buy, signal_sell, atr.
    """
    # --- Weighted average (default / SELL weights) ---
    w1 = cfg.weight_mtf_trend
    w2 = cfg.weight_volume
    w3 = cfg.weight_breakout
    w4 = cfg.weight_vwap
    total_w = w1 + w2 + w3 + w4

    mtf = df["mtf_score"].fillna(0)
    vol = df["vol_score"].fillna(0)
    lvl = df["levels_score"].fillna(0)
    vwap = df["vwap_score"].fillna(0)

    df["composite_score"] = (
        (mtf * w1 + vol * w2 + lvl * w3 + vwap * w4) / total_w
    ).clip(0, 10)

    # --- BUY-specific composite score if overrides exist ---
    bw1 = cfg.buy_weight_mtf_trend
    bw2 = cfg.buy_weight_volume
    bw3 = cfg.buy_weight_breakout
    bw4 = cfg.buy_weight_vwap
    has_buy_weights = any(w is not None for w in [bw1, bw2, bw3, bw4])

    if has_buy_weights:
        bw1 = bw1 if bw1 is not None else w1
        bw2 = bw2 if bw2 is not None else w2
        bw3 = bw3 if bw3 is not None else w3
        bw4 = bw4 if bw4 is not None else w4
        total_bw = bw1 + bw2 + bw3 + bw4
        buy_composite = (
            (mtf * bw1 + vol * bw2 + lvl * bw3 + vwap * bw4) / total_bw
        ).clip(0, 10)
    else:
        buy_composite = df["composite_score"]

    # --- Direction filter ---
    direction = df["mtf_direction"].fillna("neutral")
    threshold = cfg.min_score_threshold
    sell_score = df["composite_score"]

    buy_eligible = (buy_composite >= threshold) & (direction == "bull")
    sell_eligible = (sell_score >= threshold) & (direction == "bear")

    # --- Smart Cooldown State Machine ---
    signal_buy = _apply_cooldown(buy_eligible.values, buy_composite.values, threshold)
    signal_sell = _apply_cooldown(sell_eligible.values, sell_score.values, threshold)

    df["signal_buy"] = signal_buy
    df["signal_sell"] = signal_sell

    # Overwrite composite_score on buy bars with buy-specific score
    if has_buy_weights:
        df.loc[df["signal_buy"], "composite_score"] = buy_composite[df["signal_buy"]]

    # --- ATR for backtest ---
    atr = ta.atr(df["high"], df["low"], df["close"], length=atr_period)
    df["atr"] = atr

    return df


def _apply_cooldown(eligible: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    """Smart cooldown state machine.

    States:
        READY (0)         → score >= threshold → SIGNAL_FIRED (1)
        SIGNAL_FIRED (1)  → score < threshold  → COOLING (2)
        COOLING (2)       → score >= threshold  → SIGNAL_FIRED (1)

    Returns bool array: True only on SIGNAL_FIRED transitions.
    """
    READY = 0
    SIGNAL_FIRED = 1
    COOLING = 2

    n = len(eligible)
    signals = np.zeros(n, dtype=bool)
    state = READY

    for i in range(n):
        if state == READY:
            if eligible[i]:
                state = SIGNAL_FIRED
                signals[i] = True
        elif state == SIGNAL_FIRED:
            if scores[i] < threshold:
                state = COOLING
        elif state == COOLING:
            if eligible[i]:
                state = SIGNAL_FIRED
                signals[i] = True

    return signals
