"""Engine 2: Volume Analysis — Volume MA + Surge Detection.

Scores volume strength on a 0-10 scale based on volume ratio to MA.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils.config import VolumeConfig


def compute_volume(df: pd.DataFrame, cfg: VolumeConfig) -> pd.DataFrame:
    """Compute volume score and flags.

    Adds columns: vol_ma, vol_ratio, vol_surge, vol_trend_up, vol_score.
    """
    vol = df["volume"].astype(float)

    # Volume moving average
    df["vol_ma"] = vol.rolling(cfg.ma_period, min_periods=1).mean()

    # Volume ratio
    df["vol_ratio"] = np.where(df["vol_ma"] > 0, vol / df["vol_ma"], 0.0)

    # Surge flag
    df["vol_surge"] = df["vol_ratio"] >= cfg.surge_mult

    # Volume trend: 3 consecutive rising volume bars
    vol_rising = vol > vol.shift(1)
    df["vol_trend_up"] = (
        vol_rising
        & vol_rising.shift(1).fillna(False)
        & vol_rising.shift(2).fillna(False)
    )

    # Scoring (step function)
    ratio = df["vol_ratio"]
    df["vol_score"] = np.select(
        [
            ratio >= 2.0,
            ratio >= 1.5,
            ratio >= 1.2,
            ratio >= 1.0,
        ],
        [10.0, 7.0, 5.0, 3.0],
        default=0.0,
    )

    return df
