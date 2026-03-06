"""Multi-timeframe resampling with lookahead bias prevention."""

from __future__ import annotations

import pandas as pd


# Mapping from friendly aliases to pandas resample rules
_TF_MAP = {
    "15min": "15min",
    "15": "15min",
    "m15": "15min",
    "M15": "15min",
    "1h": "1h",
    "1H": "1h",
    "60": "1h",
    "4h": "4h",
    "4H": "4h",
    "240": "4h",
    "1d": "1D",
    "1D": "1D",
    "D": "1D",
    "daily": "1D",
}


def resample_ohlcv(df_5m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample 5-minute OHLCV data to a higher timeframe.

    Uses closed='left', label='left' to prevent lookahead bias:
    - A 1H bar labeled 09:30 contains data from 09:30 to 10:25
    - It is only available AFTER 10:30 (enforced via shift in aligner)

    Args:
        df_5m: 5-minute OHLCV DataFrame with DatetimeIndex.
        timeframe: Target timeframe ('1h', '4h', '1d').

    Returns:
        Resampled OHLCV DataFrame.
    """
    rule = _TF_MAP.get(timeframe, timeframe)

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    # Only include columns that exist
    agg = {k: v for k, v in agg.items() if k in df_5m.columns}

    df_htf = (
        df_5m
        .resample(rule, closed="left", label="left")
        .agg(agg)
        .dropna(subset=["close"])
    )

    return df_htf
