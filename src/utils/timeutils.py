"""Session and time helpers for the MultiTF Day Trader."""

from __future__ import annotations

import pandas as pd

def add_session_flags(
    df: pd.DataFrame,
    tz: str = "America/New_York",
    premarket_start_hour: int = 1,
    premarket_end_hour: int = 9,
    rth_start_hour: int = 9,
    rth_start_minute: int = 30,
) -> pd.DataFrame:
    """Add session flags: is_premarket, is_rth, trading_day.

    All times are in the configured timezone.
    """
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize(tz)
    else:
        idx = idx.tz_convert(tz)

    hours = idx.hour
    minutes = idx.minute

    # Pre-market window
    df["is_premarket"] = (hours >= premarket_start_hour) & (hours < premarket_end_hour)

    # RTH: from rth_start onwards (e.g., 09:30+)
    rth_minutes = hours * 60 + minutes
    rth_start_min = rth_start_hour * 60 + rth_start_minute
    df["is_rth"] = rth_minutes >= rth_start_min

    # Trading day ID: date-based, useful for daily level resets
    df["trading_day"] = idx.date

    return df
