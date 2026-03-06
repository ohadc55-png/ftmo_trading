"""Data loading from CSV files, yfinance, and SQLite databases."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np

# Common column name aliases mapping to canonical lowercase names
_COLUMN_ALIASES: dict[str, str] = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "vol": "volume",
    "adj close": "close",
    "adj_close": "close",
    "date": "date",
    "datetime": "date",
    "time": "date",
    "timestamp": "date",
}

REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase canonical form."""
    rename_map = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "_")
        # Try direct match first
        if key in _COLUMN_ALIASES:
            rename_map[col] = _COLUMN_ALIASES[key]
        # Try without underscores
        elif key.replace("_", " ") in _COLUMN_ALIASES:
            rename_map[col] = _COLUMN_ALIASES[key.replace("_", " ")]
    df = df.rename(columns=rename_map)
    return df


def _detect_datetime_column(df: pd.DataFrame) -> str | None:
    """Find the datetime column in the DataFrame."""
    # Check if index is already datetime
    if isinstance(df.index, pd.DatetimeIndex):
        return None  # Index is already datetime

    # Check for 'date' column
    if "date" in df.columns:
        return "date"

    # Check first column if it looks like dates
    first_col = df.columns[0]
    try:
        pd.to_datetime(df[first_col].head(5))
        return first_col
    except (ValueError, TypeError):
        pass

    return None


def load_csv(
    path: str | Path,
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """Load OHLCV data from a CSV file.

    Auto-detects column names and datetime format.
    Returns a DataFrame with DatetimeIndex localized to the given timezone
    and lowercase columns: open, high, low, close, volume.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    df = _normalize_columns(df)

    # Set datetime index
    dt_col = _detect_datetime_column(df)
    if dt_col is not None:
        df[dt_col] = pd.to_datetime(df[dt_col])
        df = df.set_index(dt_col)
    elif not isinstance(df.index, pd.DatetimeIndex):
        # Try parsing the index
        df.index = pd.to_datetime(df.index)

    df.index.name = "datetime"

    # Localize timezone
    if df.index.tz is None:
        try:
            df.index = df.index.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
        except TypeError:
            df.index = df.index.tz_localize(tz)
    else:
        df.index = df.index.tz_convert(tz)

    # Validate required columns
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # Keep only OHLCV, ensure numeric
    df = df[["open", "high", "low", "close", "volume"]].copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort by time, drop duplicates
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Drop rows with NaN in OHLC
    df = df.dropna(subset=["open", "high", "low", "close"])

    # Fill missing volume with 0
    df["volume"] = df["volume"].fillna(0)

    return df


def load_yfinance(
    ticker: str = "NQ=F",
    interval: str = "5m",
    period: str = "60d",
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """Download data from Yahoo Finance.

    Returns same schema as load_csv: DatetimeIndex with OHLCV columns.
    Note: yfinance 5m data is limited to ~60 days of history.
    """
    import yfinance as yf

    data = yf.download(ticker, interval=interval, period=period, progress=False)

    if data.empty:
        raise ValueError(f"No data returned from yfinance for {ticker}")

    # yfinance returns MultiIndex columns when single ticker, flatten
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    df = _normalize_columns(data)

    # Keep only OHLCV
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].copy()

    # Ensure timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(tz)
    else:
        df.index = df.index.tz_convert(tz)

    df.index.name = "datetime"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0)

    return df


def load_sqlite(
    db_path: str | Path,
    table: str = "ohlcv_5m",
    tz: str = "America/New_York",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load OHLCV data from a SQLite database.

    Args:
        db_path: Path to the .db file.
        table: Table name (e.g. ohlcv_5m, ohlcv_1m, ohlcv_1h).
        tz: Target timezone.
        start_date: Optional start filter (ISO format, e.g. '2025-03-01').
        end_date: Optional end filter.

    Returns same schema as load_csv: DatetimeIndex with OHLCV columns.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    query = f"SELECT datetime, open, high, low, close, volume FROM [{table}]"
    conditions = []
    if start_date:
        conditions.append(f"datetime >= '{start_date}'")
    if end_date:
        conditions.append(f"datetime <= '{end_date}'")
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY datetime"

    df = pd.read_sql_query(query, conn)
    conn.close()

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    df.index.name = "datetime"

    # Handle timezone - DB stores UTC timestamps (+00:00)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC", ambiguous="NaT", nonexistent="shift_forward")
    df.index = df.index.tz_convert(tz)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0)

    return df
