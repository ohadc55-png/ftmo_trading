"""CLI script to run the MultiTF Day Trader indicator pipeline.

Usage:
    python scripts/run_indicator.py --yfinance
    python scripts/run_indicator.py --csv data/raw/nq_1m.csv
    python scripts/run_indicator.py --yfinance --output scored.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.timeutils import add_session_flags
from src.data.loader import load_csv, load_yfinance
from src.engines.mtf_trend import compute_mtf_trend
from src.engines.volume import compute_volume
from src.engines.key_levels import compute_key_levels
from src.engines.vwap import compute_vwap
from src.engines.composite import compute_composite


def run(
    csv_path: str | None = None,
    use_yfinance: bool = False,
    config_path: str = "config/settings.yaml",
    output_path: str | None = None,
    last_n: int = 20,
) -> None:
    cfg = load_config(config_path)

    # Load data
    print("Loading data...")
    if csv_path:
        df = load_csv(csv_path, tz=cfg.timezone)
        print(f"  Loaded {len(df)} bars from {csv_path}")
    elif use_yfinance:
        interval = cfg.base_timeframe.replace("min", "m")
        period = "7d" if "1m" in cfg.base_timeframe else "60d"
        df = load_yfinance(ticker=cfg.ticker, interval=interval, period=period, tz=cfg.timezone)
        print(f"  Downloaded {len(df)} bars from yfinance ({cfg.ticker}, {interval})")
    else:
        print("ERROR: Provide --csv <path> or --yfinance")
        sys.exit(1)

    print(f"  Date range: {df.index[0]} -- {df.index[-1]}")

    # Pipeline
    print("Adding session flags...")
    ecfg = cfg.engines
    df = add_session_flags(
        df, tz=cfg.timezone,
        premarket_start_hour=ecfg.key_levels.premarket_start_hour,
        premarket_end_hour=ecfg.key_levels.premarket_end_hour,
        rth_start_hour=ecfg.vwap.rth_start_hour,
        rth_start_minute=ecfg.vwap.rth_start_minute,
    )

    print("Engine 1: MTF Trend...")
    df = compute_mtf_trend(df, ecfg.mtf_trend, cfg.mtf_timeframes)

    print("Engine 2: Volume...")
    df = compute_volume(df, ecfg.volume)

    print("Engine 3: Key Levels...")
    df = compute_key_levels(df, ecfg.key_levels)

    print("Engine 4: VWAP...")
    df = compute_vwap(df, ecfg.vwap)

    print("Engine 5: Composite + Signals...")
    df = compute_composite(df, cfg.scoring, atr_period=cfg.backtest.atr_period)

    # Summary
    n_buy = df["signal_buy"].sum()
    n_sell = df["signal_sell"].sum()
    print(f"\n  Total bars: {len(df)}")
    print(f"  BUY signals: {n_buy}")
    print(f"  SELL signals: {n_sell}")

    # Score distribution
    scores = df["composite_score"].dropna()
    print(f"\n  Score stats:")
    print(f"    min={scores.min():.2f}  max={scores.max():.2f}  "
          f"mean={scores.mean():.2f}  median={scores.median():.2f}")

    above_threshold = (scores >= cfg.scoring.min_score_threshold).sum()
    print(f"    Bars >= {cfg.scoring.min_score_threshold}: {above_threshold}")

    # Sub-score means
    for col in ["mtf_score", "vol_score", "levels_score", "vwap_score"]:
        if col in df.columns:
            s = df[col].dropna()
            print(f"    {col}: mean={s.mean():.2f}  max={s.max():.2f}")

    # Direction distribution
    if "mtf_direction" in df.columns:
        dir_counts = df["mtf_direction"].value_counts()
        print(f"\n  MTF Direction: {dict(dir_counts)}")

    # Last N signal bars
    signals = df[df["signal_buy"] | df["signal_sell"]].tail(last_n)
    if not signals.empty:
        print(f"\n  Last {len(signals)} signals:")
        for ts, row in signals.iterrows():
            d = "BUY" if row["signal_buy"] else "SELL"
            print(f"    {str(ts)[:16]}  {d:>4s}  score={row['composite_score']:.1f}  "
                  f"mtf={row['mtf_score']:.1f}  vol={row['vol_score']:.0f}  "
                  f"lvl={row['levels_score']:.1f}  vwap={row['vwap_score']:.0f}")

    if output_path:
        df.to_csv(output_path)
        print(f"\nSaved {len(df)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MultiTF Day Trader Indicator")
    parser.add_argument("--csv", type=str, help="Path to CSV file")
    parser.add_argument("--yfinance", action="store_true", help="Download from Yahoo Finance")
    parser.add_argument("--config", type=str, default="config/settings.yaml")
    parser.add_argument("--output", type=str, help="Save scored data to CSV")
    parser.add_argument("--last", type=int, default=20, help="Show last N signals")
    args = parser.parse_args()

    run(csv_path=args.csv, use_yfinance=args.yfinance,
        config_path=args.config, output_path=args.output, last_n=args.last)


if __name__ == "__main__":
    main()
