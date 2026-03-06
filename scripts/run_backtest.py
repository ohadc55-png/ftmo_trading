"""CLI script to run backtest on MultiTF Day Trader signals.

Usage:
    python scripts/run_backtest.py --yfinance
    python scripts/run_backtest.py --csv data/raw/nq_5m.csv --capital 100000
    python scripts/run_backtest.py --yfinance --output trades.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.timeutils import add_session_flags
from src.data.loader import load_csv, load_yfinance, load_sqlite
from src.engines.mtf_trend import compute_mtf_trend
from src.engines.volume import compute_volume
from src.engines.key_levels import compute_key_levels
from src.engines.vwap import compute_vwap
from src.engines.composite import compute_composite
from src.backtest.simulator import simulate_trades, trades_to_dataframe
from src.backtest.analysis import (
    compute_stats,
    breakdown_by_outcome,
    breakdown_by_direction,
    breakdown_by_score_range,
    top_trades,
)


def run_pipeline(cfg, csv_path=None, use_yfinance=False, db_path=None, db_table=None, start_date=None, end_date=None):
    """Run full indicator pipeline, return scored DataFrame."""
    print("Loading data...")
    if db_path:
        table = db_table or f"ohlcv_{cfg.base_timeframe.replace('min', 'm').replace('m', 'm')}"
        # Map base_timeframe to table name: "5m" -> "ohlcv_5m", "1m" -> "ohlcv_1m"
        tf = cfg.base_timeframe.replace("min", "m")
        table = db_table or f"ohlcv_{tf}"
        df = load_sqlite(db_path, table=table, tz=cfg.timezone, start_date=start_date, end_date=end_date)
        print(f"  Loaded {len(df)} bars from {db_path} ({table})")
    elif csv_path:
        df = load_csv(csv_path, tz=cfg.timezone)
        print(f"  Loaded {len(df)} bars from {csv_path}")
    elif use_yfinance:
        interval = cfg.base_timeframe.replace("min", "m")
        period = "7d" if "1m" in cfg.base_timeframe else "60d"
        df = load_yfinance(ticker=cfg.ticker, interval=interval, period=period, tz=cfg.timezone)
        print(f"  Downloaded {len(df)} bars from yfinance ({cfg.ticker}, {interval})")
    else:
        print("ERROR: Provide --csv <path>, --yfinance, or --db <path>")
        sys.exit(1)

    print(f"  Date range: {df.index[0]} -- {df.index[-1]}")

    ecfg = cfg.engines
    df = add_session_flags(
        df, tz=cfg.timezone,
        premarket_start_hour=ecfg.key_levels.premarket_start_hour,
        premarket_end_hour=ecfg.key_levels.premarket_end_hour,
        rth_start_hour=ecfg.vwap.rth_start_hour,
        rth_start_minute=ecfg.vwap.rth_start_minute,
    )

    print("Computing engines...")
    df = compute_mtf_trend(df, ecfg.mtf_trend, cfg.mtf_timeframes)
    df = compute_volume(df, ecfg.volume)
    df = compute_key_levels(df, ecfg.key_levels)
    df = compute_vwap(df, ecfg.vwap)
    df = compute_composite(df, cfg.scoring, atr_period=cfg.backtest.atr_period)

    n_buy = df["signal_buy"].sum()
    n_sell = df["signal_sell"].sum()
    print(f"  Signals: {n_buy} BUY, {n_sell} SELL")

    return df


def run_backtest(
    csv_path: str | None = None,
    use_yfinance: bool = False,
    db_path: str | None = None,
    db_table: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    config_path: str = "config/settings.yaml",
    starting_capital: float = 100_000.0,
    output_path: str | None = None,
) -> None:
    cfg = load_config(config_path)
    bt = cfg.backtest

    df = run_pipeline(cfg, csv_path, use_yfinance, db_path, db_table, start_date, end_date)

    # Simulate
    print("\nSimulating trades...")
    trade_results = simulate_trades(
        df,
        starting_capital=starting_capital,
        point_value=bt.point_value,
        sl_atr_mult=bt.sl_atr_mult,
        rr_ratio=bt.rr_ratio,
        risk_pct=bt.risk_pct / 100 if bt.risk_pct > 1 else bt.risk_pct,
        max_bars_held=bt.max_bars_held,
        use_runner=bt.use_runner,
        tp_contracts=bt.tp_contracts,
        runner_contracts=bt.runner_contracts,
        trail_atr_mult=bt.trail_atr_mult,
        runner_max_bars=bt.runner_max_bars,
    )

    trades_df = trades_to_dataframe(trade_results)
    print(f"  Trades executed: {len(trades_df)}")

    if trades_df.empty:
        print("  No trades were executed.")
        return

    stats = compute_stats(trades_df, starting_capital)

    print(f"\n{'=' * 65}")
    print(f"  BACKTEST RESULTS  |  Capital: ${starting_capital:,.0f}")
    print(f"  Point value: ${bt.point_value}  |  SL: ATR x{bt.sl_atr_mult}  |  RR: {bt.rr_ratio}")
    print(f"{'=' * 65}")
    print(f"  Total trades:       {stats['total_trades']}")
    print(f"  Winners: {stats['winners']}  |  Losers: {stats['losers']}  |  BE: {stats['breakeven']}")
    print(f"  Win rate:           {stats['win_rate']:.1f}%")
    print(f"  TP: {stats['tp_exits']}  |  SL: {stats['sl_exits']}  |  Timeout: {stats['timeout_exits']}")
    print(f"{'=' * 65}")
    print(f"  Total P&L:          ${stats['total_pnl']:+,.2f}")
    print(f"  Total return:       {stats['total_return_pct']:+.2f}%")
    print(f"  Final account:      ${stats['final_account']:,.2f}")
    print(f"  Profit factor:      {stats['profit_factor']:.2f}")
    print(f"  Expectancy:         ${stats['expectancy']:+,.2f} per trade")
    print(f"{'=' * 65}")
    print(f"  Avg win:            ${stats['avg_win']:+,.2f}")
    print(f"  Avg loss:           ${stats['avg_loss']:+,.2f}")
    print(f"  Largest win:        ${stats['largest_win']:+,.2f}")
    print(f"  Largest loss:       ${stats['largest_loss']:+,.2f}")
    print(f"{'=' * 65}")
    print(f"  Max drawdown:       ${stats['max_drawdown']:+,.2f}  ({stats['max_drawdown_pct']:.2f}%)")
    print(f"  Avg bars held:      {stats['avg_bars_held']:.1f}")
    print(f"  Max consec wins:    {stats['max_consec_wins']}")
    print(f"  Max consec losses:  {stats['max_consec_losses']}")
    print(f"{'=' * 65}")

    # Breakdown by direction
    print(f"\n{'=' * 65}")
    print("  BY DIRECTION")
    print(f"{'=' * 65}")
    dir_df = breakdown_by_direction(trades_df)
    for _, row in dir_df.iterrows():
        print(f"  {row['direction'].upper():>4s}: {row['trades']:.0f} trades  "
              f"WR: {row['win_rate']:.1f}%  P&L: ${row['total_pnl']:+,.2f}")

    # Breakdown by outcome
    print(f"\n{'=' * 65}")
    print("  BY EXIT TYPE")
    print(f"{'=' * 65}")
    out_df = breakdown_by_outcome(trades_df)
    for _, row in out_df.iterrows():
        print(f"  {row['outcome']:>7s}: {row['trades']:.0f} trades  "
              f"P&L: ${row['total_pnl']:+,.2f}  Avg bars: {row['avg_bars']:.1f}")

    # Breakdown by score range
    print(f"\n{'=' * 65}")
    print("  BY SCORE RANGE")
    print(f"{'=' * 65}")
    score_df = breakdown_by_score_range(trades_df)
    for _, row in score_df.iterrows():
        print(f"  {row['score_range']:>8s}: {row['trades']:3.0f} trades  "
              f"WR: {row['win_rate']:5.1f}%  "
              f"P&L: ${row['total_pnl']:+10,.2f}  "
              f"Avg: ${row['avg_pnl']:+8,.2f}")

    # Top trades
    print(f"\n{'=' * 65}")
    print("  TOP 10 BEST TRADES")
    print(f"{'=' * 65}")
    best = top_trades(trades_df, n=10, worst=False)
    for _, t in best.iterrows():
        print(f"  {str(t['entry_time'])[:16]}  {t['direction']:>4s}  "
              f"Score {t['score']:.1f}  "
              f"${t['pnl_dollars']:+10,.2f}  ({t['pnl_points']:+.2f} pts)  "
              f"{t['outcome']}  {t['bars_held']:.0f} bars")

    print(f"\n{'=' * 65}")
    print("  TOP 10 WORST TRADES")
    print(f"{'=' * 65}")
    worst = top_trades(trades_df, n=10, worst=True)
    for _, t in worst.iterrows():
        print(f"  {str(t['entry_time'])[:16]}  {t['direction']:>4s}  "
              f"Score {t['score']:.1f}  "
              f"${t['pnl_dollars']:+10,.2f}  ({t['pnl_points']:+.2f} pts)  "
              f"{t['outcome']}  {t['bars_held']:.0f} bars")

    # Equity curve snapshot
    print(f"\n{'=' * 65}")
    print("  EQUITY CURVE")
    print(f"{'=' * 65}")
    step = max(1, len(trades_df) // 15)
    for idx in range(0, len(trades_df), step):
        row = trades_df.iloc[idx]
        print(f"  #{idx + 1:3d}  {str(row['entry_time'])[:16]}  "
              f"${row['account_after']:>12,.2f}")
    last = trades_df.iloc[-1]
    print(f"  #{len(trades_df):3d}  {str(last['entry_time'])[:16]}  "
          f"${last['account_after']:>12,.2f}  (FINAL)")
    print(f"{'=' * 65}")

    if output_path:
        trades_df.to_csv(output_path, index=False)
        print(f"\nTrade log saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MultiTF Day Trader Backtest")
    parser.add_argument("--csv", type=str, help="Path to CSV file")
    parser.add_argument("--yfinance", action="store_true", help="Download from Yahoo Finance")
    parser.add_argument("--db", type=str, help="Path to SQLite database")
    parser.add_argument("--table", type=str, help="DB table name (default: auto from timeframe)")
    parser.add_argument("--start", type=str, help="Start date filter (e.g. 2025-03-01)")
    parser.add_argument("--end", type=str, help="End date filter")
    parser.add_argument("--config", type=str, default="config/settings.yaml")
    parser.add_argument("--capital", type=float, default=100_000.0, help="Starting capital ($)")
    parser.add_argument("--output", type=str, help="Save trade log to CSV")
    args = parser.parse_args()

    run_backtest(
        csv_path=args.csv,
        use_yfinance=args.yfinance,
        db_path=args.db,
        db_table=args.table,
        start_date=args.start,
        end_date=args.end,
        config_path=args.config,
        starting_capital=args.capital,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
