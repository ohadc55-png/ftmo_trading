"""Backtest analysis: statistics, drawdown, breakdown by mode/direction."""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute_stats(trades_df: pd.DataFrame, starting_capital: float = 100_000.0) -> dict:
    """Compute comprehensive backtest statistics.

    Returns a dict with all key metrics.
    """
    if trades_df.empty:
        return {"total_trades": 0}

    n = len(trades_df)
    winners = trades_df[trades_df["pnl_dollars"] > 0]
    losers = trades_df[trades_df["pnl_dollars"] < 0]
    breakeven = trades_df[trades_df["pnl_dollars"] == 0]

    n_win = len(winners)
    n_loss = len(losers)
    n_be = len(breakeven)
    win_rate = n_win / n * 100 if n > 0 else 0

    total_pnl = trades_df["pnl_dollars"].sum()
    avg_win = winners["pnl_dollars"].mean() if n_win > 0 else 0
    avg_loss = losers["pnl_dollars"].mean() if n_loss > 0 else 0
    largest_win = winners["pnl_dollars"].max() if n_win > 0 else 0
    largest_loss = losers["pnl_dollars"].min() if n_loss > 0 else 0

    gross_profit = winners["pnl_dollars"].sum() if n_win > 0 else 0
    gross_loss = abs(losers["pnl_dollars"].sum()) if n_loss > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_pnl = trades_df["pnl_dollars"].mean()
    std_pnl = trades_df["pnl_dollars"].std() if n > 1 else 0

    # Equity curve and drawdown
    equity = _build_equity_curve(trades_df, starting_capital)
    max_dd, max_dd_pct, dd_start, dd_end = _compute_max_drawdown(equity)

    # Average bars held
    avg_bars = trades_df["bars_held"].mean()

    # Consecutive wins/losses
    max_consec_wins, max_consec_losses = _consecutive_streaks(trades_df)

    # By outcome
    tp_count = (trades_df["outcome"] == "TP").sum()
    sl_count = (trades_df["outcome"] == "SL").sum()
    timeout_count = (trades_df["outcome"] == "timeout").sum()

    final_account = trades_df["account_after"].iloc[-1]
    total_return_pct = (final_account - starting_capital) / starting_capital * 100

    # Expectancy
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss) if n > 0 else 0

    return {
        "total_trades": n,
        "winners": n_win,
        "losers": n_loss,
        "breakeven": n_be,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "total_return_pct": total_return_pct,
        "final_account": final_account,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "avg_pnl": avg_pnl,
        "expectancy": expectancy,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "dd_start": dd_start,
        "dd_end": dd_end,
        "avg_bars_held": avg_bars,
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "tp_exits": tp_count,
        "sl_exits": sl_count,
        "timeout_exits": timeout_count,
    }


def breakdown_by_outcome(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Breakdown statistics per exit type (TP, SL, timeout)."""
    if trades_df.empty:
        return pd.DataFrame()

    rows = []
    for outcome in ["TP", "SL", "timeout"]:
        subset = trades_df[trades_df["outcome"] == outcome]
        n = len(subset)
        if n == 0:
            continue
        rows.append({
            "outcome": outcome,
            "trades": n,
            "total_pnl": subset["pnl_dollars"].sum(),
            "avg_pnl": subset["pnl_dollars"].mean(),
            "avg_score": subset["score"].mean(),
            "avg_bars": subset["bars_held"].mean(),
        })
    return pd.DataFrame(rows)


def breakdown_by_direction(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Breakdown statistics per direction (long, short)."""
    if trades_df.empty:
        return pd.DataFrame()

    rows = []
    for direction in ["buy", "sell"]:
        subset = trades_df[trades_df["direction"] == direction]
        n = len(subset)
        w = (subset["pnl_dollars"] > 0).sum()
        l = (subset["pnl_dollars"] < 0).sum()
        rows.append({
            "direction": direction,
            "trades": n,
            "winners": w,
            "losers": l,
            "win_rate": w / n * 100 if n > 0 else 0,
            "total_pnl": subset["pnl_dollars"].sum(),
            "avg_pnl": subset["pnl_dollars"].mean(),
            "avg_score": subset["score"].mean(),
        })
    return pd.DataFrame(rows)


def breakdown_by_score_range(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Breakdown by score ranges to see which scores perform best."""
    if trades_df.empty:
        return pd.DataFrame()

    bins = [0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.1]
    labels = ["0-5.0", "5.0-5.5", "5.5-6.0", "6.0-6.5", "6.5-7.0",
              "7.0-7.5", "7.5-8.0", "8.0-8.5", "8.5-9.0", "9.0-9.5", "9.5+"]
    trades_df = trades_df.copy()
    trades_df["score_range"] = pd.cut(trades_df["score"], bins=bins, labels=labels, right=False)

    rows = []
    for label in labels:
        subset = trades_df[trades_df["score_range"] == label]
        n = len(subset)
        if n == 0:
            continue
        w = (subset["pnl_dollars"] > 0).sum()
        rows.append({
            "score_range": label,
            "trades": n,
            "winners": w,
            "win_rate": w / n * 100,
            "total_pnl": subset["pnl_dollars"].sum(),
            "avg_pnl": subset["pnl_dollars"].mean(),
        })
    return pd.DataFrame(rows)


def top_trades(trades_df: pd.DataFrame, n: int = 10, worst: bool = False) -> pd.DataFrame:
    """Return top N best or worst trades."""
    if trades_df.empty:
        return pd.DataFrame()
    sorted_df = trades_df.sort_values("pnl_dollars", ascending=worst)
    cols = [
        "entry_time", "exit_time", "direction", "score",
        "entry_price", "exit_price", "pnl_dollars", "pnl_points",
        "outcome", "bars_held", "contracts",
    ]
    return sorted_df[cols].head(n).reset_index(drop=True)


def _build_equity_curve(trades_df: pd.DataFrame, starting_capital: float) -> pd.Series:
    """Build cumulative equity curve from trades."""
    equity = [starting_capital]
    for _, row in trades_df.iterrows():
        equity.append(equity[-1] + row["pnl_dollars"])
    return pd.Series(equity)


def _compute_max_drawdown(equity: pd.Series) -> tuple[float, float, int, int]:
    """Compute maximum drawdown in dollars and percent."""
    peak = equity.expanding().max()
    drawdown = equity - peak
    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()

    # Find the peak before this drawdown
    dd_start = equity[:max_dd_idx + 1].idxmax()
    dd_end = max_dd_idx

    peak_val = equity[dd_start]
    max_dd_pct = (max_dd / peak_val * 100) if peak_val > 0 else 0

    return max_dd, max_dd_pct, dd_start, dd_end


def _consecutive_streaks(trades_df: pd.DataFrame) -> tuple[int, int]:
    """Find maximum consecutive wins and losses."""
    wins = (trades_df["pnl_dollars"] > 0).astype(int).values
    max_w = 0
    max_l = 0
    curr_w = 0
    curr_l = 0

    for w in wins:
        if w == 1:
            curr_w += 1
            curr_l = 0
            max_w = max(max_w, curr_w)
        else:
            curr_l += 1
            curr_w = 0
            max_l = max(max_l, curr_l)

    return max_w, max_l
