"""Trade simulator: walk forward through signal bars, simulate TP/SL exits.

Supports two modes:
  1. Simple: all contracts exit at TP/SL/timeout
  2. Runner: 2 contracts — Contract 1 exits at TP, Contract 2 becomes a runner
     with trailing stop (SL moves to breakeven, then trails)

For each signal bar:
  - Entry price = close of the signal bar
  - Stop loss distance = ATR * sl_atr_mult
  - Take profit distance = SL distance * RR ratio
  - Walk forward bar-by-bar until TP or SL hit
  - If both hit in same bar: SL wins (conservative)
  - Only one position at a time
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class TradeResult:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str       # "buy" or "sell"
    entry_price: float
    exit_price: float
    sl_price: float
    tp_price: float
    sl_distance: float
    contracts: int
    pnl_dollars: float
    pnl_points: float
    score: float
    outcome: str         # "TP", "SL", "timeout", "TP+runner", "TP+BE", "TP+trail"
    bars_held: int
    account_before: float
    account_after: float
    # Runner fields (None if no runner)
    runner_exit_price: float | None = None
    runner_pnl: float | None = None
    runner_outcome: str | None = None


def simulate_trades(
    df: pd.DataFrame,
    starting_capital: float = 100_000.0,
    point_value: float = 20.0,
    sl_atr_mult: float = 1.5,
    rr_ratio: float = 1.5,
    risk_pct: float = 0.005,
    max_bars_held: int = 60,
    max_contracts: int = 4,
    min_sl_points: float = 10.0,
    use_runner: bool = False,
    tp_contracts: int = 1,
    runner_contracts: int = 1,
    trail_atr_mult: float = 1.5,
    runner_max_bars: int = 120,
) -> list[TradeResult]:
    """Walk forward through the DataFrame and simulate every signal.

    Args:
        df: Full pipeline DataFrame with signal_buy, signal_sell, composite_score, atr.
        starting_capital: Starting account balance.
        point_value: Dollar value per point (NQ=$20, ES=$50, etc.).
        sl_atr_mult: ATR multiplier for stop loss distance.
        rr_ratio: Reward-to-risk ratio for take profit.
        risk_pct: Account risk per trade (0.005 = 0.5%).
        max_bars_held: Maximum bars to hold before forced exit (timeout).
        max_contracts: Maximum contracts per trade (prevents over-leveraging on low ATR).
        min_sl_points: Minimum SL distance in points (skips trades with tiny ATR).
        use_runner: Enable split-contract runner mode.
        tp_contracts: Number of contracts that close at TP.
        runner_contracts: Number of contracts that become runners.
        trail_atr_mult: ATR multiplier for trailing stop on the runner.
        runner_max_bars: Max bars for the runner after TP1 hit.

    Returns:
        List of TradeResult for every trade taken.
    """
    trades: list[TradeResult] = []
    account = starting_capital
    in_position = False

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    atrs = df["atr"].values
    signal_buy = df["signal_buy"].values
    signal_sell = df["signal_sell"].values
    scores = df["composite_score"].values
    timestamps = df.index
    n = len(df)

    for i in range(n):
        if in_position:
            continue

        direction = None
        if signal_buy[i]:
            direction = "buy"
        elif signal_sell[i]:
            direction = "sell"

        if direction is None:
            continue

        atr = atrs[i]
        if np.isnan(atr) or atr <= 0:
            continue

        entry_price = closes[i]
        sl_distance = max(atr * sl_atr_mult, min_sl_points)
        tp_distance = sl_distance * rr_ratio

        if direction == "buy":
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance

        risk_amount = account * risk_pct

        if use_runner:
            # Runner mode: tp_contracts + runner_contracts
            contracts = tp_contracts + runner_contracts
        else:
            contracts = max(1, int(risk_amount / (sl_distance * point_value)))
            contracts = min(contracts, max_contracts)

        # ===== Phase 1: Walk forward to TP or SL =====
        in_position = True
        exit_price = entry_price
        outcome = "timeout"
        exit_bar = i
        exit_time = timestamps[i]
        tp_hit = False

        for j in range(i + 1, min(i + 1 + max_bars_held, n)):
            bar_high = highs[j]
            bar_low = lows[j]

            if direction == "buy":
                if bar_low <= sl_price:
                    exit_price = sl_price
                    outcome = "SL"
                    exit_bar = j
                    exit_time = timestamps[j]
                    break
                elif bar_high >= tp_price:
                    exit_price = tp_price
                    outcome = "TP"
                    exit_bar = j
                    exit_time = timestamps[j]
                    tp_hit = True
                    break
            else:
                if bar_high >= sl_price:
                    exit_price = sl_price
                    outcome = "SL"
                    exit_bar = j
                    exit_time = timestamps[j]
                    break
                elif bar_low <= tp_price:
                    exit_price = tp_price
                    outcome = "TP"
                    exit_bar = j
                    exit_time = timestamps[j]
                    tp_hit = True
                    break
        else:
            last_bar = min(i + max_bars_held, n - 1)
            exit_price = closes[last_bar]
            exit_bar = last_bar
            exit_time = timestamps[last_bar]

        # ===== Compute P&L =====
        runner_exit_price = None
        runner_pnl = None
        runner_outcome = None

        if use_runner and tp_hit:
            # TP contracts: closed at TP
            if direction == "buy":
                c1_pnl_pts = tp_price - entry_price
            else:
                c1_pnl_pts = entry_price - tp_price
            c1_pnl = c1_pnl_pts * point_value * tp_contracts

            # Contract 2: Runner with trailing stop
            trail_dist = atr * trail_atr_mult
            runner_sl = entry_price  # Start at breakeven

            if direction == "buy":
                extreme = tp_price  # Best price seen
            else:
                extreme = tp_price

            runner_exit = entry_price  # Default: breakeven
            runner_outcome = "BE"
            runner_bar = exit_bar

            for k in range(exit_bar + 1, min(exit_bar + 1 + runner_max_bars, n)):
                bar_high = highs[k]
                bar_low = lows[k]

                if direction == "buy":
                    # Update extreme high
                    if bar_high > extreme:
                        extreme = bar_high
                        # Trail stop follows: extreme - trail_dist, but never below entry
                        runner_sl = max(entry_price, extreme - trail_dist)

                    # Check if trail stop hit
                    if bar_low <= runner_sl:
                        runner_exit = runner_sl
                        runner_outcome = "trail" if runner_sl > entry_price else "BE"
                        runner_bar = k
                        break
                else:
                    # Update extreme low
                    if bar_low < extreme:
                        extreme = bar_low
                        # Trail stop follows: extreme + trail_dist, but never above entry
                        runner_sl = min(entry_price, extreme + trail_dist)

                    # Check if trail stop hit
                    if bar_high >= runner_sl:
                        runner_exit = runner_sl
                        runner_outcome = "trail" if runner_sl < entry_price else "BE"
                        runner_bar = k
                        break
            else:
                # Runner timeout — close at market
                last_runner = min(exit_bar + runner_max_bars, n - 1)
                runner_exit = closes[last_runner]
                runner_outcome = "timeout"
                runner_bar = last_runner

            if direction == "buy":
                c2_pnl_pts = runner_exit - entry_price
            else:
                c2_pnl_pts = entry_price - runner_exit

            c2_pnl = c2_pnl_pts * point_value * runner_contracts
            runner_exit_price = runner_exit
            runner_pnl = c2_pnl

            total_pnl = c1_pnl + c2_pnl
            total_pnl_pts = c1_pnl_pts  # Report TP points as main
            outcome = "TP+" + runner_outcome
            exit_time = timestamps[runner_bar]
            bars_held = runner_bar - i

        elif use_runner and not tp_hit:
            # Both contracts hit SL or timeout together
            if direction == "buy":
                pnl_pts = exit_price - entry_price
            else:
                pnl_pts = entry_price - exit_price
            total_pnl = pnl_pts * point_value * contracts
            total_pnl_pts = pnl_pts
            bars_held = exit_bar - i

        else:
            # Simple mode (no runner)
            if direction == "buy":
                total_pnl_pts = exit_price - entry_price
            else:
                total_pnl_pts = entry_price - exit_price
            total_pnl = total_pnl_pts * point_value * contracts
            bars_held = exit_bar - i

        account_after = account + total_pnl

        trades.append(TradeResult(
            entry_time=timestamps[i],
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            sl_price=sl_price,
            tp_price=tp_price,
            sl_distance=sl_distance,
            contracts=contracts,
            pnl_dollars=total_pnl,
            pnl_points=total_pnl_pts,
            score=scores[i],
            outcome=outcome,
            bars_held=bars_held,
            account_before=account,
            account_after=account_after,
            runner_exit_price=runner_exit_price,
            runner_pnl=runner_pnl,
            runner_outcome=runner_outcome,
        ))

        account = account_after
        in_position = False

    return trades


def trades_to_dataframe(trades: list[TradeResult]) -> pd.DataFrame:
    """Convert trade results to a DataFrame for analysis."""
    if not trades:
        return pd.DataFrame()

    records = []
    for t in trades:
        records.append({
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "sl_price": t.sl_price,
            "tp_price": t.tp_price,
            "sl_distance": t.sl_distance,
            "contracts": t.contracts,
            "pnl_dollars": t.pnl_dollars,
            "pnl_points": t.pnl_points,
            "score": t.score,
            "outcome": t.outcome,
            "bars_held": t.bars_held,
            "account_before": t.account_before,
            "account_after": t.account_after,
            "runner_exit_price": t.runner_exit_price,
            "runner_pnl": t.runner_pnl,
            "runner_outcome": t.runner_outcome,
        })
    return pd.DataFrame(records)
