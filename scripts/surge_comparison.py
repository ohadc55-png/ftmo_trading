"""Compare surge_mult=1.5 vs 1.3 vs 1.2 using the webapp's exact tiered logic."""

import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from collections import Counter

from src.utils.config import load_config
from src.utils.timeutils import add_session_flags
from src.engines.mtf_trend import compute_mtf_trend
from src.engines.key_levels import _compute_daily_hl, _compute_prev_day, _compute_premarket_hl

# ── Webapp parameters (exact copy) ──
RR_RATIO = 5.0
TIER1_MAX_SL = 25
TIER2_MAX_SL = 50
TIER2_MAX_LOSS = 1000.0
THRESHOLD = 5.0
SMART_DL = 1100
POINT_VALUE = 20.0
EXCLUDE_HOURS = [0, 1, 2, 3, 4, 5, 6]
SELL_WEIGHTS = np.array([3, 2, 3], dtype=float)
BUY_WEIGHTS = np.array([3, 1, 4], dtype=float)
MAX_BARS_HELD = 60
RUNNER_MAX_BARS = 120
SL_ATR_MULT = 1.5
TRAIL_ATR_MULT = 1.5


def cooldown(eligible, scores, threshold):
    n = len(eligible)
    signals = np.zeros(n, dtype=bool)
    state = 0
    for i in range(n):
        if state == 0:
            if eligible[i]:
                state = 1
                signals[i] = True
        elif state == 1:
            if scores[i] < threshold:
                state = 2
        elif state == 2:
            if eligible[i]:
                state = 1
                signals[i] = True
    return signals


def simulate_tiered(sig_buy, sig_sell, highs, lows, closes, atrs, timestamps, n):
    """Simulate with exact webapp tiered logic + Smart DL."""
    trades = []
    account = 100_000.0
    in_position = False
    daily_pnl = {}

    for i in range(n):
        if in_position:
            continue

        direction = None
        if sig_buy[i]:
            direction = "buy"
        elif sig_sell[i]:
            direction = "sell"
        if direction is None:
            continue

        atr = atrs[i]
        if np.isnan(atr) or atr <= 0:
            continue

        entry_price = closes[i]
        sl_distance = max(atr * SL_ATR_MULT, 10.0)

        # Tier logic
        if sl_distance <= TIER1_MAX_SL:
            tier, contracts = 1, 2
        elif sl_distance <= TIER2_MAX_SL:
            pot_loss = sl_distance * POINT_VALUE * 1
            if pot_loss > TIER2_MAX_LOSS:
                continue
            tier, contracts = 2, 1
        else:
            continue

        # Smart DL
        day_key = str(timestamps[i].date())
        day_pnl = daily_pnl.get(day_key, 0)
        pot_loss = sl_distance * POINT_VALUE * contracts
        remaining = SMART_DL + day_pnl
        if day_pnl <= -SMART_DL or pot_loss > remaining:
            continue

        tp_distance = sl_distance * RR_RATIO
        if direction == "buy":
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance

        # Walk forward
        in_position = True
        exit_price = entry_price
        outcome = "timeout"
        exit_bar = i
        tp_hit = False

        for j in range(i + 1, min(i + 1 + MAX_BARS_HELD, n)):
            if direction == "buy":
                if lows[j] <= sl_price:
                    exit_price, outcome, exit_bar = sl_price, "SL", j
                    break
                elif highs[j] >= tp_price:
                    exit_price, outcome, exit_bar, tp_hit = tp_price, "TP", j, True
                    break
            else:
                if highs[j] >= sl_price:
                    exit_price, outcome, exit_bar = sl_price, "SL", j
                    break
                elif lows[j] <= tp_price:
                    exit_price, outcome, exit_bar, tp_hit = tp_price, "TP", j, True
                    break
        else:
            last = min(i + MAX_BARS_HELD, n - 1)
            exit_price, exit_bar = closes[last], last

        # P&L
        runner_pnl_val = 0.0
        if tp_hit and tier == 1:
            c1_pts = (tp_price - entry_price) if direction == "buy" else (entry_price - tp_price)
            c1_pnl = c1_pts * POINT_VALUE

            trail_dist = atr * TRAIL_ATR_MULT
            runner_sl = entry_price
            extreme = tp_price
            runner_exit = entry_price
            r_outcome = "BE"

            for k in range(exit_bar + 1, min(exit_bar + 1 + RUNNER_MAX_BARS, n)):
                if direction == "buy":
                    if highs[k] > extreme:
                        extreme = highs[k]
                        runner_sl = max(entry_price, extreme - trail_dist)
                    if lows[k] <= runner_sl:
                        runner_exit = runner_sl
                        r_outcome = "trail" if runner_sl > entry_price else "BE"
                        break
                else:
                    if lows[k] < extreme:
                        extreme = lows[k]
                        runner_sl = min(entry_price, extreme + trail_dist)
                    if highs[k] >= runner_sl:
                        runner_exit = runner_sl
                        r_outcome = "trail" if runner_sl < entry_price else "BE"
                        break
            else:
                last_r = min(exit_bar + RUNNER_MAX_BARS, n - 1)
                runner_exit = closes[last_r]
                r_outcome = "timeout"

            c2_pts = (runner_exit - entry_price) if direction == "buy" else (entry_price - runner_exit)
            c2_pnl = c2_pts * POINT_VALUE
            total_pnl = c1_pnl + c2_pnl
            runner_pnl_val = c2_pnl
            outcome = f"TP+{r_outcome}"

        elif tp_hit and tier == 2:
            pts = (tp_price - entry_price) if direction == "buy" else (entry_price - tp_price)
            total_pnl = pts * POINT_VALUE * 1
            outcome = "TP"

        else:
            pts = (exit_price - entry_price) if direction == "buy" else (entry_price - exit_price)
            total_pnl = pts * POINT_VALUE * contracts

        account += total_pnl
        daily_pnl[day_key] = daily_pnl.get(day_key, 0) + total_pnl
        in_position = False

        trades.append({
            "pnl": total_pnl,
            "outcome": outcome,
            "direction": direction,
            "tier": tier,
            "contracts": contracts,
            "sl_dist": sl_distance,
            "entry": entry_price,
            "bars": exit_bar - i,
            "time": str(timestamps[i]),
        })

    return trades, account


def load_from_sqlite(db_path, start_date=None, end_date=None):
    """Load 1m data from SQLite, resample to 5m."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    query = "SELECT datetime, open, high, low, close, volume FROM ohlcv_1m"
    conditions = []
    if start_date:
        conditions.append(f"datetime >= '{start_date}'")
    if end_date:
        conditions.append(f"datetime <= '{end_date}'")
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY datetime"

    df = pd.read_sql(query, conn, parse_dates=["datetime"])
    conn.close()

    df = df.set_index("datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("America/New_York")

    # Resample 1m -> 5m
    df = df.resample("5min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0)
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default=None, help="SQLite DB path")
    parser.add_argument("--start", type=str, default=None, help="Start date (e.g. 2024-01-01)")
    parser.add_argument("--end", type=str, default=None, help="End date (e.g. 2025-12-31)")
    args = parser.parse_args()

    cfg = load_config("config/settings.yaml")
    ecfg = cfg.engines

    if args.db:
        print(f"Loading from {args.db} ({args.start} to {args.end})...")
        df = load_from_sqlite(args.db, args.start, args.end)
    else:
        # Fetch data from Yahoo Finance
        print("Fetching 60d of NQ=F 5m data...")
        ticker = yf.Ticker("NQ=F")
        df = ticker.history(period="60d", interval="5m")
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        df = df[["open", "high", "low", "close", "volume"]].copy()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("America/New_York")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"])
        df["volume"] = df["volume"].fillna(0)

    # Pipeline (computed once)
    df = add_session_flags(df, tz=cfg.timezone,
        premarket_start_hour=ecfg.key_levels.premarket_start_hour,
        premarket_end_hour=ecfg.key_levels.premarket_end_hour,
        rth_start_hour=ecfg.vwap.rth_start_hour,
        rth_start_minute=ecfg.vwap.rth_start_minute)
    df = compute_mtf_trend(df, ecfg.mtf_trend, cfg.mtf_timeframes)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # Volume base (same for all)
    vol = df["volume"].astype(float)
    vol_ma = vol.rolling(20, min_periods=1).mean()
    vol_ratio = np.where(vol_ma > 0, vol / vol_ma, 0.0)
    vol_score = np.select(
        [vol_ratio >= 2.0, vol_ratio >= 1.5, vol_ratio >= 1.2, vol_ratio >= 1.0],
        [10.0, 7.0, 5.0, 3.0], default=0.0)

    # Key levels (computed once)
    _compute_daily_hl(df)
    _compute_prev_day(df)
    _compute_premarket_hl(df)

    # Common arrays
    mtf_arr = df["mtf_score"].fillna(0).values
    direction_arr = df["mtf_direction"].fillna("neutral").values
    bull_mask = direction_arr == "bull"
    bear_mask = direction_arr == "bear"
    hour_arr = df.index.hour.values
    hour_mask = np.ones(len(df), dtype=bool)
    for h in EXCLUDE_HOURS:
        hour_mask &= hour_arr != h

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    atrs = df["atr"].values
    timestamps = df.index
    close_s = df["close"]
    buf = ecfg.key_levels.breakout_buffer_points
    nn = len(df)

    print(f"Data: {nn:,} bars | {df.index[0].date()} to {df.index[-1].date()}\n")

    # ═══════════════════════════════════════
    # Summary table
    # ═══════════════════════════════════════
    print("=" * 110)
    print(f"{'Surge':>7} | {'Trades':>7} | {'Win%':>6} | {'Total P&L':>12} | "
          f"{'PF':>6} | {'MaxDD':>10} | {'DD%':>7} | "
          f"{'AvgWin':>9} | {'AvgLoss':>9} | {'Expect':>9} | "
          f"{'T1':>4} | {'T2':>4}")
    print("-" * 110)

    all_results = {}

    for mult in [1.5, 1.3, 1.2]:
        surge = pd.Series(vol_ratio >= mult, index=df.index)

        # BRK scores
        brk_ph = (close_s > df["prev_day_high"] + buf) & surge
        brk_pl = (close_s < df["prev_day_low"] - buf) & surge
        brk_dh = (close_s > df["daily_high"].shift(1) + buf) & surge
        brk_dl = (close_s < df["daily_low"].shift(1) - buf) & surge
        brk_pmh = (close_s > df["premarket_high"] + buf) & surge
        brk_pml = (close_s < df["premarket_low"] - buf) & surge
        ret_ph = (brk_ph.shift(1).fillna(0).rolling(10, min_periods=1).max().astype(bool)
                  & (abs(close_s - df["prev_day_high"]) <= buf * 2))
        ret_pl = (brk_pl.shift(1).fillna(0).rolling(10, min_periods=1).max().astype(bool)
                  & (abs(close_s - df["prev_day_low"]) <= buf * 2))

        brk = (brk_ph.astype(float) * 3 + brk_pl.astype(float) * 3 +
               brk_dh.astype(float) * 2 + brk_dl.astype(float) * 2 +
               brk_pmh.astype(float) * 2 + brk_pml.astype(float) * 2 +
               ret_ph.astype(float) * 2 + ret_pl.astype(float) * 2).clip(0, 10).values

        buy_score = np.clip((mtf_arr * 3 + vol_score * 1 + brk * 4) / BUY_WEIGHTS.sum(), 0, 10)
        sell_score_arr = np.clip((mtf_arr * 3 + vol_score * 2 + brk * 3) / SELL_WEIGHTS.sum(), 0, 10)

        buy_elig = (buy_score >= THRESHOLD) & bull_mask & hour_mask
        sell_elig = (sell_score_arr >= THRESHOLD) & bear_mask & hour_mask

        sig_buy = cooldown(buy_elig, buy_score, THRESHOLD)
        sig_sell = cooldown(sell_elig, sell_score_arr, THRESHOLD)

        trades, final = simulate_tiered(sig_buy, sig_sell, highs, lows, closes, atrs, timestamps, nn)
        all_results[mult] = trades

        if not trades:
            print(f"  x{mult:.1f} | NO TRADES")
            continue

        pnls = np.array([t["pnl"] for t in trades])
        winners = pnls[pnls > 0]
        losers = pnls[pnls < 0]
        nt = len(trades)
        nw = len(winners)
        wr = nw / nt * 100
        tot = pnls.sum()
        gp = winners.sum() if len(winners) else 0
        gl = abs(losers.sum()) if len(losers) else 0
        pf = gp / gl if gl > 0 else float("inf")
        aw = winners.mean() if len(winners) else 0
        al = losers.mean() if len(losers) else 0
        exp = (wr / 100 * aw) + ((1 - wr / 100) * al)

        equity = np.cumsum(np.insert(pnls, 0, 100000))
        peak = np.maximum.accumulate(equity)
        dd = equity - peak
        max_dd = dd.min()
        max_dd_pct = (max_dd / peak[np.argmin(dd)]) * 100

        t1 = sum(1 for t in trades if t["tier"] == 1)
        t2 = sum(1 for t in trades if t["tier"] == 2)

        print(f"  x{mult:.1f} | {nt:>7} | {wr:>5.1f}% | ${tot:>+11,.0f} | "
              f"{pf:>6.2f} | ${max_dd:>+9,.0f} | {max_dd_pct:>6.1f}% | "
              f"${aw:>+8,.0f} | ${al:>+8,.0f} | ${exp:>+8,.0f} | "
              f"{t1:>4} | {t2:>4}")

    print("=" * 110)

    # ═══════════════════════════════════════
    # Detailed breakdown per config
    # ═══════════════════════════════════════
    for mult in [1.5, 1.3, 1.2]:
        trades = all_results.get(mult, [])
        if not trades:
            continue

        pnls = [t["pnl"] for t in trades]

        print(f"\n{'-' * 60}")
        print(f"  SURGE x{mult:.1f} -- Detailed Breakdown")
        print(f"{'-' * 60}")

        # By outcome
        oc = Counter(t["outcome"] for t in trades)
        print("  By outcome:")
        for k in sorted(oc.keys()):
            sub = [t["pnl"] for t in trades if t["outcome"] == k]
            sub_w = sum(1 for p in sub if p > 0)
            print(f"    {k:>12}: {oc[k]:>3} trades | P&L ${sum(sub):>+10,.0f} | "
                  f"WR {sub_w/len(sub)*100:>5.1f}% | Avg ${np.mean(sub):>+8,.0f}")

        # By direction
        print("  By direction:")
        for d in ["buy", "sell"]:
            sub = [t for t in trades if t["direction"] == d]
            if not sub:
                continue
            sp = [t["pnl"] for t in sub]
            sw = sum(1 for p in sp if p > 0)
            print(f"    {d.upper():>5}: {len(sub):>3} trades | P&L ${sum(sp):>+10,.0f} | "
                  f"WR {sw/len(sub)*100:.1f}%")

        # Avg/Median trade
        print(f"  Avg trade:    ${np.mean(pnls):>+,.0f}")
        print(f"  Median trade: ${np.median(pnls):>+,.0f}")

        # Consecutive streaks
        streak_w = streak_l = max_w = max_l = 0
        for p in pnls:
            if p > 0:
                streak_w += 1
                streak_l = 0
                max_w = max(max_w, streak_w)
            else:
                streak_l += 1
                streak_w = 0
                max_l = max(max_l, streak_l)
        print(f"  Max consec wins: {max_w} | Max consec losses: {max_l}")


if __name__ == "__main__":
    main()
