"""Generate detailed monthly/yearly/trade data for surge x1.2 report."""

import sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import pandas_ta as ta
from collections import defaultdict

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


def simulate_tiered(sig_buy, sig_sell, highs, lows, closes, atrs, timestamps, scores_buy, scores_sell, n):
    """Simulate with exact webapp tiered logic + Smart DL. Returns detailed trades."""
    trades = []
    account = 100_000.0
    in_position = False
    daily_pnl = {}

    for i in range(n):
        if in_position:
            continue

        direction = None
        score = 0
        if sig_buy[i]:
            direction = "buy"
            score = scores_buy[i]
        elif sig_sell[i]:
            direction = "sell"
            score = scores_sell[i]
        if direction is None:
            continue

        atr = atrs[i]
        if np.isnan(atr) or atr <= 0:
            continue

        entry_price = closes[i]
        sl_distance = max(atr * SL_ATR_MULT, 10.0)

        if sl_distance <= TIER1_MAX_SL:
            tier, contracts = 1, 2
        elif sl_distance <= TIER2_MAX_SL:
            pot_loss = sl_distance * POINT_VALUE * 1
            if pot_loss > TIER2_MAX_LOSS:
                continue
            tier, contracts = 2, 1
        else:
            continue

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
            "pnl": round(total_pnl, 2),
            "outcome": outcome,
            "direction": direction,
            "tier": tier,
            "contracts": contracts,
            "sl_dist": round(sl_distance, 2),
            "entry": round(entry_price, 2),
            "exit": round(exit_price, 2),
            "score": round(score, 2),
            "bars": exit_bar - i,
            "time": str(timestamps[i]),
            "exit_time": str(timestamps[exit_bar]) if exit_bar < n else str(timestamps[-1]),
            "account": round(account, 2),
            "atr": round(atr, 2),
        })

    return trades, account


def load_from_sqlite(db_path, start_date=None, end_date=None):
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
    parser.add_argument("--db", type=str, required=True)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--output", type=str, default="reports/report_data.json")
    args = parser.parse_args()

    cfg = load_config("config/settings.yaml")
    ecfg = cfg.engines

    print(f"Loading from {args.db}...")
    df = load_from_sqlite(args.db, args.start, args.end)

    df = add_session_flags(df, tz=cfg.timezone,
        premarket_start_hour=ecfg.key_levels.premarket_start_hour,
        premarket_end_hour=ecfg.key_levels.premarket_end_hour,
        rth_start_hour=ecfg.vwap.rth_start_hour,
        rth_start_minute=ecfg.vwap.rth_start_minute)
    df = compute_mtf_trend(df, ecfg.mtf_trend, cfg.mtf_timeframes)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    vol = df["volume"].astype(float)
    vol_ma = vol.rolling(20, min_periods=1).mean()
    vol_ratio = np.where(vol_ma > 0, vol / vol_ma, 0.0)
    vol_score = np.select(
        [vol_ratio >= 2.0, vol_ratio >= 1.5, vol_ratio >= 1.2, vol_ratio >= 1.0],
        [10.0, 7.0, 5.0, 3.0], default=0.0)

    _compute_daily_hl(df)
    _compute_prev_day(df)
    _compute_premarket_hl(df)

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

    print(f"Data: {nn:,} bars | {df.index[0].date()} to {df.index[-1].date()}")

    # Run for x1.2
    mult = 1.2
    surge = pd.Series(vol_ratio >= mult, index=df.index)

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

    trades, final = simulate_tiered(sig_buy, sig_sell, highs, lows, closes, atrs, timestamps, buy_score, sell_score_arr, nn)

    print(f"Trades: {len(trades)}, Final account: ${final:,.0f}")

    # Build monthly data
    monthly = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0, "losses": 0, "gross_profit": 0, "gross_loss": 0})
    yearly = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0, "losses": 0,
                                   "gross_profit": 0, "gross_loss": 0,
                                   "max_dd": 0, "t1": 0, "t2": 0,
                                   "buy_trades": 0, "buy_wins": 0,
                                   "sell_trades": 0, "sell_wins": 0,
                                   "pnls": []})

    for t in trades:
        dt = pd.Timestamp(t["time"])
        ym = f"{dt.year}-{dt.month:02d}"
        y = str(dt.year)

        monthly[ym]["pnl"] += t["pnl"]
        monthly[ym]["trades"] += 1
        if t["pnl"] > 0:
            monthly[ym]["wins"] += 1
            monthly[ym]["gross_profit"] += t["pnl"]
        elif t["pnl"] < 0:
            monthly[ym]["losses"] += 1
            monthly[ym]["gross_loss"] += abs(t["pnl"])

        yearly[y]["pnl"] += t["pnl"]
        yearly[y]["trades"] += 1
        yearly[y]["pnls"].append(t["pnl"])
        if t["pnl"] > 0:
            yearly[y]["wins"] += 1
            yearly[y]["gross_profit"] += t["pnl"]
        elif t["pnl"] < 0:
            yearly[y]["losses"] += 1
            yearly[y]["gross_loss"] += abs(t["pnl"])
        if t["tier"] == 1:
            yearly[y]["t1"] += 1
        else:
            yearly[y]["t2"] += 1
        if t["direction"] == "buy":
            yearly[y]["buy_trades"] += 1
            if t["pnl"] > 0:
                yearly[y]["buy_wins"] += 1
        else:
            yearly[y]["sell_trades"] += 1
            if t["pnl"] > 0:
                yearly[y]["sell_wins"] += 1

    # Compute yearly max DD
    for y in yearly:
        pnls = yearly[y]["pnls"]
        equity = np.cumsum(np.insert(np.array(pnls), 0, 0))
        peak = np.maximum.accumulate(equity)
        dd = equity - peak
        yearly[y]["max_dd"] = round(float(dd.min()), 2)
        yearly[y]["pf"] = round(yearly[y]["gross_profit"] / yearly[y]["gross_loss"], 2) if yearly[y]["gross_loss"] > 0 else 999
        yearly[y]["wr"] = round(yearly[y]["wins"] / yearly[y]["trades"] * 100, 1) if yearly[y]["trades"] > 0 else 0
        yearly[y]["expectancy"] = round(yearly[y]["pnl"] / yearly[y]["trades"], 2) if yearly[y]["trades"] > 0 else 0
        yearly[y]["buy_wr"] = round(yearly[y]["buy_wins"] / yearly[y]["buy_trades"] * 100, 1) if yearly[y]["buy_trades"] > 0 else 0
        yearly[y]["sell_wr"] = round(yearly[y]["sell_wins"] / yearly[y]["sell_trades"] * 100, 1) if yearly[y]["sell_trades"] > 0 else 0
        del yearly[y]["pnls"]  # Don't serialize

    # Round monthly values
    for ym in monthly:
        monthly[ym]["pnl"] = round(monthly[ym]["pnl"], 2)
        monthly[ym]["gross_profit"] = round(monthly[ym]["gross_profit"], 2)
        monthly[ym]["gross_loss"] = round(monthly[ym]["gross_loss"], 2)
        monthly[ym]["wr"] = round(monthly[ym]["wins"] / monthly[ym]["trades"] * 100, 1) if monthly[ym]["trades"] > 0 else 0
        monthly[ym]["pf"] = round(monthly[ym]["gross_profit"] / monthly[ym]["gross_loss"], 2) if monthly[ym]["gross_loss"] > 0 else 999

    # Round yearly values
    for y in yearly:
        yearly[y]["pnl"] = round(yearly[y]["pnl"], 2)
        yearly[y]["gross_profit"] = round(yearly[y]["gross_profit"], 2)
        yearly[y]["gross_loss"] = round(yearly[y]["gross_loss"], 2)

    result = {
        "monthly": dict(monthly),
        "yearly": dict(yearly),
        "trades": trades,
    }

    with open(args.output, "w") as f:
        json.dump(result, f)

    print(f"Data saved to {args.output}")
    print(f"Monthly periods: {len(monthly)}, Yearly: {len(yearly)}, Trades: {len(trades)}")


if __name__ == "__main__":
    main()
