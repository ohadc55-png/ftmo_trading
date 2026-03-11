"""Golden Days Analysis for surge x1.2 (2020-2026)."""

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
            if eligible[i]: state = 1; signals[i] = True
        elif state == 1:
            if scores[i] < threshold: state = 2
        elif state == 2:
            if eligible[i]: state = 1; signals[i] = True
    return signals


def simulate_tiered(sig_buy, sig_sell, highs, lows, closes, atrs, timestamps, scores_buy, scores_sell, n):
    trades = []
    account = 100_000.0
    in_position = False
    daily_pnl = {}

    for i in range(n):
        if in_position: continue
        direction = None; score = 0
        if sig_buy[i]: direction = "buy"; score = scores_buy[i]
        elif sig_sell[i]: direction = "sell"; score = scores_sell[i]
        if direction is None: continue

        atr = atrs[i]
        if np.isnan(atr) or atr <= 0: continue

        entry_price = closes[i]
        sl_distance = max(atr * SL_ATR_MULT, 10.0)

        if sl_distance <= TIER1_MAX_SL: tier, contracts = 1, 2
        elif sl_distance <= TIER2_MAX_SL:
            if sl_distance * POINT_VALUE > TIER2_MAX_LOSS: continue
            tier, contracts = 2, 1
        else: continue

        day_key = str(timestamps[i].date())
        day_pnl_val = daily_pnl.get(day_key, 0)
        pot_loss = sl_distance * POINT_VALUE * contracts
        remaining = SMART_DL + day_pnl_val
        if day_pnl_val <= -SMART_DL or pot_loss > remaining: continue

        tp_distance = sl_distance * RR_RATIO
        if direction == "buy":
            sl_price = entry_price - sl_distance; tp_price = entry_price + tp_distance
        else:
            sl_price = entry_price + sl_distance; tp_price = entry_price - tp_distance

        in_position = True; exit_price = entry_price; outcome = "timeout"; exit_bar = i; tp_hit = False

        for j in range(i + 1, min(i + 1 + MAX_BARS_HELD, n)):
            if direction == "buy":
                if lows[j] <= sl_price: exit_price, outcome, exit_bar = sl_price, "SL", j; break
                elif highs[j] >= tp_price: exit_price, outcome, exit_bar, tp_hit = tp_price, "TP", j, True; break
            else:
                if highs[j] >= sl_price: exit_price, outcome, exit_bar = sl_price, "SL", j; break
                elif lows[j] <= tp_price: exit_price, outcome, exit_bar, tp_hit = tp_price, "TP", j, True; break
        else:
            last = min(i + MAX_BARS_HELD, n - 1); exit_price, exit_bar = closes[last], last

        if tp_hit and tier == 1:
            c1_pts = (tp_price - entry_price) if direction == "buy" else (entry_price - tp_price)
            c1_pnl = c1_pts * POINT_VALUE
            trail_dist = atr * TRAIL_ATR_MULT; runner_sl = entry_price; extreme = tp_price
            runner_exit = entry_price; r_outcome = "BE"
            for k in range(exit_bar + 1, min(exit_bar + 1 + RUNNER_MAX_BARS, n)):
                if direction == "buy":
                    if highs[k] > extreme: extreme = highs[k]; runner_sl = max(entry_price, extreme - trail_dist)
                    if lows[k] <= runner_sl: runner_exit = runner_sl; r_outcome = "trail" if runner_sl > entry_price else "BE"; break
                else:
                    if lows[k] < extreme: extreme = lows[k]; runner_sl = min(entry_price, extreme + trail_dist)
                    if highs[k] >= runner_sl: runner_exit = runner_sl; r_outcome = "trail" if runner_sl < entry_price else "BE"; break
            else:
                last_r = min(exit_bar + RUNNER_MAX_BARS, n - 1); runner_exit = closes[last_r]; r_outcome = "timeout"
            c2_pnl = ((runner_exit - entry_price) if direction == "buy" else (entry_price - runner_exit)) * POINT_VALUE
            total_pnl = c1_pnl + c2_pnl; outcome = f"TP+{r_outcome}"
        elif tp_hit and tier == 2:
            total_pnl = ((tp_price - entry_price) if direction == "buy" else (entry_price - tp_price)) * POINT_VALUE
            outcome = "TP"
        else:
            total_pnl = ((exit_price - entry_price) if direction == "buy" else (entry_price - exit_price)) * POINT_VALUE * contracts

        account += total_pnl
        daily_pnl[day_key] = daily_pnl.get(day_key, 0) + total_pnl
        in_position = False

        trades.append({
            "pnl": round(total_pnl, 2),
            "outcome": outcome,
            "direction": direction,
            "tier": tier,
            "day": day_key,
            "time": str(timestamps[i]),
            "score": round(score, 2),
        })

    return trades, account, daily_pnl


def load_from_sqlite(db_path, start_date=None, end_date=None):
    import sqlite3
    conn = sqlite3.connect(db_path)
    query = "SELECT datetime, open, high, low, close, volume FROM ohlcv_1m"
    conditions = []
    if start_date: conditions.append(f"datetime >= '{start_date}'")
    if end_date: conditions.append(f"datetime <= '{end_date}'")
    if conditions: query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY datetime"
    df = pd.read_sql(query, conn, parse_dates=["datetime"]); conn.close()
    df = df.set_index("datetime")
    if df.index.tz is None: df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("America/New_York")
    df = df.resample("5min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0)
    return df


def run_for_mult(df, cfg, mult):
    ecfg = cfg.engines
    df2 = df.copy()
    df2 = add_session_flags(df2, tz=cfg.timezone,
        premarket_start_hour=ecfg.key_levels.premarket_start_hour,
        premarket_end_hour=ecfg.key_levels.premarket_end_hour,
        rth_start_hour=ecfg.vwap.rth_start_hour,
        rth_start_minute=ecfg.vwap.rth_start_minute)
    df2 = compute_mtf_trend(df2, ecfg.mtf_trend, cfg.mtf_timeframes)
    df2["atr"] = ta.atr(df2["high"], df2["low"], df2["close"], length=14)

    vol = df2["volume"].astype(float)
    vol_ma = vol.rolling(20, min_periods=1).mean()
    vol_ratio = np.where(vol_ma > 0, vol / vol_ma, 0.0)
    vol_score = np.select([vol_ratio >= 2.0, vol_ratio >= 1.5, vol_ratio >= 1.2, vol_ratio >= 1.0],
                          [10.0, 7.0, 5.0, 3.0], default=0.0)

    _compute_daily_hl(df2); _compute_prev_day(df2); _compute_premarket_hl(df2)

    mtf_arr = df2["mtf_score"].fillna(0).values
    direction_arr = df2["mtf_direction"].fillna("neutral").values
    bull_mask = direction_arr == "bull"; bear_mask = direction_arr == "bear"
    hour_mask = np.ones(len(df2), dtype=bool)
    for hh in EXCLUDE_HOURS: hour_mask &= df2.index.hour.values != hh

    close_s = df2["close"]; buf = ecfg.key_levels.breakout_buffer_points
    surge = pd.Series(vol_ratio >= mult, index=df2.index)

    brk_ph = (close_s > df2["prev_day_high"] + buf) & surge
    brk_pl = (close_s < df2["prev_day_low"] - buf) & surge
    brk_dh = (close_s > df2["daily_high"].shift(1) + buf) & surge
    brk_dl = (close_s < df2["daily_low"].shift(1) - buf) & surge
    brk_pmh = (close_s > df2["premarket_high"] + buf) & surge
    brk_pml = (close_s < df2["premarket_low"] - buf) & surge
    ret_ph = (brk_ph.shift(1).fillna(0).rolling(10, min_periods=1).max().astype(bool)
              & (abs(close_s - df2["prev_day_high"]) <= buf * 2))
    ret_pl = (brk_pl.shift(1).fillna(0).rolling(10, min_periods=1).max().astype(bool)
              & (abs(close_s - df2["prev_day_low"]) <= buf * 2))
    brk = (brk_ph.astype(float)*3 + brk_pl.astype(float)*3 + brk_dh.astype(float)*2 + brk_dl.astype(float)*2 +
           brk_pmh.astype(float)*2 + brk_pml.astype(float)*2 + ret_ph.astype(float)*2 + ret_pl.astype(float)*2).clip(0,10).values

    buy_score = np.clip((mtf_arr*3 + vol_score*1 + brk*4) / BUY_WEIGHTS.sum(), 0, 10)
    sell_score_arr = np.clip((mtf_arr*3 + vol_score*2 + brk*3) / SELL_WEIGHTS.sum(), 0, 10)
    buy_elig = (buy_score >= THRESHOLD) & bull_mask & hour_mask
    sell_elig = (sell_score_arr >= THRESHOLD) & bear_mask & hour_mask
    sig_buy = cooldown(buy_elig, buy_score, THRESHOLD)
    sig_sell = cooldown(sell_elig, sell_score_arr, THRESHOLD)

    return simulate_tiered(sig_buy, sig_sell, df2["high"].values, df2["low"].values,
                           df2["close"].values, df2["atr"].values, df2.index,
                           buy_score, sell_score_arr, len(df2))


def main():
    db = "nq_data.db"
    cfg = load_config("config/settings.yaml")

    print("Loading data 2020-01 to 2026-03...")
    df = load_from_sqlite(db, "2020-01-01", "2026-03-31")
    print(f"Bars: {len(df):,} | {df.index[0].date()} to {df.index[-1].date()}\n")

    for mult in [1.2, 1.5]:
        print(f"\n{'=' * 80}")
        print(f"  SURGE x{mult} -- Golden Days Analysis (2020 - Mar 2026)")
        print(f"{'=' * 80}")

        trades, final, daily_pnl = run_for_mult(df, cfg, mult)
        pnls = [t["pnl"] for t in trades]
        total_pnl = sum(pnls)
        winners = [p for p in pnls if p > 0]

        print(f"  Total trades: {len(trades)}")
        print(f"  Total P&L: ${total_pnl:+,.0f}")
        print(f"  Win rate: {len(winners)/len(trades)*100:.1f}%")

        # Daily P&L analysis
        daily_items = sorted(daily_pnl.items())
        daily_vals = [v for _, v in daily_items]

        # Golden days: $10K+
        golden = [(d, p) for d, p in daily_items if p >= 10000]
        big_days_5k = [(d, p) for d, p in daily_items if p >= 5000]
        big_days_3k = [(d, p) for d, p in daily_items if p >= 3000]
        red_days = [(d, p) for d, p in daily_items if p <= -1000]

        golden_total = sum(p for _, p in golden)
        big5k_total = sum(p for _, p in big_days_5k)

        print(f"\n  Trading days total: {len(daily_items)}")
        print(f"  Profitable days: {sum(1 for v in daily_vals if v > 0)} ({sum(1 for v in daily_vals if v > 0)/len(daily_vals)*100:.0f}%)")
        print(f"  Losing days: {sum(1 for v in daily_vals if v < 0)} ({sum(1 for v in daily_vals if v < 0)/len(daily_vals)*100:.0f}%)")
        print(f"  Avg daily P&L: ${np.mean(daily_vals):+,.0f}")

        print(f"\n  --- GOLDEN DAYS ($10K+) ---")
        print(f"  Count: {len(golden)} days out of {len(daily_items)} ({len(golden)/len(daily_items)*100:.1f}%)")
        print(f"  Golden days P&L: ${golden_total:+,.0f}")
        print(f"  % of total profit: {golden_total/total_pnl*100:.1f}%")
        print(f"  Avg golden day: ${golden_total/len(golden):+,.0f}" if golden else "")

        print(f"\n  --- BIG DAYS ($5K+) ---")
        print(f"  Count: {len(big_days_5k)} days ({len(big_days_5k)/len(daily_items)*100:.1f}%)")
        print(f"  Big days P&L: ${big5k_total:+,.0f}")
        print(f"  % of total profit: {big5k_total/total_pnl*100:.1f}%")

        print(f"\n  --- DAYS $3K+ ---")
        print(f"  Count: {len(big_days_3k)} days ({len(big_days_3k)/len(daily_items)*100:.1f}%)")

        print(f"\n  --- RED DAYS ($-1K or worse) ---")
        print(f"  Count: {len(red_days)} days")
        print(f"  Red days P&L: ${sum(p for _, p in red_days):+,.0f}")

        # Golden days per year
        print(f"\n  --- GOLDEN DAYS ($10K+) PER YEAR ---")
        yearly_golden = defaultdict(list)
        yearly_pnl = defaultdict(float)
        for d, p in daily_items:
            y = d[:4]
            yearly_pnl[y] += p
        for d, p in golden:
            y = d[:4]
            yearly_golden[y].append((d, p))

        print(f"  {'Year':>6} | {'Golden Days':>11} | {'Golden P&L':>12} | {'Year P&L':>12} | {'% from Golden':>13}")
        print(f"  {'-'*65}")
        for y in sorted(set(d[:4] for d, _ in daily_items)):
            gdays = yearly_golden.get(y, [])
            gpnl = sum(p for _, p in gdays)
            ypnl = yearly_pnl[y]
            pct = gpnl / ypnl * 100 if ypnl > 0 else 0
            print(f"  {y:>6} | {len(gdays):>11} | ${gpnl:>+11,.0f} | ${ypnl:>+11,.0f} | {pct:>12.1f}%")
        total_golden_pnl = sum(p for _, p in golden)
        print(f"  {'-'*65}")
        print(f"  {'TOTAL':>6} | {len(golden):>11} | ${total_golden_pnl:>+11,.0f} | ${total_pnl:>+11,.0f} | {total_golden_pnl/total_pnl*100:>12.1f}%")

        # List all golden days
        print(f"\n  --- ALL GOLDEN DAYS (${10}K+) ---")
        print(f"  {'Date':>12} | {'P&L':>10} | {'Day':>5}")
        print(f"  {'-'*35}")
        for d, p in sorted(golden, key=lambda x: x[1], reverse=True):
            dow = pd.Timestamp(d).day_name()[:3]
            print(f"  {d:>12} | ${p:>+9,.0f} | {dow}")

        # Concentration analysis
        print(f"\n  --- PROFIT CONCENTRATION ---")
        sorted_days = sorted(daily_items, key=lambda x: x[1], reverse=True)
        for top_n in [5, 10, 20, 30, 50]:
            top_pnl = sum(p for _, p in sorted_days[:top_n])
            print(f"  Top {top_n:>2} days: ${top_pnl:>+11,.0f} ({top_pnl/total_pnl*100:>5.1f}% of total)")

        # What happens without golden days?
        non_golden_pnl = total_pnl - golden_total
        print(f"\n  WITHOUT golden days: ${non_golden_pnl:+,.0f}")
        print(f"  Strategy still profitable without golden days? {'YES' if non_golden_pnl > 0 else 'NO'}")


if __name__ == "__main__":
    main()
