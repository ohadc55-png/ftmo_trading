"""Walk-Forward Validation: Train 2016-2022, Test 2023-2025 for surge x1.2 vs x1.5."""

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
            if eligible[i]:
                state = 1; signals[i] = True
        elif state == 1:
            if scores[i] < threshold: state = 2
        elif state == 2:
            if eligible[i]:
                state = 1; signals[i] = True
    return signals


def simulate_tiered(sig_buy, sig_sell, highs, lows, closes, atrs, timestamps, scores_buy, scores_sell, n):
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
            direction = "buy"; score = scores_buy[i]
        elif sig_sell[i]:
            direction = "sell"; score = scores_sell[i]
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
            if sl_distance * POINT_VALUE > TIER2_MAX_LOSS:
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
                    exit_price, outcome, exit_bar = sl_price, "SL", j; break
                elif highs[j] >= tp_price:
                    exit_price, outcome, exit_bar, tp_hit = tp_price, "TP", j, True; break
            else:
                if highs[j] >= sl_price:
                    exit_price, outcome, exit_bar = sl_price, "SL", j; break
                elif lows[j] <= tp_price:
                    exit_price, outcome, exit_bar, tp_hit = tp_price, "TP", j, True; break
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
                        r_outcome = "trail" if runner_sl > entry_price else "BE"; break
                else:
                    if lows[k] < extreme:
                        extreme = lows[k]
                        runner_sl = min(entry_price, extreme + trail_dist)
                    if highs[k] >= runner_sl:
                        runner_exit = runner_sl
                        r_outcome = "trail" if runner_sl < entry_price else "BE"; break
            else:
                last_r = min(exit_bar + RUNNER_MAX_BARS, n - 1)
                runner_exit = closes[last_r]; r_outcome = "timeout"
            c2_pts = (runner_exit - entry_price) if direction == "buy" else (entry_price - runner_exit)
            c2_pnl = c2_pts * POINT_VALUE
            total_pnl = c1_pnl + c2_pnl
            outcome = f"TP+{r_outcome}"
        elif tp_hit and tier == 2:
            pts = (tp_price - entry_price) if direction == "buy" else (entry_price - tp_price)
            total_pnl = pts * POINT_VALUE
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
            "time": str(timestamps[i]),
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


def run_backtest_for_period(db_path, start, end, mult):
    """Run full backtest for a given period and surge mult."""
    cfg = load_config("config/settings.yaml")
    ecfg = cfg.engines

    df = load_from_sqlite(db_path, start, end)
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
    for hh in EXCLUDE_HOURS:
        hour_mask &= hour_arr != hh

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    atrs = df["atr"].values
    timestamps = df.index
    close_s = df["close"]
    buf = ecfg.key_levels.breakout_buffer_points
    nn = len(df)

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

    # Compute stats
    if not trades:
        return {"trades": 0}

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

    # Yearly breakdown
    yearly = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})
    for t in trades:
        y = t["time"][:4]
        yearly[y]["pnl"] += t["pnl"]
        yearly[y]["trades"] += 1
        if t["pnl"] > 0:
            yearly[y]["wins"] += 1

    yearly_list = []
    for y in sorted(yearly.keys()):
        v = yearly[y]
        yearly_list.append({
            "year": y,
            "trades": v["trades"],
            "pnl": round(v["pnl"], 2),
            "wr": round(v["wins"] / v["trades"] * 100, 1) if v["trades"] > 0 else 0,
        })

    return {
        "trades": nt,
        "win_rate": round(wr, 1),
        "total_pnl": round(tot, 2),
        "profit_factor": round(pf, 2),
        "max_dd": round(max_dd, 2),
        "max_dd_pct": round(max_dd_pct, 1),
        "avg_win": round(aw, 2),
        "avg_loss": round(al, 2),
        "expectancy": round(exp, 2),
        "t1": t1,
        "t2": t2,
        "final_account": round(final, 2),
        "bars": nn,
        "yearly": yearly_list,
    }


def main():
    db = "nq_data.db"

    # Walk-Forward splits
    splits = [
        {"name": "Primary", "train": ("2016-01-01", "2022-12-31"), "test": ("2023-01-01", "2025-12-31")},
        {"name": "Alt-1", "train": ("2016-01-01", "2021-12-31"), "test": ("2022-01-01", "2025-12-31")},
        {"name": "Alt-2", "train": ("2016-01-01", "2020-12-31"), "test": ("2021-01-01", "2025-12-31")},
    ]

    results = {}

    for split in splits:
        print(f"\n{'=' * 70}")
        print(f"  WALK-FORWARD: {split['name']}")
        print(f"  Train: {split['train'][0]} to {split['train'][1]}")
        print(f"  Test:  {split['test'][0]} to {split['test'][1]}")
        print(f"{'=' * 70}")

        for mult in [1.5, 1.2]:
            print(f"\n  --- surge x{mult} ---")

            print(f"  Running TRAIN period...")
            train_res = run_backtest_for_period(db, split["train"][0], split["train"][1], mult)
            print(f"    Train: {train_res['trades']} trades, P&L ${train_res['total_pnl']:+,.0f}, "
                  f"WR {train_res['win_rate']}%, PF {train_res['profit_factor']}")

            print(f"  Running TEST period...")
            test_res = run_backtest_for_period(db, split["test"][0], split["test"][1], mult)
            print(f"    Test:  {test_res['trades']} trades, P&L ${test_res['total_pnl']:+,.0f}, "
                  f"WR {test_res['win_rate']}%, PF {test_res['profit_factor']}")

            key = f"{split['name']}_x{mult}"
            results[key] = {
                "split": split["name"],
                "mult": mult,
                "train": train_res,
                "test": test_res,
            }

    # Summary
    print(f"\n\n{'=' * 90}")
    print(f"  WALK-FORWARD SUMMARY")
    print(f"{'=' * 90}")
    print(f"{'Split':>10} | {'Surge':>6} | {'Train P&L':>12} | {'Train PF':>8} | "
          f"{'Test P&L':>12} | {'Test PF':>8} | {'Test WR':>7} | {'Test DD':>10} | {'Degr%':>6}")
    print("-" * 90)

    for split in splits:
        for mult in [1.5, 1.2]:
            key = f"{split['name']}_x{mult}"
            r = results[key]
            tr = r["train"]
            te = r["test"]
            # Degradation: test expectancy vs train expectancy
            train_exp = tr["expectancy"]
            test_exp = te["expectancy"]
            degr = ((test_exp - train_exp) / abs(train_exp) * 100) if train_exp != 0 else 0
            marker = " <--" if mult == 1.2 else ""
            print(f"{split['name']:>10} | x{mult:.1f}   | ${tr['total_pnl']:>+11,.0f} | {tr['profit_factor']:>8.2f} | "
                  f"${te['total_pnl']:>+11,.0f} | {te['profit_factor']:>8.2f} | {te['win_rate']:>6.1f}% | "
                  f"${te['max_dd']:>+9,.0f} | {degr:>+5.0f}%{marker}")
        print("-" * 90)

    # Save results
    with open("reports/walk_forward_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to reports/walk_forward_results.json")


if __name__ == "__main__":
    main()
