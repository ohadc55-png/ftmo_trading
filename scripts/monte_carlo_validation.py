"""Monte Carlo Simulation: Shuffle 11,687 trades 10,000 times to get confidence intervals."""

import json
import numpy as np

# Load trades from report data
with open("reports/report_data.json") as f:
    d = json.load(f)

trades = d["trades"]
pnls = np.array([t["pnl"] for t in trades])
n_trades = len(pnls)

print(f"Monte Carlo Simulation")
print(f"Trades: {n_trades:,}")
print(f"Original P&L: ${pnls.sum():+,.0f}")
print(f"Original Max DD: ", end="")

# Original equity curve stats
equity = np.cumsum(np.insert(pnls, 0, 100000))
peak = np.maximum.accumulate(equity)
dd = equity - peak
orig_dd = dd.min()
orig_dd_pct = (orig_dd / peak[np.argmin(dd)]) * 100
print(f"${orig_dd:+,.0f} ({orig_dd_pct:.1f}%)")

N_SIMS = 10000
START_CAPITAL = 100_000

print(f"\nRunning {N_SIMS:,} simulations...")

# Pre-allocate results
sim_pnls = np.zeros(N_SIMS)
sim_max_dds = np.zeros(N_SIMS)
sim_max_dd_pcts = np.zeros(N_SIMS)
sim_final_accounts = np.zeros(N_SIMS)
sim_max_consec_losses = np.zeros(N_SIMS, dtype=int)
sim_win_rates = np.zeros(N_SIMS)
sim_profit_factors = np.zeros(N_SIMS)

rng = np.random.default_rng(42)

for i in range(N_SIMS):
    # Shuffle trades randomly
    shuffled = rng.permutation(pnls)

    # Build equity curve
    equity = START_CAPITAL + np.cumsum(shuffled)
    equity_full = np.insert(equity, 0, START_CAPITAL)

    # Total P&L (same every time since sum doesn't change)
    sim_pnls[i] = shuffled.sum()
    sim_final_accounts[i] = equity[-1]

    # Max drawdown
    peak = np.maximum.accumulate(equity_full)
    dd = equity_full - peak
    min_dd = dd.min()
    sim_max_dds[i] = min_dd
    idx = np.argmin(dd)
    sim_max_dd_pcts[i] = (min_dd / peak[idx] * 100) if peak[idx] > 0 else 0

    # Max consecutive losses
    is_loss = shuffled <= 0
    max_consec = 0
    curr = 0
    for loss in is_loss:
        if loss:
            curr += 1
            max_consec = max(max_consec, curr)
        else:
            curr = 0
    sim_max_consec_losses[i] = max_consec

    # Win rate (same every time)
    sim_win_rates[i] = np.sum(shuffled > 0) / n_trades * 100

    # Profit factor
    gp = shuffled[shuffled > 0].sum()
    gl = abs(shuffled[shuffled < 0].sum())
    sim_profit_factors[i] = gp / gl if gl > 0 else 999

    if (i + 1) % 2000 == 0:
        print(f"  {i+1:,} / {N_SIMS:,} done...")

print(f"  {N_SIMS:,} / {N_SIMS:,} done!")

# Compute percentiles
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

print(f"\n{'=' * 80}")
print(f"  MONTE CARLO RESULTS ({N_SIMS:,} simulations, {n_trades:,} trades)")
print(f"{'=' * 80}")

print(f"\n  Max Drawdown Distribution ($):")
dd_pcts = np.percentile(sim_max_dds, percentiles)
for p, v in zip(percentiles, dd_pcts):
    marker = " <-- WORST CASE" if p == 1 else (" <-- MEDIAN" if p == 50 else "")
    print(f"    {p:>3}th percentile: ${v:>+12,.0f}{marker}")
print(f"    Original:         ${orig_dd:>+12,.0f}")

print(f"\n  Max Drawdown Distribution (%):")
dd_pct_vals = np.percentile(sim_max_dd_pcts, percentiles)
for p, v in zip(percentiles, dd_pct_vals):
    print(f"    {p:>3}th percentile: {v:>+8.1f}%")
print(f"    Original:         {orig_dd_pct:>+8.1f}%")

print(f"\n  Max Consecutive Losses Distribution:")
cl_pcts = np.percentile(sim_max_consec_losses, percentiles)
for p, v in zip(percentiles, cl_pcts):
    marker = " <-- WORST CASE" if p == 99 else (" <-- MEDIAN" if p == 50 else "")
    print(f"    {p:>3}th percentile: {v:>6.0f}{marker}")
print(f"    Original:         {29:>6}")

# Lowest equity point
print(f"\n  Lowest Account Balance Distribution:")
sim_min_equity = np.zeros(N_SIMS)
for i in range(N_SIMS):
    shuffled = rng.permutation(pnls)
    equity = START_CAPITAL + np.cumsum(shuffled)
    sim_min_equity[i] = equity.min()

min_eq_pcts = np.percentile(sim_min_equity, percentiles)
for p, v in zip(percentiles, min_eq_pcts):
    marker = " <-- WORST CASE" if p == 1 else ""
    print(f"    {p:>3}th percentile: ${v:>12,.0f}{marker}")

# Risk of ruin (account dropping below $90K, $80K, $70K, $60K)
print(f"\n  Risk of Ruin (probability of account dropping below threshold):")
for threshold in [90000, 80000, 70000, 60000, 50000]:
    pct = np.mean(sim_min_equity < threshold) * 100
    print(f"    Below ${threshold:>6,}: {pct:>5.1f}%")

# Summary stats
print(f"\n{'=' * 80}")
print(f"  CONFIDENCE INTERVALS")
print(f"{'=' * 80}")
print(f"  95% CI for Max Drawdown:        ${np.percentile(sim_max_dds, 5):>+10,.0f} to ${np.percentile(sim_max_dds, 95):>+10,.0f}")
print(f"  95% CI for Max Drawdown %:       {np.percentile(sim_max_dd_pcts, 5):>+7.1f}% to {np.percentile(sim_max_dd_pcts, 95):>+7.1f}%")
print(f"  95% CI for Max Consec Losses:   {np.percentile(sim_max_consec_losses, 5):>6.0f} to {np.percentile(sim_max_consec_losses, 95):>6.0f}")
print(f"  99th percentile Max DD:         ${np.percentile(sim_max_dds, 1):>+10,.0f} ({np.percentile(sim_max_dd_pcts, 1):>+.1f}%)")
print(f"  99th percentile Consec Losses:  {np.percentile(sim_max_consec_losses, 99):>6.0f}")

# Key question: how robust is the strategy?
prob_dd_gt_50k = np.mean(sim_max_dds < -50000) * 100
prob_dd_gt_100k = np.mean(sim_max_dds < -100000) * 100
print(f"\n  Probability of DD > $50K:  {prob_dd_gt_50k:.1f}%")
print(f"  Probability of DD > $100K: {prob_dd_gt_100k:.1f}%")

# Save results
results = {
    "n_simulations": N_SIMS,
    "n_trades": n_trades,
    "original": {
        "total_pnl": round(float(pnls.sum()), 2),
        "max_dd": round(float(orig_dd), 2),
        "max_dd_pct": round(float(orig_dd_pct), 1),
    },
    "max_dd_percentiles": {str(p): round(float(v), 2) for p, v in zip(percentiles, dd_pcts)},
    "max_dd_pct_percentiles": {str(p): round(float(v), 1) for p, v in zip(percentiles, dd_pct_vals)},
    "max_consec_loss_percentiles": {str(p): int(v) for p, v in zip(percentiles, cl_pcts)},
    "min_equity_percentiles": {str(p): round(float(v), 2) for p, v in zip(percentiles, min_eq_pcts)},
    "risk_of_ruin": {
        str(t): round(float(np.mean(sim_min_equity < t) * 100), 1)
        for t in [90000, 80000, 70000, 60000, 50000]
    },
    "confidence_95": {
        "max_dd_low": round(float(np.percentile(sim_max_dds, 5)), 2),
        "max_dd_high": round(float(np.percentile(sim_max_dds, 95)), 2),
        "max_dd_pct_low": round(float(np.percentile(sim_max_dd_pcts, 5)), 1),
        "max_dd_pct_high": round(float(np.percentile(sim_max_dd_pcts, 95)), 1),
        "consec_loss_low": int(np.percentile(sim_max_consec_losses, 5)),
        "consec_loss_high": int(np.percentile(sim_max_consec_losses, 95)),
    },
    "prob_dd_gt_50k": round(float(prob_dd_gt_50k), 1),
    "prob_dd_gt_100k": round(float(prob_dd_gt_100k), 1),
}

with open("reports/monte_carlo_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to reports/monte_carlo_results.json")
