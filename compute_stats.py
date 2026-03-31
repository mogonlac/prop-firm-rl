"""
compute_stats.py — Confidence intervals + Sharpe ratio from backtest results.
Run after backtest_combined_v3.py has generated results/funded_outcomes.csv
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Load exact outcomes ────────────────────────────────────────────────────────
outcomes_df    = pd.read_csv(os.path.join(_HERE, "results", "funded_outcomes.csv"))
equity_df      = pd.read_csv(os.path.join(_HERE, "results", "funded_equity.csv"))
ch_equity_df   = pd.read_csv(os.path.join(_HERE, "results", "challenge_equity.csv"))

N_ATTEMPTS = len(ch_equity_df["attempt_id"].unique())
w1         = outcomes_df[outcomes_df["window"] == 1]
N_PASS     = len(w1)
w1_payouts = w1[w1["outcome"] == "payout"]["payout_net"]
N_PAYOUT   = len(w1_payouts)

# ── 1. Binomial CIs (Wilson interval) ─────────────────────────────────────────
def wilson_ci(k, n, z=1.96):
    p     = k / n
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return centre - margin, centre + margin

pass_lo,   pass_hi   = wilson_ci(N_PASS,   N_ATTEMPTS)
payout_lo, payout_hi = wilson_ci(N_PAYOUT, N_PASS)

# ── 0. Challenge duration ─────────────────────────────────────────────────────
ch_days = ch_equity_df.groupby("attempt_id")["day"].max()

print("=" * 55)
print("  CHALLENGE PHASE — DURATION")
print("=" * 55)
print(f"  Attempts:             {N_ATTEMPTS}")
print(f"  Mean days:            {ch_days.mean():.1f}")
print(f"  Median days:          {ch_days.median():.0f}")
print(f"  Std days:             {ch_days.std():.1f}")
print(f"  Min / Max:            {ch_days.min():.0f}  /  {ch_days.max():.0f}")
print()
print("=" * 55)
print("  CONFIDENCE INTERVALS (95%, Wilson)")
print("=" * 55)
print(f"  Challenge pass rate:  {N_PASS/N_ATTEMPTS:.1%}  "
      f"[{pass_lo:.1%}, {pass_hi:.1%}]  (n={N_ATTEMPTS})")
print(f"  Funded payout rate:   {N_PAYOUT/N_PASS:.1%}  "
      f"[{payout_lo:.1%}, {payout_hi:.1%}]  (n={N_PASS})")

# ── 2. Avg payout CI (exact values from funded_outcomes.csv) ──────────────────
mean_p = w1_payouts.mean()
se_p   = w1_payouts.sem()
t_crit = stats.t.ppf(0.975, df=N_PAYOUT - 1)
ci_lo  = mean_p - t_crit * se_p
ci_hi  = mean_p + t_crit * se_p

print(f"\n  Avg net payout (W1):  ${mean_p:,.0f}  "
      f"[${ci_lo:,.0f}, ${ci_hi:,.0f}]  (n={N_PAYOUT})")
print(f"  Std net payout (W1):  ${w1_payouts.std():,.0f}")
print(f"  Min / Max:            ${w1_payouts.min():,.0f}  /  ${w1_payouts.max():,.0f}")

# ── 3. Sharpe — funded W1 daily P&L ───────────────────────────────────────────
eq_w1     = equity_df[equity_df["window"] == 1].sort_values(["attempt_id", "day"])
eq_w1     = eq_w1.copy()
eq_w1["daily_pnl"] = eq_w1.groupby("attempt_id")["balance"].diff()
daily_pnl = eq_w1["daily_pnl"].dropna()

mu            = daily_pnl.mean()
sigma         = daily_pnl.std()
sharpe_daily  = mu / sigma if sigma > 0 else 0
sharpe_annual = sharpe_daily * np.sqrt(252)

traded   = daily_pnl[daily_pnl != 0]
win_rate = (traded > 0).mean()

# ── 4. Calmar — annualised return / max peak-to-trough drawdown (W1) ──────────
ann_return = mu * 252

def _max_dd(grp):
    bal = grp["balance"].values
    peak = np.maximum.accumulate(bal)
    return float((peak - bal).max())

max_dd = eq_w1.groupby("attempt_id").apply(_max_dd).max()
calmar = ann_return / max_dd if max_dd > 0 else float("inf")

print(f"\n{'=' * 55}")
print(f"  FUNDED PHASE — DAILY P&L STATS (W1)")
print(f"{'=' * 55}")
print(f"  Mean daily P&L:       ${mu:,.2f}")
print(f"  Std daily P&L:        ${sigma:,.2f}")
print(f"  Daily Sharpe:          {sharpe_daily:.3f}")
print(f"  Annualised Sharpe:     {sharpe_annual:.3f}  (x sqrt(252))")
print(f"  Annualised return:    ${ann_return:,.0f}  (mean daily x 252)")
print(f"  Max episode drawdown: ${max_dd:,.0f}")
print(f"  Calmar ratio:          {calmar:.3f}  (ann. return / max DD)")
print(f"  Day win rate:          {win_rate:.1%}  (excl. skipped days)")
print(f"  Total trade-days:      {len(traded):,}")
print(f"\n{'=' * 55}\n")
