# prop-firm-rl

Systematic EV extraction from prop trading firms using reinforcement learning.

A two-phase pipeline applied to the Topstep $50K challenge: a rule-based agent passes the Trading Combine, then a PPO model optimises contract sizing in the funded account. Backtested on 6 years of MES futures data (2020–2026) with results reported on a fully held-out validation set.

**2.8x return on deployed capital — holdout set**

See [`paper.pdf`](paper.pdf) for full methodology, results, and economic analysis.

---

## Architecture

**Phase 1 — Challenge (rule-based)**
Fixed agent: 50 MES contracts, TP=5 pts, SL=24 pts on ORB signals. No learning — parameters were found by PPO during development then fixed.

**Phase 2 — Funded (PPO)**
A PPO model (stable-baselines3) controls contract sizing only, with fixed TP=6 / SL=18 pts. VIX and its 14-day MA are included in the observation for regime-aware sizing. Follows Topstep's scaling plan (10→50 contracts across $0–$2,000 profit tiers).

---

## Results (holdout)

| Metric | Value |
|---|---|
| Challenge pass rate | 58.4% [54.0%, 62.6%] |
| Funded payout rate (W1) | 48.3% [42.6%, 54.0%] |
| Avg net payout (W1) | $1,358 [$1,318, $1,399] |
| EV per attempt | $247 |
| Return on deployed capital | 2.8x |

95% confidence intervals shown. All figures net of commissions and slippage.

---

## Data

Source: Databento GLBX.MDP3, 1-minute OHLCV MES continuous futures, 2020-01-02 to 2026-03-13 (1,602 trading days).

Splits: Train 0–70% · Test 70–85% · Holdout 85–100%

> Holdout is the only fully unbiased split — test carried indirect selection pressure during checkpoint selection.

---

## Files

| File | Purpose |
|---|---|
| `challenge_orb_env.py` | Gymnasium env for the Trading Combine |
| `challenge_orb_agent.py` | Fixed-parameter rule-based challenge agent |
| `funded_orb_env.py` | Gymnasium env for the funded account (VIX-aware PPO) |
| `train_funded_v3.py` | PPO training script |
| `backtest_combined_v3.py` | End-to-end backtest: challenge → funded, saves results |
| `run_funded_mc.py` | Monte Carlo runner for the funded env |
| `visualize.py` | Monte Carlo equity curve chart |
| `compute_stats.py` | Confidence intervals and Sharpe from backtest results |
| `sensitivity_analysis.py` | EV heatmap across challenge pass rate / funded payout rate |
| `preprocess_mes.py` | Raw tick data → preprocessed_mes.csv |
| `download_vix.py` | Download VIX daily closes → data/vix.csv |

---

## Usage

```bash
# 1. Prepare data
python preprocess_mes.py
python download_vix.py

# 2. Train
python train_funded_v3.py

# 3. Backtest
python backtest_combined_v3.py --attempts 500 --split holdout

# 4. Monte Carlo (funded stage only)
python run_funded_mc.py --episodes 2000 --split oos

# 5. Visualize + stats
python visualize.py
python compute_stats.py
```

---

## Payout rules (Topstep XFA Consistency Path)

- Minimum 3 days traded
- Net profit ≥ $250
- No single day > 40% of total profit
- Payout = 50% of balance, capped at $6,000 gross (90% to trader)
- Trailing drawdown = $2,000, floor rises with peak balance, capped at breakeven
