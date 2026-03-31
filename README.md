# propbot v10

Automated trading bot for the **Topstep $50K XFA Funded Account**, using an Opening Range Breakout (ORB) strategy on MES futures.

## Architecture

Two-phase pipeline:

**Phase 1 — Challenge (rule-based)**
A fixed-parameter agent attempts the Trading Combine: 50 MES contracts, TP=5 ticks, SL=24 ticks. No learning — the rule set was validated via walk-forward analysis.

**Phase 2 — Funded (PPO)**
Accounts that pass the challenge enter the funded stage. A PPO model (stable-baselines3) controls contract sizing only, with fixed TP=6 / SL=18 ticks. VIX and VIX 14-day MA are included in the observation to enable regime-aware sizing. The scaling plan mirrors Topstep's XFA rules (10/20/30/40/50 MES contracts at $0/$500/$1,000/$1,500/$2,000 profit tiers).

## Data splits

| Split | Range | Notes |
|---|---|---|
| Train | 0–70% | PPO trained on this |
| Test | 70–85% | Used for checkpoint selection during training |
| Holdout | 85–100% | Completely unseen — use this for unbiased evaluation |
| OOS | 70–100% | Test + holdout combined, for Monte Carlo visualization |
| All | 0–100% | Full dataset, used for dense Monte Carlo runs |

> Holdout is the only truly unbiased split. Test had indirect selection pressure applied — the best checkpoint was chosen partly based on test performance.

## Key results (holdout)

| Metric | Value |
|---|---|
| Challenge pass rate | 58.4% |
| Funded window 1 payout | ~48% |
| Funded window 1 blowup | ~42% |
| Avg net payout (W1) | ~$1,340 |
| EV per challenge attempt | ~$270 |

## Files

| File | Purpose |
|---|---|
| `challenge_orb_env.py` | Gymnasium env for the Trading Combine |
| `challenge_orb_agent.py` | Fixed-parameter rule-based challenge agent |
| `funded_orb_env.py` | Gymnasium env for the funded account (VIX-aware PPO) |
| `train_funded_v3.py` | PPO training script |
| `eval_v3.py` | Evaluate a trained model on a single split |
| `backtest_combined_v3.py` | End-to-end backtest: challenge → funded, saves CSVs |
| `run_funded_mc.py` | Standalone Monte Carlo runner for the funded env |
| `visualize.py` | Monte Carlo equity curve chart from funded_equity.csv |
| `preprocess_mes.py` | Raw tick data → preprocessed_mes.csv |
| `download_vix.py` | Download VIX daily closes → data/vix.csv |
| `TOPSTEPX_RULES.md` | Exact Topstep XFA rules, commissions, payout policy |
| `requirements.txt` | Python dependencies |

## Usage

### 1. Preprocess data
```bash
python preprocess_mes.py
python download_vix.py
```

### 2. Train the funded model
```bash
python train_funded_v3.py
```

### 3. Run end-to-end backtest
```bash
python backtest_combined_v3.py --attempts 500 --split holdout
```

### 4. Run Monte Carlo (funded only, larger episode count)
```bash
python run_funded_mc.py --episodes 2000 --split oos
```

### 5. Visualize
```bash
python visualize.py
# output: results/plots/funded_w1_montecarlo.png
```

## Payout logic (XFA Consistency Path)

- Minimum 3 days traded
- Net profit ≥ $250
- No single day > 40% of total profit
- Payout = 50% of balance, capped at $6,000 gross (90% to trader)
- Trailing drawdown = $2,000, floor rises with peak balance, capped at $0

See `TOPSTEPX_RULES.md` for full details.
