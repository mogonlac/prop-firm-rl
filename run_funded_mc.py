"""
run_funded_mc.py -- Run N funded episodes across the full dataset for Monte Carlo.

Uses split="all" (train + test + holdout combined) so episodes have the full
history to draw from without wrapping around a small holdout window.

Overwrites results/funded_equity.csv with the new episodes.

Usage:
  python run_funded_mc.py
  python run_funded_mc.py --episodes 5000
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from funded_orb_env import FundedOrbEnvV3

MAX_FUNDED_DAYS = 300

_HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--split", type=str, default="all",
                        choices=["train", "test", "holdout", "all", "oos"])
    parser.add_argument("--model", type=str,
                        default=os.path.join(_HERE,
                            "models/funded_v3/v3_checkpoint_4000000_steps.zip"))
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = PPO.load(args.model, device="cpu")

    print(f"Loading funded env (split={args.split})...")
    env = FundedOrbEnvV3(split=args.split)
    print(f"  {env.n_days} total days across all splits")

    equity_rows = []
    n_payout = 0
    n_blowup = 0

    print(f"Running {args.episodes} episodes...")
    for ep in range(args.episodes):
        if ep % 500 == 0:
            print(f"  {ep}/{args.episodes}  payout={n_payout}  blowup={n_blowup}")

        obs, _ = env.reset(seed=ep)
        done, day = False, 0
        info = {}
        equity_rows.append({"episode_id": ep, "day": day, "balance": env.balance})

        while not done and day < MAX_FUNDED_DAYS:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, t, tr, info = env.step(action)
            done = t or tr
            day += 1
            equity_rows.append({"episode_id": ep, "day": day, "balance": env.balance})

        reason = info.get("reason", "stuck")
        if reason == "payout":
            n_payout += 1
        elif reason == "blowup":
            n_blowup += 1

    print(f"\nDone.")
    print(f"  payout={n_payout/args.episodes:.1%}  blowup={n_blowup/args.episodes:.1%}")

    results_dir = os.path.join(_HERE, "results")
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, "funded_equity.csv")
    pd.DataFrame(equity_rows).to_csv(out, index=False)
    print(f"  Saved {out}  ({len(equity_rows)} rows)")


if __name__ == "__main__":
    main()
