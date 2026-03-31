"""
backtest_combined_v3.py -- End-to-end: challenge (rule-based) -> funded (PPO v3).

Phase 1: Rule-based agent attempts the Trading Combine.
Phase 2: Passing accounts go to funded; PPO v3 (VIX-aware sizing, fixed TP=6/SL=18)
         runs up to 2 payout windows per account.

Saves to results/:
  challenge_equity.csv  -- daily balance for every challenge episode
  funded_equity.csv     -- daily balance for every funded episode (w1 + w2)

Usage:
  python backtest_combined_v3.py
  python backtest_combined_v3.py --attempts 500 --split holdout
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from challenge_orb_env import ChallengeOrbEnv, INITIAL_BALANCE as CH_INIT_BAL
from challenge_orb_agent import ChallengeOrbAgent
from funded_orb_env import precompute_orb, FundedOrbEnvV3

_HERE = os.path.dirname(os.path.abspath(__file__))

CHALLENGE_SUB      =  49.0
ACTIVATION_FEE     = 149.0
PROFIT_SPLIT       =   0.90
PAYOUT_CAP         = 6_000.0
MAX_FUNDED_DAYS    = 300
MAX_FUNDED_PAYOUTS =   2


def _gross_payout(balance: float) -> float:
    return min(max(balance, 0.0) * 0.50, PAYOUT_CAP)


def _net_payout(gross: float) -> float:
    return gross * PROFIT_SPLIT


def run_challenge(split: str, n: int):
    """Run n challenge episodes. Returns (results, equity_rows)."""
    env   = ChallengeOrbEnv(split=split)
    agent = ChallengeOrbAgent()
    results     = []
    equity_rows = []
    for ep in range(n):
        obs, _ = env.reset(seed=ep)
        agent.reset()
        done = False
        day  = 0
        equity_rows.append({"attempt_id": ep, "day": day,
                             "balance": env.balance})
        while not done:
            action = agent.act(env)
            obs, _, t, tr, info = env.step(action)
            done = t or tr
            day += 1
            equity_rows.append({"attempt_id": ep, "day": day,
                                 "balance": env.balance})
        results.append(info.get("reason", "timeout"))
    return results, equity_rows


def run_funded_window(env: FundedOrbEnvV3, model, seed: int,
                      initial_balance: float = 0.0,
                      trailing_floor: float = -2000.0) -> dict:
    obs, _ = env.reset(seed=seed, options={
        "initial_balance": initial_balance,
        "trailing_floor":  trailing_floor,
    })
    done, days = False, 0
    info   = {}
    equity = [initial_balance]
    while not done and days < MAX_FUNDED_DAYS:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, t, tr, info = env.step(action)
        done = t or tr
        days += 1
        equity.append(env.balance)
    reason = info.get("reason", "stuck")
    gross  = _gross_payout(env.balance) if reason == "payout" else 0.0
    return {
        "reason":        reason,
        "payout_gross":  gross,
        "payout_net":    _net_payout(gross),
        "final_balance": env.balance,
        "days":          days,
        "equity":        equity,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts", type=int, default=500)
    parser.add_argument("--split",    type=str, default="holdout",
                        choices=["train", "test", "holdout"])
    parser.add_argument("--model",    type=str,
                        default=os.path.join(_HERE,
                            "models/funded_v3/v3_checkpoint_4000000_steps.zip"))
    args = parser.parse_args()

    for s in ["train", "test", "holdout"]:
        precompute_orb(s)

    print(f"Loading PPO v3 model: {args.model}")
    model = PPO.load(args.model, device="cpu")
    funded_env = FundedOrbEnvV3(split=args.split)

    print(f"\n{'='*65}")
    print(f"  Combined Backtest v3  split={args.split}  "
          f"attempts={args.attempts}")
    print(f"  Funded model: PPO v3 (VIX-aware, TP=6/SL=18)")
    print(f"{'='*65}")

    # ── Phase 1: Challenge ────────────────────────────────────────────────
    print(f"\n[1] Challenge phase...")
    ch, challenge_equity = run_challenge(args.split, args.attempts)
    n_pass    = ch.count("pass")
    n_blowup  = ch.count("blowup")
    n_timeout = ch.count("timeout")
    print(f"    pass={n_pass/args.attempts:.1%} ({n_pass})  "
          f"blowup={n_blowup/args.attempts:.1%}  "
          f"timeout={n_timeout/args.attempts:.1%}")

    # ── Phase 2: Funded ───────────────────────────────────────────────────
    print(f"\n[2] Funded phase ({n_pass} accounts, up to "
          f"{MAX_FUNDED_PAYOUTS} windows)...")

    p1_nets, p2_nets = [], []
    outcomes = {"payout1": 0, "blowup1": 0, "stuck1": 0,
                "payout2": 0, "blowup2": 0, "stuck2": 0}
    funded_equity   = []
    funded_outcomes = []
    funded_idx      = 0

    for i, ch_result in enumerate(ch):
        if ch_result != "pass":
            continue

        r1 = run_funded_window(funded_env, model,
                               seed=funded_idx * 2,
                               initial_balance=0.0,
                               trailing_floor=-2000.0)
        funded_idx += 1
        outcomes[f"{r1['reason'][:6]}1"] = outcomes.get(f"{r1['reason'][:6]}1", 0) + 1

        for day, bal in enumerate(r1["equity"]):
            funded_equity.append({"attempt_id": i, "window": 1,
                                  "day": day, "balance": bal})

        funded_outcomes.append({
            "attempt_id":    i,
            "window":        1,
            "outcome":       r1["reason"],
            "payout_gross":  r1["payout_gross"],
            "payout_net":    r1["payout_net"],
            "final_balance": r1["final_balance"],
            "days":          r1["days"],
        })

        if r1["reason"] == "payout":
            p1_nets.append(r1["payout_net"])

            remaining = r1["final_balance"] - r1["payout_gross"]
            r2 = run_funded_window(funded_env, model,
                                   seed=funded_idx * 2 + 1,
                                   initial_balance=remaining,
                                   trailing_floor=0.0)
            funded_idx += 1
            outcomes[f"{r2['reason'][:6]}2"] = outcomes.get(f"{r2['reason'][:6]}2", 0) + 1

            for day, bal in enumerate(r2["equity"]):
                funded_equity.append({"attempt_id": i, "window": 2,
                                      "day": day, "balance": bal})

            funded_outcomes.append({
                "attempt_id":    i,
                "window":        2,
                "outcome":       r2["reason"],
                "payout_gross":  r2["payout_gross"],
                "payout_net":    r2["payout_net"],
                "final_balance": r2["final_balance"],
                "days":          r2["days"],
            })

            if r2["reason"] == "payout":
                p2_nets.append(r2["payout_net"])

    # ── Save CSVs ─────────────────────────────────────────────────────────
    results_dir = os.path.join(_HERE, "results")
    os.makedirs(results_dir, exist_ok=True)
    pd.DataFrame(challenge_equity).to_csv(
        os.path.join(results_dir, "challenge_equity.csv"), index=False)
    pd.DataFrame(funded_equity).to_csv(
        os.path.join(results_dir, "funded_equity.csv"), index=False)
    pd.DataFrame(funded_outcomes).to_csv(
        os.path.join(results_dir, "funded_outcomes.csv"), index=False)
    print(f"\n  Saved results/challenge_equity.csv  ({len(challenge_equity)} rows)")
    print(f"  Saved results/funded_equity.csv     ({len(funded_equity)} rows)")
    print(f"  Saved results/funded_outcomes.csv   ({len(funded_outcomes)} rows)")

    # ── Print results ─────────────────────────────────────────────────────
    n_p1 = len(p1_nets);  n_p2 = len(p2_nets)
    avg_p1 = np.mean(p1_nets) if p1_nets else 0.0
    avg_p2 = np.mean(p2_nets) if p2_nets else 0.0

    print(f"\n    Window 1 (n={n_pass}):")
    print(f"      payout={n_p1/max(n_pass,1):5.1%} ({n_p1})  "
          f"blowup={outcomes.get('blowup1',0)/max(n_pass,1):5.1%}  "
          f"stuck={outcomes.get('stuck1',0)/max(n_pass,1):5.1%}  "
          f"avg_net=${avg_p1:,.0f}")

    if n_p1:
        print(f"    Window 2 (n={n_p1} who paid out):")
        print(f"      payout={n_p2/max(n_p1,1):5.1%} ({n_p2})  "
              f"blowup={outcomes.get('blowup2',0)/max(n_p1,1):5.1%}  "
              f"stuck={outcomes.get('stuck2',0)/max(n_p1,1):5.1%}  "
              f"avg_net=${avg_p2:,.0f}")

    # ── Economics ─────────────────────────────────────────────────────────
    gross_revenue   = sum(p1_nets) + sum(p2_nets)
    challenge_cost  = args.attempts * CHALLENGE_SUB
    activation_cost = n_pass * ACTIVATION_FEE
    total_cost      = challenge_cost + activation_cost
    net_profit      = gross_revenue - total_cost
    ev_per_attempt  = net_profit / args.attempts

    print(f"\n{'-'*65}")
    print(f"  Gross revenue:         ${gross_revenue:>10,.0f}")
    print(f"  Challenge subs:       -${challenge_cost:>10,.0f}  "
          f"({args.attempts} x ${CHALLENGE_SUB:.0f})")
    print(f"  Activation fees:      -${activation_cost:>10,.0f}  "
          f"({n_pass} x ${ACTIVATION_FEE:.0f})")
    print(f"  Total costs:          -${total_cost:>10,.0f}")
    print(f"{'-'*65}")
    print(f"  Net profit:            ${net_profit:>10,.0f}")
    print(f"  EV per attempt:        ${ev_per_attempt:>10,.2f}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
