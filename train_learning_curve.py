"""
train_learning_curve.py -- Train a fresh PPO model and log payout rate over time.

Evaluates on holdout every --eval-freq steps and saves results to
results/learning_curve.csv so the improvement can be visualized.

Saves model to models/funded_curve/ (never touches the 4M production model).

Usage:
  python train_learning_curve.py
  python train_learning_curve.py --steps 2000000 --eval-freq 100000
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import csv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from funded_orb_env import FundedOrbEnvV3, precompute_orb

_HERE = os.path.dirname(os.path.abspath(__file__))

MAX_EVAL_STEPS = 300


class LearningCurveCallback(BaseCallback):
    def __init__(self, csv_path: str, eval_freq: int = 100_000,
                 n_eval: int = 300, verbose: int = 1):
        super().__init__(verbose)
        self.csv_path  = csv_path
        self.eval_freq = eval_freq
        self.n_eval    = n_eval
        self._last_eval = 0

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timesteps", "split", "payout_rate", "blowup_rate",
                 "stuck_rate", "avg_payout", "ev_per_ep"])

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self.eval_freq:
            self._last_eval = self.num_timesteps
            self._evaluate("holdout")
        return True

    def _evaluate(self, split: str):
        env = FundedOrbEnvV3(split=split)
        payouts, n_blowup, n_stuck = [], 0, 0

        for ep in range(self.n_eval):
            obs, _ = env.reset(seed=ep)
            done, steps = False, 0
            info = {}
            while not done and steps < MAX_EVAL_STEPS:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, t, tr, info = env.step(action)
                done = t or tr
                steps += 1
            reason = info.get("reason", "stuck")
            if reason == "payout":
                payouts.append(info.get("payout", 0.0))
            elif reason == "blowup":
                n_blowup += 1
            else:
                n_stuck += 1

        n = self.n_eval
        p_rate  = len(payouts) / n
        b_rate  = n_blowup / n
        s_rate  = n_stuck / n
        avg_p   = float(np.mean(payouts)) if payouts else 0.0
        ev      = sum(payouts) / n

        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                self.num_timesteps, split,
                round(p_rate, 4), round(b_rate, 4), round(s_rate, 4),
                round(avg_p, 2), round(ev, 2)])

        if self.verbose:
            print(f"  [{self.num_timesteps:>9,}] {split:7s} | "
                  f"payout={p_rate:5.1%}  blowup={b_rate:5.1%}  "
                  f"stuck={s_rate:5.1%}  avg=${avg_p:.0f}  EV=${ev:.1f}")


def make_env(seed: int):
    def _init():
        env = FundedOrbEnvV3(split="train")
        env.reset(seed=seed)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs",      type=int,   default=10)
    parser.add_argument("--steps",     type=int,   default=4_000_000)
    parser.add_argument("--eval-freq", type=int,   default=100_000)
    parser.add_argument("--n-eval",    type=int,   default=150)
    parser.add_argument("--lr",        type=float, default=3e-4)
    args = parser.parse_args()

    print("Precomputing ORB caches...")
    for s in ["train", "test", "holdout"]:
        precompute_orb(s)

    csv_path  = os.path.join(_HERE, "results", "learning_curve.csv")
    save_dir  = os.path.join(_HERE, "models", "funded_curve")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nBuilding {args.envs} parallel envs (train split)...")
    if args.envs == 1:
        vec_env = DummyVecEnv([make_env(0)])
    else:
        vec_env = SubprocVecEnv(
            [make_env(i) for i in range(args.envs)],
            start_method="spawn",
        )

    model = PPO(
        "MlpPolicy",
        vec_env,
        device="cpu",
        verbose=0,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.1,
        clip_range_vf=0.1,
        ent_coef=0.05,
        learning_rate=args.lr,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[64, 64]),
    )

    print(f"\n{'='*60}")
    print(f"  Learning Curve Training  (separate from production model)")
    print(f"  Steps={args.steps:,}  Envs={args.envs}  EvalFreq={args.eval_freq:,}")
    print(f"  CSV -> {csv_path}")
    print(f"  Model -> {save_dir}/")
    print(f"{'='*60}\n")

    model.learn(
        total_timesteps=args.steps,
        callback=LearningCurveCallback(
            csv_path=csv_path,
            eval_freq=args.eval_freq,
            n_eval=args.n_eval,
        ),
        progress_bar=True,
        reset_num_timesteps=True,
    )

    model.save(os.path.join(save_dir, "curve_model"))
    print(f"\nDone. CSV saved -> {csv_path}")


if __name__ == "__main__":
    main()
