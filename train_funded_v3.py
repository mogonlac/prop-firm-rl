"""
train_funded_v3.py -- Train PPO sizing agent with VIX-aware observations.

Pipeline:
  - Always take every ORB setup (TP=6 / SL=18)
  - PPO controls sizing fraction (0 -> scaling cap contracts)
  - PPO observes account state + VIX (normalized + ratio vs 14d MA)

Usage:
  python train_funded_v3.py
  python train_funded_v3.py --envs 12 --steps 8000000
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from funded_orb_env import FundedOrbEnvV3, N_OBS_V3, precompute_orb, MAX_CONTRACTS, SCALING_PLAN_50K

_HERE = os.path.dirname(os.path.abspath(__file__))


class EvalCallback(BaseCallback):
    def __init__(self, eval_freq: int = 500_000,
                 n_eval: int = 200, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval    = n_eval

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq < self.training_env.num_envs:
            self._evaluate("train")
            self._evaluate("test")
        return True

    def _evaluate(self, split: str):
        env = FundedOrbEnvV3(split=split)
        payouts, blowups = [], 0

        for ep in range(self.n_eval):
            obs, _ = env.reset(seed=ep)
            done, steps = False, 0
            while not done and steps < 200:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, t, tr, info = env.step(action)
                done = t or tr
                steps += 1
            r = info.get("reason", "")
            if r == "payout":  payouts.append(info["payout"])
            elif r == "blowup": blowups += 1

        n_pay = len(payouts)
        p_rate = n_pay / self.n_eval
        avg_p  = np.mean(payouts) if payouts else 0.0
        ev     = sum(payouts) / self.n_eval
        print(f"  [{self.num_timesteps:>9,}] {split:7s} | "
              f"payout={p_rate:5.1%} ({n_pay}/{self.n_eval}) | "
              f"blowup={blowups/self.n_eval:5.1%} | "
              f"avg_payout=${avg_p:.0f} | EV/ep=${ev:.1f}")


def make_env(split: str, seed: int):
    def _init():
        env = FundedOrbEnvV3(split=split)
        env.reset(seed=seed)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO sizing with VIX-aware observations")
    parser.add_argument("--envs",       type=int,   default=10)
    parser.add_argument("--steps",      type=int,   default=5_000_000)
    parser.add_argument("--split",      type=str,   default="train",
                        choices=["train", "test", "holdout"])
    parser.add_argument("--save",       type=str,
                        default="models/funded_v3/model")
    parser.add_argument("--checkpoint", type=int,   default=1_000_000)
    parser.add_argument("--eval-freq",  type=int,   default=500_000)
    parser.add_argument("--lr",         type=float, default=3e-4)
    args = parser.parse_args()

    save_dir = os.path.join(_HERE, args.save)
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    os.makedirs(os.path.join(_HERE, "logs", "funded_v3"), exist_ok=True)

    print("Precomputing ORB caches...")
    for s in ["train", "test", "holdout"]:
        precompute_orb(s)

    print(f"\nBuilding {args.envs} parallel envs...")
    if args.envs == 1:
        vec_env = DummyVecEnv([make_env(args.split, 0)])
    else:
        vec_env = SubprocVecEnv(
            [make_env(args.split, i) for i in range(args.envs)],
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
        tensorboard_log=os.path.join(_HERE, "logs", "funded_v3"),
    )

    print(f"\n{'='*60}")
    print(f"  Training: PPO v3 (VIX-aware sizing)")
    print(f"  Fixed: TP=6 / SL=18")
    print(f"  Split={args.split}  Envs={args.envs}  Steps={args.steps:,}")
    print(f"{'='*60}\n")

    callbacks = [
        CheckpointCallback(
            save_freq=max(1, args.checkpoint // args.envs),
            save_path=os.path.dirname(save_dir),
            name_prefix="v3_checkpoint",
            verbose=0,
        ),
        EvalCallback(
            eval_freq=args.eval_freq,
            n_eval=200,
        ),
    ]

    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=True,
    )

    model.save(save_dir)
    print(f"\nModel saved -> {save_dir}.zip")

    print("\nFinal evaluation (500 episodes per split):")
    final_model = PPO.load(save_dir, device="cpu")
    for split in ["train", "test", "holdout"]:
        env = FundedOrbEnvV3(split=split)
        payouts, blowups = [], 0
        for ep in range(500):
            obs, _ = env.reset(seed=ep)
            done, steps = False, 0
            while not done and steps < 200:
                action, _ = final_model.predict(obs, deterministic=True)
                obs, _, t, tr, info = env.step(action)
                done = t or tr
                steps += 1
            r = info.get("reason", "")
            if r == "payout":  payouts.append(info["payout"])
            elif r == "blowup": blowups += 1
        n = 500
        print(f"  {split:8s}: payout={len(payouts)/n:5.1%} | "
              f"blowup={blowups/n:5.1%} | "
              f"avg_payout=${np.mean(payouts) if payouts else 0:.0f} | "
              f"EV/ep=${sum(payouts)/n:.1f}")


if __name__ == "__main__":
    main()
