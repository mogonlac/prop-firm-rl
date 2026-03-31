import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stable_baselines3 import PPO
from funded_orb_env import FundedOrbEnvV3

model = PPO.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
    "models/funded_v3/v3_checkpoint_4000000_steps"), device="cpu")

for split in ["train", "test", "holdout"]:
    env = FundedOrbEnvV3(split=split)
    payouts, blowups = [], 0
    for ep in range(500):
        obs, _ = env.reset(seed=ep)
        done, steps = False, 0
        while not done and steps < 200:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, t, tr, info = env.step(action)
            done = t or tr
            steps += 1
        r = info.get("reason", "")
        if r == "payout": payouts.append(info["payout"])
        elif r == "blowup": blowups += 1
    print(f"{split:8s}: payout={len(payouts)/500:5.1%} | blowup={blowups/500:5.1%} | "
          f"avg_payout=${np.mean(payouts) if payouts else 0:.0f} | EV/ep=${sum(payouts)/500:.1f}")
