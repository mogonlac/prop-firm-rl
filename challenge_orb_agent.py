"""
challenge_orb_agent.py — Same fixed-action agent, unchanged from original.
Copied here so fixed/ folder is self-contained.
"""

import numpy as np
from challenge_orb_env import (
    ChallengeOrbEnv,
    MAX_CONTRACTS, MIN_TP, MAX_TP, MIN_SL, MAX_SL,
)

N  = 50
TP =  5
SL = 24

_ACTION = np.array([
    N  / MAX_CONTRACTS,
    (TP - MIN_TP) / (MAX_TP - MIN_TP),
    (SL - MIN_SL) / (MAX_SL - MIN_SL),
], dtype=np.float32)


class ChallengeOrbAgent:
    def reset(self):
        pass

    def act(self, env: ChallengeOrbEnv) -> np.ndarray:
        return _ACTION
