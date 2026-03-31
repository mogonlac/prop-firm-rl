"""
challenge_orb_env.py — FIXED version: uses corrected ORB precompute.

Fix: ORB breakout uses HIGH/LOW detection with OR boundary entry price.
No scaling plan in the challenge (Trading Combine has full position size).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from funded_orb_env import precompute_orb

INITIAL_BALANCE    = 50_000.0
MAX_TRAILING_DD    =  2_000.0
MAX_CONTRACTS      =     50
POINT_VALUE        =      5.0
COMMISSION_PER_LEG =      0.37
SLIPPAGE_BASE      =      0.25
SLIPPAGE_SCALE     =      0.02

PROFIT_TARGET = 3_000.0
CONSISTENCY   = 0.50

MIN_TP   =  2
MAX_TP   = 24
MIN_SL   =  2
MAX_SL   = 24
MAX_DAYS = 30

N_OBS = 8


def _rt_cost(n: int) -> float:
    leg = (n * COMMISSION_PER_LEG
           + n * SLIPPAGE_BASE * (1.0 + SLIPPAGE_SCALE * max(0, n - 1)))
    return 2.0 * leg


class ChallengeOrbEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, split: str = "train",
                 max_contracts: int = MAX_CONTRACTS):
        super().__init__()
        cache = precompute_orb(split)
        d = np.load(cache)

        self.has_trade    = d["has_trade"].astype(bool)
        self.direction    = d["direction"].astype(np.int32)
        self.entry_price  = d["entry_price"]
        self.or_range_pct = d["or_range_pct"]
        self.n_post_bars  = d["n_post_bars"].astype(np.int32)
        self.post_highs   = d["post_highs"]
        self.post_lows    = d["post_lows"]
        self.post_closes  = d["post_closes"]
        self.max_contracts = max_contracts

        self._valid = np.where(self.has_trade)[0]

        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(N_OBS,), dtype=np.float32)
        self._rng = np.random.default_rng()

        self.balance        = INITIAL_BALANCE
        self.trailing_floor = INITIAL_BALANCE - MAX_TRAILING_DD
        self.best_day_pnl   = 0.0
        self.days_traded    = 0
        self._day_ptr       = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.balance        = INITIAL_BALANCE
        self.trailing_floor = INITIAL_BALANCE - MAX_TRAILING_DD
        self.best_day_pnl   = 0.0
        self.days_traded    = 0

        max_start = max(0, len(self._valid) - MAX_DAYS - 1)
        self._day_ptr = int(self._rng.integers(0, max_start + 1))

        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        profit   = self.balance - INITIAL_BALANCE
        headroom = (self.balance - self.trailing_floor) / MAX_TRAILING_DD
        c_danger = 0.0
        if profit > 1e-6 and self.best_day_pnl > 0:
            c_danger = self.best_day_pnl / (profit * CONSISTENCY + 1e-8)
        di = self._valid[self._day_ptr]
        return np.array([
            np.clip(headroom,                          -1.0, 3.0),
            np.clip(profit / PROFIT_TARGET,            -2.0, 3.0),
            np.clip(c_danger,                           0.0, 5.0),
            max(0.0, PROFIT_TARGET - max(0.0, profit)) / PROFIT_TARGET,
            np.clip(self.best_day_pnl / PROFIT_TARGET,  0.0, 3.0),
            self.days_traded / MAX_DAYS,
            float(self.direction[di]),
            float(np.clip(self.or_range_pct[di] / 2.0,  0.0, 5.0)),
        ], dtype=np.float32)

    def _simulate(self, di: int, n: int, tp: int, sl: int) -> float:
        entry = float(self.entry_price[di])
        dir_  = int(self.direction[di])
        n_pb  = int(self.n_post_bars[di])
        cost  = _rt_cost(n)

        tp_price = entry + dir_ * tp
        sl_price = entry - dir_ * sl
        ph = self.post_highs[di]
        pl = self.post_lows[di]
        pc = self.post_closes[di]

        for j in range(n_pb):
            if dir_ == 1:
                if pl[j] <= sl_price:
                    return -(sl * n * POINT_VALUE + cost)
                if ph[j] >= tp_price:
                    return  tp * n * POINT_VALUE - cost
            else:
                if ph[j] >= sl_price:
                    return -(sl * n * POINT_VALUE + cost)
                if pl[j] <= tp_price:
                    return  tp * n * POINT_VALUE - cost

        exit_p = float(pc[n_pb - 1])
        return dir_ * (exit_p - entry) * n * POINT_VALUE - cost

    def step(self, action):
        n  = int(round(float(np.clip(action[0], 0.0, 1.0)) * self.max_contracts))
        tp = max(MIN_TP, int(round(
            float(np.clip(action[1], 0.0, 1.0)) * (MAX_TP - MIN_TP) + MIN_TP)))
        sl = max(MIN_SL, int(round(
            float(np.clip(action[2], 0.0, 1.0)) * (MAX_SL - MIN_SL) + MIN_SL)))
        n  = max(0, min(n, self.max_contracts))

        di  = self._valid[self._day_ptr]
        pnl = self._simulate(di, n, tp, sl) if n > 0 else 0.0

        self.days_traded += 1
        self.balance     += pnl

        new_floor = self.balance - MAX_TRAILING_DD
        self.trailing_floor = min(
            max(self.trailing_floor, new_floor), INITIAL_BALANCE)

        profit = self.balance - INITIAL_BALANCE
        if pnl > self.best_day_pnl:
            self.best_day_pnl = pnl

        reward = pnl / (MAX_TRAILING_DD * 2)
        if n == 0:
            reward -= 0.02
        eff_best = self.best_day_pnl
        if profit > 1e-6 and eff_best > 0:
            c_ratio = eff_best / (profit * CONSISTENCY)
            if c_ratio > 1.0:
                reward -= 0.02 * min(c_ratio - 1.0, 3.0)

        if self.balance <= self.trailing_floor:
            reward -= 1.5
            return self._obs(), reward, True, False, {"reason": "blowup"}

        if profit >= PROFIT_TARGET:
            if eff_best <= 0 or (eff_best / profit) <= CONSISTENCY:
                reward += 5.0
                return self._obs(), reward, True, False, {"reason": "pass"}

        self._day_ptr += 1
        if self._day_ptr >= len(self._valid):
            self._day_ptr = 0

        if self.days_traded >= MAX_DAYS:
            return self._obs(), reward - 0.5, True, False, {"reason": "timeout"}

        return self._obs(), reward, False, False, {}

    def render(self):
        pass
