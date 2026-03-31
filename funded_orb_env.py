"""
funded_orb_env.py — Funded ORB environment with VIX-aware PPO sizing.

ORB entry: HIGH/LOW breakout detection, entry at OR boundary (no look-ahead).
Fixed regime: TP=6 / SL=18 ticks.
PPO controls contract sizing only (1 action: sizing fraction 0-1).

Observation (11 features):
  [0]  headroom      DD buffer normalized
  [1]  profit        running P&L / 1000
  [2]  c_danger      consistency violation risk
  [3]  days_prog     days traded / qualifying days needed
  [4]  profit_gap    how far from $250 payout threshold
  [5]  best_day_pnl  best single day / 1000
  [6]  direction     today's breakout direction
  [7]  or_rp         OR range %
  [8]  or_range_pts  OR range in points (normalized /20)
  [9]  vix_norm      VIX close / 30.0, clipped 0-3
  [10] vix_ratio     vix / 14d-MA, clipped 0.5-2.0
"""

import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, 'data')

# ── Constants ─────────────────────────────────────────────────────────────────
INITIAL_BALANCE         =      0.0   # XFA balance starts at $0 (net P&L)
MAX_TRAILING_DD         = 2_000.0
MAX_CONTRACTS           = 50
POINT_VALUE             = 5.0
COMMISSION_PER_LEG      = 0.37
SLIPPAGE_BASE           = 0.25
SLIPPAGE_SCALE          = 0.02
OR_MINUTES              = 30
QUALIFYING_DAYS_NEEDED  = 3
FUNDED_CONSISTENCY_PCT  = 0.40
MIN_PROFIT_FOR_PAYOUT   = 250.0
MAX_POST_BARS           = 450
ENTRY_SLIPPAGE          = 0.25

N_OBS_V3 = 11
FIXED_TP  = 6
FIXED_SL  = 18

# ── Scaling Plan ($50K XFA on TopstepX) ───────────────────────────────────────
SCALING_PLAN_50K = [
    (    0,  10),   # $0+    → 10 MES
    (  500,  20),   # $500+  → 20 MES
    ( 1000,  30),   # $1000+ → 30 MES
    ( 1500,  40),   # $1500+ → 40 MES
    ( 2000,  50),   # $2000+ → 50 MES
]


def _leg_cost(n: int) -> float:
    return (n * COMMISSION_PER_LEG
            + n * SLIPPAGE_BASE * (1.0 + SLIPPAGE_SCALE * max(0, n - 1)))

def _rt_cost(n: int) -> float:
    return 2.0 * _leg_cost(n)

def _scaling_cap(balance: float, plan=SCALING_PLAN_50K) -> int:
    cap = plan[0][1]
    for threshold, contracts in plan:
        if balance >= threshold:
            cap = contracts
        else:
            break
    return cap


# ── Precompute ORB outcomes ────────────────────────────────────────────────────

def precompute_orb(split: str) -> str:
    cache = os.path.join(_DATA, f"fixed_orb_{split}.npz")
    if os.path.exists(cache):
        return cache

    print(f"  Precomputing ORB data ({split})...")
    df = pd.read_csv(os.path.join(_DATA, "preprocessed_mes.csv"))
    df = df[df["close"] > 1000].copy()
    df["ct_hour"]   = (df["hour"] * 23.0).round().astype(int)
    df["ct_minute"] = pd.to_datetime(df["ts_event"]).dt.minute

    all_dates = np.sort(df["trading_date"].unique())
    n_total   = len(all_dates)
    bounds    = {"train":   (0,                    int(n_total * 0.70)),
                 "test":    (int(n_total * 0.70),  int(n_total * 0.85)),
                 "holdout": (int(n_total * 0.85),  n_total)}
    lo, hi    = bounds[split]
    dates     = all_dates[lo:hi]
    n_days    = len(dates)

    date_set  = set(dates)
    grp       = dict(list(df[df["trading_date"].isin(date_set)]
                          .groupby("trading_date")))

    has_trade    = np.zeros(n_days, dtype=bool)
    direction    = np.zeros(n_days, dtype=np.int8)
    entry_price  = np.zeros(n_days, dtype=np.float32)
    or_range_pct = np.zeros(n_days, dtype=np.float32)
    n_post_bars  = np.zeros(n_days, dtype=np.int32)
    post_highs   = np.zeros((n_days, MAX_POST_BARS), dtype=np.float32)
    post_lows    = np.zeros((n_days, MAX_POST_BARS), dtype=np.float32)
    post_closes  = np.zeros((n_days, MAX_POST_BARS), dtype=np.float32)
    post_ct_h    = np.zeros((n_days, MAX_POST_BARS), dtype=np.int8)
    post_ct_m    = np.zeros((n_days, MAX_POST_BARS), dtype=np.int8)

    skipped_ambiguous = 0

    for di, date in enumerate(dates):
        day_df = grp.get(date)
        if day_df is None or len(day_df) < 100:
            continue

        ct_h   = day_df["ct_hour"].values
        ct_m   = day_df["ct_minute"].values
        highs  = day_df["high"].values
        lows   = day_df["low"].values
        closes = day_df["close"].values

        rth_mask = (
            ((ct_h == 8) & (ct_m >= 30)) |
            ((ct_h >= 9) & (ct_h <= 14)) |
            ((ct_h == 15) & (ct_m < 10))
        )
        rth_idx = np.where(rth_mask)[0]
        if len(rth_idx) < OR_MINUTES + 30:
            continue

        or_idx = rth_idx[:OR_MINUTES]
        or_h   = highs[or_idx].max()
        or_l   = lows[or_idx].min()
        or_rng = or_h - or_l
        if or_rng < 0.25:
            continue

        entry = direction_d = entry_bar = None
        for i in rth_idx[OR_MINUTES:]:
            if ct_h[i] == 14 and ct_m[i] >= 30:
                break
            if ct_h[i] >= 15:
                break

            h_above = highs[i] > or_h
            l_below = lows[i] < or_l

            if h_above and l_below:
                skipped_ambiguous += 1
                continue

            if h_above:
                entry = or_h + ENTRY_SLIPPAGE
                direction_d = 1
                entry_bar = i
                break
            elif l_below:
                entry = or_l - ENTRY_SLIPPAGE
                direction_d = -1
                entry_bar = i
                break

        if entry is None:
            continue

        post = np.arange(entry_bar + 1, len(highs))
        n_pb = min(len(post), MAX_POST_BARS)
        if n_pb == 0:
            continue

        has_trade[di]    = True
        direction[di]    = direction_d
        entry_price[di]  = entry
        or_range_pct[di] = or_rng / entry * 100.0
        n_post_bars[di]  = n_pb
        idx              = post[:n_pb]
        post_highs[di, :n_pb]  = highs[idx]
        post_lows[di, :n_pb]   = lows[idx]
        post_closes[di, :n_pb] = closes[idx]
        post_ct_h[di, :n_pb]   = ct_h[idx]
        post_ct_m[di, :n_pb]   = ct_m[idx]

    n_trades = int(has_trade.sum())
    print(f"  {n_trades}/{n_days} trade days "
          f"({skipped_ambiguous} ambiguous bars skipped). Saved -> {cache}")
    np.savez_compressed(cache,
        has_trade=has_trade, direction=direction,
        entry_price=entry_price, or_range_pct=or_range_pct,
        n_post_bars=n_post_bars,
        post_highs=post_highs, post_lows=post_lows, post_closes=post_closes,
        post_ct_h=post_ct_h, post_ct_m=post_ct_m)
    return cache


def precompute_orb_all() -> str:
    """Concatenate all three splits into one combined ORB cache."""
    cache = os.path.join(_DATA, "fixed_orb_all.npz")
    if os.path.exists(cache):
        return cache
    arrays = {}
    for split in ["train", "test", "holdout"]:
        precompute_orb(split)
        d = np.load(os.path.join(_DATA, f"fixed_orb_{split}.npz"))
        arrays[split] = d
    keys = ["has_trade", "direction", "entry_price", "or_range_pct",
            "n_post_bars", "post_highs", "post_lows", "post_closes",
            "post_ct_h", "post_ct_m"]
    combined = {k: np.concatenate([arrays[s][k] for s in ["train", "test", "holdout"]], axis=0)
                for k in keys}
    n = int(combined["has_trade"].sum())
    print(f"  Combined ORB cache: {n}/{len(combined['has_trade'])} trade days -> {cache}")
    np.savez_compressed(cache, **combined)
    return cache


def precompute_orb_oos() -> str:
    """Concatenate test + holdout splits (out-of-sample) into one ORB cache."""
    cache = os.path.join(_DATA, "fixed_orb_oos.npz")
    if os.path.exists(cache):
        return cache
    arrays = {}
    for split in ["test", "holdout"]:
        precompute_orb(split)
        d = np.load(os.path.join(_DATA, f"fixed_orb_{split}.npz"))
        arrays[split] = d
    keys = ["has_trade", "direction", "entry_price", "or_range_pct",
            "n_post_bars", "post_highs", "post_lows", "post_closes",
            "post_ct_h", "post_ct_m"]
    combined = {k: np.concatenate([arrays[s][k] for s in ["test", "holdout"]], axis=0)
                for k in keys}
    n = int(combined["has_trade"].sum())
    print(f"  OOS ORB cache: {n}/{len(combined['has_trade'])} trade days -> {cache}")
    np.savez_compressed(cache, **combined)
    return cache


# ── VIX helpers ───────────────────────────────────────────────────────────────

def _load_vix_lookup() -> dict:
    path = os.path.join(_DATA, "vix.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["date"].astype(str), df["vix_close"].astype(float)))


def _build_vix_ma(vix_map: dict, window: int = 14) -> dict:
    if not vix_map:
        return {}
    s = pd.Series(vix_map).sort_index()
    return s.rolling(window, min_periods=1).mean().to_dict()


# ── Environment ───────────────────────────────────────────────────────────────

class FundedOrbEnvV3(gym.Env):
    """
    Funded ORB env: PPO controls contract sizing, fixed TP=6/SL=18.
    VIX included in observation for regime-aware sizing.
    """
    metadata = {"render_modes": []}

    def __init__(self, split: str = "train", scaling_plan=SCALING_PLAN_50K):
        super().__init__()
        self.fixed_tp     = FIXED_TP
        self.fixed_sl     = FIXED_SL
        self.scaling_plan = scaling_plan

        if split == "all":
            cache = precompute_orb_all()
        elif split == "oos":
            cache = precompute_orb_oos()
        else:
            cache = precompute_orb(split)
        d = np.load(cache)
        self.has_trade    = d["has_trade"]
        self.direction    = d["direction"].astype(np.int32)
        self.entry_price  = d["entry_price"]
        self.or_range_pct = d["or_range_pct"]
        self.n_post_bars  = d["n_post_bars"].astype(np.int32)
        self.post_highs   = d["post_highs"]
        self.post_lows    = d["post_lows"]
        self.post_closes  = d["post_closes"]
        self.post_ct_h    = d["post_ct_h"].astype(np.int32)
        self.post_ct_m    = d["post_ct_m"].astype(np.int32)
        self.n_days       = len(self.has_trade)

        df_dates = pd.read_csv(os.path.join(_DATA, "preprocessed_mes.csv"),
                               usecols=["trading_date", "close"])
        df_dates = df_dates[df_dates["close"] > 1000]
        all_dates = np.sort(df_dates["trading_date"].unique())
        n_total = len(all_dates)
        bounds = {"train":   (0,                    int(n_total * 0.70)),
                  "test":    (int(n_total * 0.70),  int(n_total * 0.85)),
                  "holdout": (int(n_total * 0.85),  n_total),
                  "all":     (0,                    n_total),
                  "oos":     (int(n_total * 0.70),  n_total)}
        lo, hi = bounds[split]
        self.trading_dates = all_dates[lo:hi]

        vix_map = _load_vix_lookup()
        self._vix    = vix_map
        self._vix_ma = _build_vix_ma(vix_map)

        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(N_OBS_V3,), dtype=np.float32)
        self._rng = np.random.default_rng()

    def _skip_to_trade_day(self):
        while (self.current_day < self.n_days
               and not self.has_trade[self.current_day]):
            self.current_day += 1

    def _get_vix_features(self, di: int):
        if di >= len(self.trading_dates):
            return 0.5, 1.0
        date = self.trading_dates[di]
        vix_c  = self._vix.get(date, 20.0)
        vix_ma = self._vix_ma.get(date, 20.0)
        vix_norm  = float(np.clip(vix_c  / 30.0, 0.0, 3.0))
        vix_ratio = float(np.clip(vix_c / max(vix_ma, 1.0), 0.5, 2.0))
        return vix_norm, vix_ratio

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        opts = options or {}
        init_bal   = opts.get("initial_balance", INITIAL_BALANCE)
        init_floor = opts.get("trailing_floor",  init_bal - MAX_TRAILING_DD)

        self.balance        = init_bal
        self.trailing_floor = init_floor
        self.base_balance   = init_bal
        self.best_day_pnl   = 0.0
        self.days_traded    = 0

        max_start = max(0, self.n_days - 10)
        self.current_day = int(self._rng.integers(0, max_start + 1))
        self._skip_to_trade_day()
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        profit     = self.balance - INITIAL_BALANCE
        headroom   = (self.balance - self.trailing_floor) / MAX_TRAILING_DD
        new_profit = self.balance - self.base_balance
        if new_profit > 0 and self.best_day_pnl > 0:
            c_danger = self.best_day_pnl / (new_profit * FUNDED_CONSISTENCY_PCT + 1e-8)
        else:
            c_danger = 0.0
        days_prog  = self.days_traded / QUALIFYING_DAYS_NEEDED
        profit_gap = (max(0.0, MIN_PROFIT_FOR_PAYOUT - max(0.0, new_profit))
                      / MIN_PROFIT_FOR_PAYOUT)

        di = self.current_day
        if di < self.n_days and self.has_trade[di]:
            bdir   = float(self.direction[di])
            or_rp  = float(self.or_range_pct[di]) / 10.0
            ep     = float(self.entry_price[di])
            or_pts = ep * float(self.or_range_pct[di]) / 100.0
            vix_n, vix_r = self._get_vix_features(di)
        else:
            bdir, or_rp, or_pts = 0.0, 0.0, 0.0
            vix_n, vix_r = 0.5, 1.0

        return np.array([
            np.clip(headroom, -1.0, 3.0),
            np.clip(profit / 1000.0, -5.0, 20.0),
            np.clip(c_danger, 0.0, 5.0),
            np.clip(days_prog, 0.0, 3.0),
            profit_gap,
            np.clip(self.best_day_pnl / 1000.0, 0.0, 20.0),
            bdir,
            np.clip(or_rp, 0.0, 5.0),
            np.clip(or_pts / 20.0, 0.0, 5.0),
            vix_n,
            vix_r,
        ], dtype=np.float32)

    def _simulate_trade(self, n: int) -> float:
        di    = self.current_day
        entry = float(self.entry_price[di])
        dir_  = int(self.direction[di])
        n_pb  = int(self.n_post_bars[di])
        tp, sl = self.fixed_tp, self.fixed_sl
        cost  = _rt_cost(n)
        tp_p  = entry + dir_ * tp
        sl_p  = entry - dir_ * sl
        ph, pl, pc = self.post_highs[di], self.post_lows[di], self.post_closes[di]
        ch, cm = self.post_ct_h[di], self.post_ct_m[di]
        for j in range(n_pb):
            hj, mj = ch[j], cm[j]
            if hj > 15 or (hj == 15 and mj >= 10):
                ep = pc[j-1] if j > 0 else entry
                return dir_ * (ep - entry) * n * POINT_VALUE - cost
            if dir_ == 1:
                if pl[j] <= sl_p: return -sl * n * POINT_VALUE - cost
                if ph[j] >= tp_p: return  tp * n * POINT_VALUE - cost
            else:
                if ph[j] >= sl_p: return -sl * n * POINT_VALUE - cost
                if pl[j] <= tp_p: return  tp * n * POINT_VALUE - cost
        ep = pc[n_pb-1] if n_pb > 0 else entry
        return dir_ * (ep - entry) * n * POINT_VALUE - cost

    def _payout_eligible(self) -> bool:
        new_profit = self.balance - self.base_balance
        if self.days_traded < QUALIFYING_DAYS_NEEDED: return False
        if new_profit < MIN_PROFIT_FOR_PAYOUT:        return False
        if self.best_day_pnl <= 0:                    return True
        return (self.best_day_pnl / new_profit) <= FUNDED_CONSISTENCY_PCT

    def step(self, action):
        di = self.current_day
        if di >= self.n_days:
            self.current_day = 0
            self._skip_to_trade_day()
            di = self.current_day

        frac = float(np.clip(action[0], 0.0, 1.0))
        n = int(round(frac * MAX_CONTRACTS))
        n = max(0, min(n, MAX_CONTRACTS))
        if self.scaling_plan is not None and n > 0:
            n = min(n, _scaling_cap(self.balance, self.scaling_plan))

        pnl = self._simulate_trade(n) if (n > 0 and self.has_trade[di]) else 0.0

        prev_days = self.days_traded
        if n > 0:
            self.days_traded += 1

        self.balance += pnl
        new_floor = self.balance - MAX_TRAILING_DD
        self.trailing_floor = min(max(self.trailing_floor, new_floor), INITIAL_BALANCE)
        if pnl > self.best_day_pnl:
            self.best_day_pnl = pnl

        reward = pnl / (MAX_TRAILING_DD * 2)
        if self.days_traded > prev_days:
            reward += 0.25

        new_profit = self.balance - self.base_balance
        if new_profit > 1e-6 and self.best_day_pnl > 0:
            c_ratio = self.best_day_pnl / (new_profit * FUNDED_CONSISTENCY_PCT)
            if c_ratio > 1.0:
                reward -= 0.02 * min(c_ratio - 1.0, 3.0)

        if self.balance <= self.trailing_floor:
            reward -= 1.5
            return self._obs(), reward, True, False, {"reason": "blowup"}

        if self._payout_eligible():
            payout = min(max(self.balance, 0.0) * 0.50, 6000.0)
            reward += 5.0 + payout / 1200.0
            return self._obs(), reward, True, False, \
                   {"reason": "payout", "payout": payout}

        self.current_day += 1
        self._skip_to_trade_day()
        if self.current_day >= self.n_days:
            self.current_day = 0
            self._skip_to_trade_day()
        return self._obs(), reward, False, False, {}

    def render(self):
        pass
