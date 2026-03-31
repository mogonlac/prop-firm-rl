"""
plot_risk_geometry.py — 3D surface of CHALLENGE pass rate across
(Win Rate, R:R Ratio) parameter space.

Simulates Topstep $50K Trading Combine:
  - Profit target: $3,000
  - Max trailing drawdown: $2,000 (EOD, floor never moves intraday)
  - Consistency: best single day <= 50% of total profit
  - 1 trade per day (ORB), fixed dollar risk per trade

Right panel: pass rate vs win rate slices at several R:R values,
with horizontal EV=0 breakeven line.

Usage:  python plot_risk_geometry.py
Output: presentation/risk_geometry.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'Inter'

OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "plots")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "risk_geometry.png")

# ── Challenge constants (Topstep $50K, TopstepX) ─────────────────────────────
PROFIT_TARGET    = 3_000.0
MAX_DD           = 2_000.0
CONSISTENCY_PCT  = 0.50       # best day <= 50% of total profit
MAX_TRADING_DAYS = 60
RISK_PER_TRADE   = 300.0      # fixed dollar risk per trade (normalised)
COMMISSION       = 0.74       # round-turn MES
N_SIM            = 500

# ── EV breakeven: what challenge pass rate do you need? ───────────────────────
# Cost = $49/mo * ~1.5 months avg + $149 activation on pass
# Revenue on pass = funded EV (from backtest) ~$730 net
# EV = pass_rate * 730 - (49*1.5 + 149*pass_rate) = 0
# pass_rate * (730 - 149) = 73.5  =>  pass_rate = 73.5/581 ~ 12.6%
BREAKEVEN_PASS = 12.6

# ── Parameter grid ────────────────────────────────────────────────────────────
win_rates = np.linspace(0.10, 0.90, 32)
rr_ratios = np.linspace(0.25, 5.0,  32)

# ── Simulation ────────────────────────────────────────────────────────────────
rng          = np.random.default_rng(42)
pass_surface = np.zeros((len(rr_ratios), len(win_rates)))

for i, rr in enumerate(rr_ratios):
    win_pnl  =  rr * RISK_PER_TRADE - COMMISSION
    loss_pnl = -RISK_PER_TRADE       - COMMISSION

    for j, wr in enumerate(win_rates):
        passes = 0
        for _ in range(N_SIM):
            balance  = 0.0
            floor    = -MAX_DD
            day_pnls = []

            for _ in range(MAX_TRADING_DAYS):
                pnl = win_pnl if rng.random() < wr else loss_pnl
                balance += pnl
                day_pnls.append(pnl)

                # EOD trailing floor
                floor = max(floor, balance - MAX_DD)

                # blown
                if balance <= floor:
                    break

                # passed: hit profit target AND consistency OK
                if balance >= PROFIT_TARGET:
                    best = max(day_pnls)
                    if best <= CONSISTENCY_PCT * balance:
                        passes += 1
                        break

        pass_surface[i, j] = passes / N_SIM * 100.0

# ── Figure 1: 3D surface ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 8), facecolor='white')
ax  = fig.add_subplot(111, projection='3d')


WR, RR = np.meshgrid(win_rates * 100, rr_ratios)

norm   = Normalize(vmin=0, vmax=100)
colors = cm.RdYlGn(norm(pass_surface))

ax.plot_surface(WR, RR, pass_surface,
                facecolors=colors, alpha=0.88,
                linewidth=0, antialiased=True, shade=True)

# breakeven plane
ax.plot_surface(WR, RR, np.full_like(pass_surface, BREAKEVEN_PASS),
                alpha=0.15, color='silver', linewidth=0)

# EV=0 curve on surface
ev0_wr, ev0_rr, ev0_z = [], [], []
for j in range(len(win_rates)):
    for i in range(len(rr_ratios) - 1):
        z0, z1 = pass_surface[i, j], pass_surface[i+1, j]
        if (z0 - BREAKEVEN_PASS) * (z1 - BREAKEVEN_PASS) < 0:
            t = (BREAKEVEN_PASS - z0) / (z1 - z0)
            ev0_wr.append(win_rates[j] * 100)
            ev0_rr.append(rr_ratios[i] + t * (rr_ratios[i+1] - rr_ratios[i]))
            ev0_z.append(BREAKEVEN_PASS)

if ev0_wr:
    order = np.argsort(ev0_wr)
    ax.plot(np.array(ev0_wr)[order], np.array(ev0_rr)[order],
            np.array(ev0_z)[order],
            color='black', linewidth=2.5, zorder=10, label='EV=0 breakeven')


mappable = cm.ScalarMappable(cmap='RdYlGn', norm=norm)
mappable.set_array(pass_surface)
cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, pad=0.08)
cbar.set_label('Challenge Pass Rate (%)', fontsize=10)

ax.set_xlabel('Win Rate (%)',        labelpad=10, fontsize=10)
ax.set_ylabel('Reward:Risk Ratio',   labelpad=10, fontsize=10)
ax.set_zlabel('Pass Rate (%)',       labelpad=6,  fontsize=10)
ax.set_title(
    '3D Surface: Challenge Pass Rate Across (R:R, Win Rate)\n'
    'Black curve = EV=0 breakeven',
    fontsize=11, pad=14)
ax.view_init(elev=25, azim=225)
ax.legend(loc='upper left', fontsize=9)


# ── Right panel: 50 contracts, fixed TP, SL scales with R:R ─────────────────
RR_SWEEP     = np.linspace(0.05, 5.0, 80)
N_SIM_2D     = 500    # exactly 500 MC per R:R point
CONTRACTS_2D = 50
TP_PTS_2D    = 6.0    # FIXED TP=6pts → win always $1500
PV           = 5.0
COMM_RT      = 0.74
COMM_TRADE   = COMM_RT * CONTRACTS_2D   # $37 per trade

# Fixed TP=6: win=$1500 always. SL=6/rr: huge at low rr → one loss = instant bust
# pass_rate ≈ wr^N_wins (N=2 no-comm, N=3 with-comm since 2×$1463<$3000)
# → curve tracks blue (win rate) shape directly

from scipy.ndimage import uniform_filter1d

def run_challenge_sim(rr_arr, use_commission):
    results = []
    for rr in rr_arr:
        wr       = 1.0 / (1.0 + rr)
        sl_pts   = TP_PTS_2D / rr           # huge at low rr → loss >> DD
        comm     = COMM_TRADE if use_commission else 0.0
        win_pnl  = TP_PTS_2D * CONTRACTS_2D * PV - comm   # ~$1500 or ~$1463
        loss_pnl = sl_pts    * CONTRACTS_2D * PV + comm   # >> $2000 at low rr
        passes   = 0
        for _ in range(N_SIM_2D):
            balance, floor, day_pnls = 0.0, -MAX_DD, []
            for _ in range(MAX_TRADING_DAYS):
                pnl = win_pnl if rng.random() < wr else -loss_pnl
                balance += pnl
                day_pnls.append(pnl)
                floor = max(floor, balance - MAX_DD)
                if balance <= floor:
                    break
                if balance >= PROFIT_TARGET:
                    if max(day_pnls) <= CONSISTENCY_PCT * balance:
                        passes += 1
                        break
        results.append(passes / N_SIM_2D * 100)
    return uniform_filter1d(np.array(results), size=7)

trade_win_rates_2d = 100.0 / (1.0 + RR_SWEEP)
challenge_pass_2d  = run_challenge_sim(RR_SWEEP, use_commission=False)

# ── Figure 2 ──────────────────────────────────────────────────────────────────
OUT_PATH2 = os.path.join(OUT_DIR, "risk_geometry_2d.png")
fig2, ax2 = plt.subplots(figsize=(9, 7), facecolor='white')

# heatmap = challenge pass rate (green at low R:R, red at high)
heatmap = np.tile(challenge_pass_2d[np.newaxis, :], (100, 1))
ax2.imshow(
    heatmap,
    origin='lower', aspect='auto',
    extent=[RR_SWEEP[0], RR_SWEEP[-1], 0, 100],
    cmap='RdYlGn', vmin=0, vmax=100,
    alpha=0.80,
)

ax2.plot(RR_SWEEP, trade_win_rates_2d, color='steelblue', linewidth=2.5,
         label='Trade win rate  [1/(1+R:R)]')
ax2.plot(RR_SWEEP, challenge_pass_2d,  color='black',     linewidth=2.5,
         label='Challenge pass rate  (500 MC, 50 contracts)')

ax2.set_xlabel('Reward:Risk Ratio', fontsize=10)
ax2.set_ylabel('Rate (%)', fontsize=10)
ax2.set_title(
    'Win Rate & Challenge Pass Rate vs R:R  (50 MES contracts)',
    fontsize=10)
ax2.set_xlim(RR_SWEEP[0], RR_SWEEP[-1])
ax2.set_ylim(0, 100)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2)

fig2.colorbar(
    cm.ScalarMappable(cmap='RdYlGn', norm=Normalize(0, 100)),
    ax=ax2, shrink=0.8, label='Challenge Pass Rate (%)')

fig2.tight_layout()
OUT_PATH2 = os.path.join(OUT_DIR, "risk_geometry_2d.png")
fig2.savefig(OUT_PATH2, dpi=180, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUT_PATH2}")
plt.show()
