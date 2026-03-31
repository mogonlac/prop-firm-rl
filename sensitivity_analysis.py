"""
sensitivity_analysis.py — EV sensitivity to pass rate and payout rate degradation.

Shows how EV/attempt changes as challenge pass rate and funded payout rate
decline from their holdout backtest values.

Usage:
  python sensitivity_analysis.py
Output:
  results/plots/sensitivity_ev.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Holdout backtest values ────────────────────────────────────────────────────
PASS_RATE_BASE   = 0.584
PAYOUT_RATE_BASE = 0.483
AVG_NET_PAYOUT   = 1_358.0   # avg net payout W1
CHALLENGE_SUB    =    49.0   # per attempt (one subscription per attempt)
ACTIVATION_FEE   =   149.0   # per funded account activated

# ── Topstep 2024 industry average (source: Topstep 2024 Trader Statistics) ────
# 12.4% passed Combine; 3.5% of all traders received payout
# => conditional payout rate = 3.5% / 12.4% = 28.2%
IND_PASS_RATE   = 0.124
IND_PAYOUT_RATE = 0.035 / 0.124   # ~28.2%


def ev_per_attempt(pass_rate, payout_rate, avg_payout=AVG_NET_PAYOUT):
    return (pass_rate * payout_rate * avg_payout
            - CHALLENGE_SUB
            - pass_rate * ACTIVATION_FEE)


# ── Grid ───────────────────────────────────────────────────────────────────────
pass_rates   = np.round(np.arange(0.10, 0.70, 0.05), 2)
payout_rates = np.round(np.arange(0.15, 0.60, 0.05), 2)

ev_grid = np.array([
    [ev_per_attempt(pr, pay) for pay in payout_rates]
    for pr in pass_rates
])

# ── Print table ────────────────────────────────────────────────────────────────
col_w = 8
col_header = "Pass / Pay"
header = f"{col_header:>10}" + "".join(f"{p:>{col_w}.0%}" for p in payout_rates)
print("=" * len(header))
print("  EV PER ATTEMPT  (avg net payout = ${:,.0f})".format(AVG_NET_PAYOUT))
print("=" * len(header))
print(header)
print("-" * len(header))
for i, pr in enumerate(pass_rates):
    row = f"{pr:>10.0%}"
    for j, pay in enumerate(payout_rates):
        val = ev_grid[i, j]
        marker = "*" if (abs(pr - PASS_RATE_BASE) < 0.025 and
                         abs(pay - PAYOUT_RATE_BASE) < 0.025) else " "
        cell = f"${val:,.0f}{marker}" if val >= 0 else f"-${abs(val):,.0f}{marker}"
        row += f"{cell:>{col_w}}"
    print(row)
print("-" * len(header))
print(f"  * = current operating point  "
      f"(pass={PASS_RATE_BASE:.0%}, payout={PAYOUT_RATE_BASE:.0%})")
print(f"  Negative values = losing proposition at that combination.")
print("=" * len(header))

# ── Heatmap ────────────────────────────────────────────────────────────────────
plt.rcParams["font.family"] = "Inter"
fig, ax = plt.subplots(figsize=(11, 7.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

norm = TwoSlopeNorm(vmin=ev_grid.min(), vcenter=25, vmax=ev_grid.max())
im = ax.imshow(
    ev_grid,
    aspect="auto",
    origin="lower",
    extent=[payout_rates[0] - 0.025, payout_rates[-1] + 0.025,
            pass_rates[0]   - 0.025, pass_rates[-1]   + 0.025],
    cmap="RdYlGn",
    norm=norm,
)

# Annotate cells
for i, pr in enumerate(pass_rates):
    for j, pay in enumerate(payout_rates):
        val = ev_grid[i, j]
        txt = f"${val:,.0f}" if val >= 0 else f"-${abs(val):,.0f}"
        ax.text(pay, pr, txt, ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold",
                path_effects=[pe.withStroke(linewidth=0.8, foreground="black")])

ev_current  = ev_per_attempt(PASS_RATE_BASE, PAYOUT_RATE_BASE)
ev_industry = ev_per_attempt(IND_PASS_RATE, IND_PAYOUT_RATE)


# ── Colorbar ───────────────────────────────────────────────────────────────────
vmin_val = int(np.floor(ev_grid.min() / 25) * 25)
vmax_val = int(np.ceil(ev_grid.max()  / 25) * 25)
cb_ticks = list(range(vmin_val, vmax_val + 1, 25))
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=cb_ticks)
cb.set_label("EV per attempt ($)", color="black", fontsize=10)
cb.ax.yaxis.set_tick_params(color="black")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="black")

# ── Labels & title ─────────────────────────────────────────────────────────────
ax.set_xlabel("Funded payout rate", color="black", fontsize=11)
ax.set_ylabel("Challenge pass rate", color="black", fontsize=11, labelpad=12)
ax.set_title(
    r"EV per attempt — sensitivity to pass & payout rate degradation"
    "\n"
    rf"avg net payout = \${AVG_NET_PAYOUT:,.0f}  |  sub = \${CHALLENGE_SUB:.0f}/attempt  (activation fee included)",
    color="black", fontsize=10, pad=12,
)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.tick_params(colors="black")
for spine in ax.spines.values():
    spine.set_edgecolor("#aaa")


plt.tight_layout()
out = os.path.join(_HERE, "results", "plots", "sensitivity_ev.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\nSaved -> {out}")
plt.show()
