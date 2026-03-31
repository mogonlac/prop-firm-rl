"""
plot_flowchart.py — Methodology flowchart for propbot v10 presentation
Output: presentation/flowchart.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.family'] = 'Inter'

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'results', 'plots', 'flowchart.png')

# ── Colors ────────────────────────────────────────────────────────────────────
C_DATA    = '#AED6F1'   # blue      — data
C_SIGNAL  = '#FAD7A0'   # orange    — signal/logic
C_AGENT   = '#D7BDE2'   # purple    — PPO agent
C_SIM     = '#D5D8DC'   # gray      — simulation env
C_DECIDE  = '#F9E79F'   # yellow    — decision
C_OUTPUT  = '#A9DFBF'   # green     — payout/output
C_VALID   = '#A2D9CE'   # teal      — validation
C_CALLOUT = '#FDFEFE'   # white     — callout box
EDGE      = '#2C3E50'   # dark      — borders & text

fig, ax = plt.subplots(figsize=(13, 18), facecolor='white')
ax.set_xlim(0, 13)
ax.set_ylim(0, 18)
ax.axis('off')

# ── Helper functions ──────────────────────────────────────────────────────────
def box(ax, x, y, w, h, text, color, fontsize=10.5, bold=False,
        radius=0.3, sub=None):
    """Draw a rounded rectangle with centered text."""
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f'round,pad=0,rounding_size={radius}',
                          linewidth=1.4, edgecolor=EDGE, facecolor=color,
                          zorder=3)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    yo = y + 0.1 if sub else y
    ax.text(x, yo, text, ha='center', va='center', fontsize=fontsize,
            fontweight=weight, color=EDGE, zorder=4, wrap=True,
            multialignment='center')
    if sub:
        ax.text(x, y - 0.28, sub, ha='center', va='center', fontsize=8.5,
                color='#555555', zorder=4, style='italic')

def diamond(ax, x, y, w, h, text, color=C_DECIDE):
    """Draw a diamond decision shape."""
    dx, dy = w/2, h/2
    pts = [(x, y+dy), (x+dx, y), (x, y-dy), (x-dx, y)]
    diamond_patch = plt.Polygon(pts, closed=True, linewidth=1.4,
                                edgecolor=EDGE, facecolor=color, zorder=3)
    ax.add_patch(diamond_patch)
    ax.text(x, y, text, ha='center', va='center', fontsize=10.5,
            fontweight='bold', color=EDGE, zorder=4)

def arrow(ax, x1, y1, x2, y2, label=None, color=EDGE):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.8, mutation_scale=18), zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.15, my, label, fontsize=9, color=color, va='center')

def dashed_line(ax, x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color='#AAB7B8', linewidth=1.2,
            linestyle='--', zorder=2)

def callout(ax, x, y, w, h, title, lines):
    """Draw a callout box with title + bullet lines."""
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle='round,pad=0,rounding_size=0.2',
                          linewidth=1.2, edgecolor='#AAB7B8',
                          facecolor=C_CALLOUT, zorder=3,
                          linestyle='--')
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.28, title, ha='center', va='top',
            fontsize=9.5, fontweight='bold', color=EDGE, zorder=4)
    for i, line in enumerate(lines):
        ax.text(x + 0.18, y + h - 0.62 - i*0.38, line,
                ha='left', va='top', fontsize=8.5, color='#2C3E50', zorder=4)

# ── Layout: main column at x=5.0, callout at x=8.3 ──────────────────────────
CX = 5.0   # center x of main flow
BW = 3.8   # box width
BH = 0.72  # box height

# Node y positions (top to bottom)
Y = {
    'data':       16.8,
    'orb':        15.4,
    'ppo_chal':   13.9,
    'chal_sim':   12.4,
    'decide':     11.0,
    'ppo_fund':    9.2,
    'fund_sim':    7.8,
    'payout':      6.4,
    'valid':       5.0,
}

# ── Draw nodes ────────────────────────────────────────────────────────────────
box(ax, CX, Y['data'], BW, BH,
    'Historical MES Price Data + VIX',
    C_DATA, bold=True,
    sub='Daily OHLC · Intraday bars · VIX regime signal')

box(ax, CX, Y['orb'], BW, BH,
    'Opening Range Breakout (ORB)',
    C_SIGNAL, bold=True,
    sub='First 30 min RTH → direction signal (long / short)')

box(ax, CX, Y['ppo_chal'], BW, 0.88,
    'Deterministic Agent  —  Challenge Phase',
    C_SIGNAL, bold=True,
    sub='Fixed rule: 50 contracts · TP 5 pts · SL 24 pts')

box(ax, CX, Y['chal_sim'], BW, BH,
    'Challenge Simulation',
    C_SIM, bold=False,
    sub='$3K target · $2K trailing DD · 50% consistency rule')

diamond(ax, CX, Y['decide'], 3.0, 0.9, 'Pass challenge?')

box(ax, CX, Y['ppo_fund'], BW, 0.88,
    'PPO Agent  —  Funded Phase',
    C_AGENT, bold=True,
    sub='New reward: qualifying days · 40% consistency · payout cap')

box(ax, CX, Y['fund_sim'], BW, BH,
    'Funded Simulation',
    C_SIM, bold=False,
    sub='3 qualifying days · 90/10 split · max $6,000 payout')

box(ax, CX, Y['payout'], BW, BH,
    'Payout',
    C_OUTPUT, bold=True,
    sub='min(profit × 90%,  $6,000)  per cycle')

box(ax, CX, Y['valid'], BW, BH,
    'Out-of-Sample Validation',
    C_VALID, bold=True,
    sub='Train / Test / Holdout  —  generalization check')

# ── Arrows (main flow) ────────────────────────────────────────────────────────
arrow(ax, CX, Y['data']    - BH/2, CX, Y['orb']     + BH/2)
arrow(ax, CX, Y['orb']     - BH/2, CX, Y['ppo_chal']+ 0.44)
arrow(ax, CX, Y['ppo_chal']- 0.44, CX, Y['chal_sim'] + BH/2)
arrow(ax, CX, Y['chal_sim']- BH/2, CX, Y['decide']   + 0.45)

# Decision YES → funded
arrow(ax, CX, Y['decide'] - 0.45, CX, Y['ppo_fund'] + 0.44, label='YES')

# Decision NO → loop back left side
no_x = CX - BW/2
ax.annotate('', xy=(no_x, Y['orb'] - BH/2),
            xytext=(no_x, Y['decide']),
            arrowprops=dict(arrowstyle='->', color='#C0392B',
                            lw=1.6, mutation_scale=16,
                            connectionstyle='arc3,rad=0'), zorder=2)
ax.plot([CX - 1.5, no_x], [Y['decide'], Y['decide']],
        color='#C0392B', lw=1.6, zorder=2)
ax.text(no_x - 0.18, Y['decide'], 'NO', ha='right', va='center',
        fontsize=9, color='#C0392B', fontweight='bold')

arrow(ax, CX, Y['ppo_fund']- 0.44, CX, Y['fund_sim'] + BH/2)
arrow(ax, CX, Y['fund_sim'] - BH/2, CX, Y['payout']  + BH/2)
arrow(ax, CX, Y['payout']  - BH/2, CX, Y['valid']    + BH/2)

# ── PPO Callout (right side) ──────────────────────────────────────────────────
CO_X, CO_Y, CO_W, CO_H = 8.1, 8.4, 4.6, 2.35
callout(ax, CO_X, CO_Y, CO_W, CO_H,
        'What is PPO?',
        ['Proximal Policy Optimization — a',
         'reinforcement learning algorithm.',
         '',
         '· Agent explores a simulated env.',
         '· Receives reward based on outcome',
         '· Iteratively improves its policy'])

dashed_line(ax, CX + BW/2, Y['ppo_fund'], CO_X, CO_Y + CO_H/2)

# ── ORB Callout (right side) ──────────────────────────────────────────────────
CO2_X, CO2_Y, CO2_W, CO2_H = 8.1, 14.8, 4.6, 1.5
callout(ax, CO2_X, CO2_Y, CO2_W, CO2_H,
        'Why ORB?',
        ['Price-action based — no ML prediction.',
         'Regime-agnostic → generalizes OOS.',
         'Prevents direction overfitting.'])

dashed_line(ax, CX + BW/2, Y['orb'], CO2_X, CO2_Y + CO2_H/2)

# ── Key insight box (bottom) ──────────────────────────────────────────────────
insight_rect = FancyBboxPatch((0.6, 3.5), 11.8, 0.9,
                              boxstyle='round,pad=0,rounding_size=0.25',
                              linewidth=1.5, edgecolor='#1A5276',
                              facecolor='#D6EAF8', zorder=3)
ax.add_patch(insight_rect)
ax.text(6.5, 3.95,
        'Key insight: the prop firm rules ARE the reward function — '
        'the agent optimizes\ndirectly for the payout structure, not generic P&L.',
        ha='center', va='center', fontsize=9.5, color='#1A5276',
        fontweight='bold', zorder=4, multialignment='center')

plt.tight_layout(pad=0.5)
plt.savefig(OUT_PATH, dpi=180, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUT_PATH}")
plt.show()
