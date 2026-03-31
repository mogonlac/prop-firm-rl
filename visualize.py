"""
visualize.py -- Monte Carlo visualization for the funded account.

The challenge (3-5 binary-outcome steps) cannot produce a meaningful Monte Carlo
line chart, so only the funded window 1 is shown as equity curves.

Usage:
  python visualize.py
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_HERE    = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_HERE, "results")
_PLOTS   = os.path.join(_HERE, "results", "plots")

DARK_BG  = "#ffffff"
GRID_COL = "#dddddd"
TEXT_COL = "#111111"
GREEN    = "#2a9d3f"
RED      = "#d62728"
YELLOW   = "#e6a817"


def set_style():
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor":   DARK_BG,
        "axes.edgecolor":   GRID_COL,
        "axes.labelcolor":  TEXT_COL,
        "axes.titlecolor":  TEXT_COL,
        "xtick.color":      TEXT_COL,
        "ytick.color":      TEXT_COL,
        "grid.color":       GRID_COL,
        "grid.linewidth":   0.5,
        "text.color":       TEXT_COL,
        "legend.facecolor": "#f5f5f5",
        "legend.edgecolor": GRID_COL,
        "font.family":      "Inter",
        "font.size":        10,
    })



DISPLAY_EPISODES = 2000

def compress_skips(balances: np.ndarray) -> np.ndarray:
    mask = np.concatenate([[True], np.diff(balances) != 0.0])
    return balances[mask]

def load_funded(seed: int = 42):
    df = pd.read_csv(os.path.join(_RESULTS, "funded_equity.csv"))
    id_col = "episode_id" if "episode_id" in df.columns else "attempt_id"
    episodes = {}
    groups = df.groupby(id_col) if id_col == "episode_id" \
             else df[df["window"] == 1].groupby(id_col)
    rng2 = np.random.default_rng(seed + 1)
    for eid, grp in groups:
        bal = grp.sort_values("day")["balance"].values.astype(float)
        bal = bal[:101]
        # add tiny random walk on flat (skip) days so paths diverge visually
        pnl = np.diff(bal)
        skip = pnl == 0.0
        pnl[skip] += rng2.normal(0, 25.0, size=skip.sum())
        bal = np.concatenate([[bal[0]], bal[0] + np.cumsum(pnl)])
        if len(bal) > 1:
            episodes[int(eid)] = bal
    # random subsample
    rng = np.random.default_rng(seed)
    keys = list(episodes.keys())
    chosen = rng.choice(keys, size=min(DISPLAY_EPISODES, len(keys)), replace=False)
    return {k: episodes[k] for k in chosen}


def trailing_floor(bal: np.ndarray) -> np.ndarray:
    """Compute per-step trailing floor: running_max(balance) - $2000, capped at $0."""
    floor = np.full(len(bal), -2000.0)
    peak  = bal[0]
    for i, b in enumerate(bal):
        if b > peak:
            peak = b
        floor[i] = min(peak - 2000.0, 0.0)
    return floor


def median_floor(episodes: dict) -> np.ndarray:
    """Median trailing floor across all episodes, padded to max length."""
    max_len = max(len(v) for v in episodes.values())
    floors  = []
    for bal in episodes.values():
        f = trailing_floor(bal)
        # pad with final value so shorter episodes don't drag median down
        f = np.pad(f, (0, max_len - len(f)), constant_values=f[-1])
        floors.append(f)
    return np.median(floors, axis=0)


def plot_funded_w1(episodes: dict, save_path: str):
    all_ids    = list(episodes.keys())
    payout_set = {k for k, v in episodes.items() if v[-1] > 200}
    n_p = len(payout_set)
    n_b = len(all_ids) - n_p
    n   = len(all_ids)

    rng_col = np.random.default_rng(0)
    tab20 = plt.get_cmap("tab20").colors

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(
        f"Funded Account  ·  Window 1 Monte Carlo  "
        f"({n_p} payout  /  {n_b} blowup)  ·  {n} episodes",
        fontsize=13, weight="bold"
    )

    for aid in all_ids:
        if aid not in payout_set:
            ax.plot(episodes[aid], color=RED, linewidth=0.4, alpha=0.6)
    for aid in all_ids:
        if aid in payout_set:
            col = tab20[int(rng_col.integers(0, len(tab20)))]
            ax.plot(episodes[aid], color=col, linewidth=0.4, alpha=0.6)

    # dynamic trailing floor (median across blowup episodes only)
    blowup_eps = {k: v for k, v in episodes.items() if k not in payout_set}
    if blowup_eps:
        med_floor = median_floor(blowup_eps)
        ax.plot(med_floor, color=YELLOW, linewidth=2.8, linestyle="--",
                label=f"Median trailing floor  (blowup only, n={len(blowup_eps)})", zorder=5)

    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    from matplotlib.collections import LineCollection
    from matplotlib.legend_handler import HandlerBase

    class RainbowHandler(HandlerBase):
        def create_artists(self, legend, orig, x0, y0, width, height, fontsize, trans):
            colors = ["#e6194b","#f58231","#ffe119","#3cb44b","#4363d8","#911eb4"]
            n = len(colors)
            segs = [[(x0 + i*width/n, y0+height/2), (x0+(i+1)*width/n, y0+height/2)]
                    for i in range(n)]
            lc = LineCollection(segs, colors=colors, linewidth=2.5, transform=trans)
            return [lc]

    payout_line = Line2D([0], [0], label=f"Payout  ({n_p})")
    blowup_line = Line2D([0], [0], color=RED, linewidth=2.5, label=f"Blowup  ({n_b})")
    ax.axhline(0, color=TEXT_COL, linestyle="--", linewidth=0.9, alpha=0.5)

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Net Balance ($)")
    ax.grid(True, axis="y")
    floor_handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles=[payout_line, blowup_line] + floor_handles,
        handler_map={payout_line: RainbowHandler()},
        framealpha=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(_PLOTS, exist_ok=True)
    set_style()

    print("Loading funded_equity.csv...")
    episodes = load_funded(seed=args.seed)
    print(f"  {len(episodes)} window-1 episodes loaded")

    print("Rendering chart...")
    plot_funded_w1(episodes, os.path.join(_PLOTS, "funded_w1_montecarlo.png"))
    print("Done.")


if __name__ == "__main__":
    main()
