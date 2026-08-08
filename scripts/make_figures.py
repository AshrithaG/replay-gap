#!/usr/bin/env python3
"""Paper figures from results/*.divergence.json. Outputs PDF to paper/figures/.

    python scripts/make_figures.py
"""

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito (colorblind-safe): control = blue, swap = vermillion.
C_CONTROL, C_SWAP = "#0072B2", "#D55E00"

RUNS = {  # run name -> (direction, control alias)
    "pilot30": ("up", "small"),
    "easy": ("up", "small"),
    "nudge": ("up", "small"),
    "pilot30_rev": ("down", "large"),
    "easy_rev": ("down", "large"),
    "nudge_rev": ("down", "large"),
}


def load_rows():
    rows = []
    for run, (direction, control) in RUNS.items():
        data = json.loads((ROOT / "results" / f"{run}.divergence.json").read_text())
        # Derive early/late per (instance, alias): smaller fork_step = early.
        by_pair = defaultdict(list)
        for r in data:
            by_pair[(r["instance_id"], r["alias"])].append(r)
        for pair_rows in by_pair.values():
            pair_rows.sort(key=lambda r: r["fork_step"])
            for i, r in enumerate(pair_rows):
                r["bucket"] = "early" if (len(pair_rows) == 1 or i == 0) else "late"
                r["role"] = "control" if r["alias"] == control else "swap"
                r["direction"] = direction
                r["run"] = run
                rows.append(r)
    return rows


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)


def fig1(rows):
    """Grouped bars: mean edit distance by fork position; control vs swap; per direction."""
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.5), sharey=True)
    for ax, direction, title in zip(axes, ["up", "down"], ["Swap up (4B base → 14B)", "Swap down (14B base → 4B)"]):
        for xi, bucket in enumerate(["early", "late"]):
            for off, role, color in [(-0.19, "control", C_CONTROL), (0.19, "swap", C_SWAP)]:
                vals = [r["normalized_edit_distance"] for r in rows if r["direction"] == direction and r["bucket"] == bucket and r["role"] == role]
                m = mean(vals)
                ax.bar(xi + off, m, width=0.34, color=color, zorder=2)
                ax.text(xi + off, m + 0.02, f"{m:.2f}", ha="center", fontsize=8, color="#444444")
        ax.set_xticks([0, 1], ["early fork (30%)", "late fork (70%)"])
        ax.set_title(title, fontsize=9.5)
        ax.set_ylim(0, 1.05)
        style(ax)
    axes[0].set_ylabel("normalized action\nedit distance", fontsize=9)
    fig.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=C_CONTROL), plt.Rectangle((0, 0), 1, 1, color=C_SWAP)],
        labels=["same-model control", "model swap"],
        loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.13),
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig1_divergence.pdf", bbox_inches="tight")
    print("fig1 saved")


def fig2(rows):
    """ECDF of first divergent post-fork action, control vs swap (pooled)."""
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    for role, color in [("control", C_CONTROL), ("swap", C_SWAP)]:
        sub = [r for r in rows if r["role"] == role]
        # None = never diverged -> censored at the end (treat as beyond max).
        xs = sorted((r["first_divergent_action"] if r["first_divergent_action"] is not None else 10**6) for r in sub)
        n = len(xs)
        grid = list(range(0, 26))
        ecdf = [sum(1 for x in xs if x <= g) / n for g in grid]
        ax.step(grid, ecdf, where="post", color=color, linewidth=2,
                label=f"{'same-model control' if role == 'control' else 'model swap'}")
    ax.set_xlabel("first divergent action after fork", fontsize=9)
    ax.set_ylabel("fraction of branches\ndiverged", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_firstdiv.pdf", bbox_inches="tight")
    print("fig2 saved")


def fig3(rows):
    """Replay validity: fraction of the post-fork trajectory that replay gets right."""
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    labels, vals, colors = [], [], []
    for role, color in [("control", C_CONTROL), ("swap", C_SWAP)]:
        for bucket in ["early", "late"]:
            sub = [r for r in rows if r["role"] == role and r["bucket"] == bucket and r["n_actions_base"] > 0]
            v = mean(
                min(1.0, (r["first_divergent_action"] if r["first_divergent_action"] is not None else r["n_actions_base"]) / r["n_actions_base"])
                for r in sub
            )
            labels.append(f"{'control' if role == 'control' else 'swap'}\n{bucket}")
            vals.append(v)
            colors.append(color)
    xs = range(len(labels))
    ax.bar(xs, vals, width=0.6, color=colors, zorder=2)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.02, f"{v:.0%}", ha="center", fontsize=8.5, color="#444444")
    ax.set_xticks(list(xs), labels, fontsize=8.5)
    ax.set_ylabel("replay validity\n(post-fork states still correct)", fontsize=9)
    ax.set_ylim(0, 1.05)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_validity.pdf", bbox_inches="tight")
    print("fig3 saved")


if __name__ == "__main__":
    rows = load_rows()
    print(f"{len(rows)} branch rows loaded from {len(RUNS)} runs")
    fig1(rows)
    fig2(rows)
    fig3(rows)
    # Numbers cited in the text:
    for role in ("control", "swap"):
        sub = [r for r in rows if r["role"] == role]
        fd = [r["first_divergent_action"] for r in sub if r["first_divergent_action"] is not None]
        print(f"{role}: n={len(sub)}, mean edit={mean(r['normalized_edit_distance'] for r in sub):.3f}, "
              f"diverged={len(fd)}/{len(sub)}, mean 1st-div={mean(fd):.2f}")
