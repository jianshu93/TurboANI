#!/usr/bin/env python3
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


OUT = Path("/Users/jianshuzhao/Documents/Codex/2026-07-23/for/outputs/turboani_thread_logscale_cpu_efficiency_200_20260814_122233")
BLUE = "#377eb8"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        "font.size": 24,
        "axes.titlesize": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 22,
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.grid": False,
        "grid.color": "0.7",
        "grid.linestyle": "--",
        "grid.linewidth": 0.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# Same production anchor used previously:
# 12,000 genomes, 192 threads, effective speedup approximately 187.
n_anchor = 192.0
s_anchor = 187.0
serial = (1 / s_anchor - 1 / n_anchor) / (1 - 1 / n_anchor)

threads = [1, 4, 8, 12, 16, 24, 32, 48, 64, 96]
speedup = [1.0 / (serial + (1.0 - serial) / n) for n in threads]

pd.DataFrame({"threads": threads, "extrapolated_speedup": speedup}).to_csv(
    OUT / "turboani_extrapolated_speedup_points_1_to_96.tsv",
    sep="\t",
    index=False,
)

fig, ax = plt.subplots(figsize=(8.2, 6.0), constrained_layout=True)
ax.plot(
    threads,
    speedup,
    color=BLUE,
    linewidth=3.0,
    zorder=2,
)
ax.scatter(
    threads,
    speedup,
    s=150,
    marker="s",
    color=BLUE,
    edgecolor="black",
    linewidth=0.9,
    zorder=3,
)
ax.set_xlim(0, 100)
ax.set_ylim(0, 102)
ax.set_xticks([1, 4, 8, 12, 16, 24, 32, 48, 64, 96])
ax.tick_params(axis="x", rotation=35)
ax.set_xlabel("Threads")
ax.set_ylabel("Speedup vs 1 thread")
fig.savefig(OUT / "turboani_extrapolated_speedup_points_1_to_96.pdf")
plt.close(fig)

print(OUT / "turboani_extrapolated_speedup_points_1_to_96.pdf")
