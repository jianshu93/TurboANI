#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MPLCONFIG_DIR = SCRIPT_DIR / "mplconfig"
XDG_CACHE_DIR = SCRIPT_DIR / "cache"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


METHODS = [
    "fastANI",
    "superani",
    "TurboANI-sensitive",
    "TurboANI-default",
    "TurboANI-fast",
]
COLORS = {
    "fastANI": "#7FC97F",
    "superani": "#386CB0",
    "TurboANI-sensitive": "#E41A1C",
    "TurboANI-default": "#BEAED4",
    "TurboANI-fast": "#FDC086",
}
MARKERS = {
    "fastANI": "o",
    "superani": "D",
    "TurboANI-sensitive": "^",
    "TurboANI-default": "s",
    "TurboANI-fast": "v",
}
LABELS = {
    "fastANI": "fastANI",
    "superani": "skani",
    "TurboANI-sensitive": "TurboANI sensitive",
    "TurboANI-default": "TurboANI default",
    "TurboANI-fast": "TurboANI fast",
}
PANEL_GROUP = "All pairs"


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
            "font.size": 20,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 16,
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


def plot_speedup_vs_recovery(summary: pd.DataFrame, out_pdf: Path) -> None:
    summary = summary[summary["tolerance"].eq(1.0)].copy()
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(6.8, 6.2))

    max_x = max(10.0, float(summary["ani_recovery"].max() * 100.0) * 1.12)
    max_y = float(summary["speedup_vs_fastani"].max())
    y_ticks = [1, 3, 10, 30, 100]

    panel = summary[summary["orthoani_group"].eq(PANEL_GROUP)].set_index("method").loc[METHODS]
    truth_pairs = int(panel["truth_pairs"].iloc[0])
    for method, row in panel.iterrows():
        x = float(row["ani_recovery"]) * 100.0
        y = float(row["speedup_vs_fastani"])
        ax.scatter(
            x,
            y,
            s=150,
            marker=MARKERS[method],
            color=COLORS[method],
            edgecolor="black",
            linewidth=0.9,
            zorder=3,
        )
        ax.annotate(
            LABELS[method],
            xy=(x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=11.5,
            color="black",
        )

    ax.set_title(f"{PANEL_GROUP}\nn={truth_pairs:,}")
    ax.set_xlim(0, max_x)
    ax.set_yscale("log")
    ax.set_ylim(0.8, max(115.0, max_y * 1.25))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{tick:g}x" for tick in y_ticks])
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlabel("ANI recovery within +/-1% (%)")
    ax.set_ylabel("Speedup vs fastANI at 12,000 genomes")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[method],
            color="none",
            markerfacecolor=COLORS[method],
            markeredgecolor="black",
            markeredgewidth=0.9,
            markersize=9,
            label=LABELS[method],
        )
        for method in METHODS
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        ncol=1,
        frameon=False,
        handletextpad=0.4,
    )
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot speedup versus ANI recovery for the +/-1% recovery threshold only."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=SCRIPT_DIR / "ani_speed_accuracy_12000_scaled_summary.tsv",
    )
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary, sep="\t")
    plot_data = summary[
        summary["tolerance"].eq(1.0) & summary["orthoani_group"].eq(PANEL_GROUP)
    ].copy()
    data_path = args.out_dir / "speedup_vs_ani_recovery_tol1_panelA.tsv"
    pdf_path = args.out_dir / "speedup_vs_ani_recovery_tol1_panelA.pdf"
    plot_data.to_csv(data_path, sep="\t", index=False, float_format="%.6f")
    plot_speedup_vs_recovery(summary, pdf_path)
    print(data_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
