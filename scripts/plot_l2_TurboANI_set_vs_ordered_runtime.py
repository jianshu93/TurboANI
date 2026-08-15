from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    exp = Path(__file__).resolve().parent
    df = pd.read_csv(exp / "l2_bitset_vs_ordered_60_timing.tsv", sep="\t")

    labels = ["Bitset L2", "Ordered-map L2"]
    l2 = df["l2_cpu"].to_numpy()
    wall = df["total"].to_numpy()
    windows = int(df["l2_windows"].iloc[0])
    ratio_l2 = l2[1] / l2[0]
    ratio_wall = wall[1] / wall[0]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    colors = ["#7FC97F", "#BEAED4"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.8), constrained_layout=False)
    fig.subplots_adjust(top=0.72, bottom=0.22, left=0.08, right=0.98, wspace=0.32)
    panels = [
        (axes[0], l2, "L2 stage CPU time", "CPU seconds", ratio_l2),
        (axes[1], wall, "End-to-end elapsed time", "Wall seconds", ratio_wall),
    ]

    for ax, vals, title, ylabel, ratio in panels:
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=18)
        ymax = max(vals) * 1.34
        ax.set_ylim(0, ymax)
        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + ymax * 0.025,
                f"{value:.1f}s",
                ha="center",
                va="bottom",
                fontsize=12,
            )
        ax.text(
            0.5,
            0.91,
            f"{ratio:.2f}x slower\nordered vs bitset",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
        )

    fig.suptitle(
        "FastANI-style L2 data structure comparison, 60 genomes\n"
        f"Same L2 windows: {windows:,}",
        y=0.98,
        fontsize=14,
    )
    fig.savefig(exp / "fastani_style_l2_bitset_vs_ordered_runtime.pdf", bbox_inches="tight")
    fig.savefig(
        exp / "fastani_style_l2_bitset_vs_ordered_runtime.png",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
