from pathlib import Path
import csv

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],  # system fallback handled automatically
        "font.size": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 12,
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


ROOT = Path(
    "/Users/jianshuzhao/Library/Mobile Documents/"
    "com~apple~CloudDocs/TurboANI_paper"
)

SUMMARY = ROOT / "fastani_cpp_vs_rust_bitani_200_stage_summary_new.tsv"
OUT = ROOT / "bitani_vs_cpp_fastani_two_axis_stacked_bar.pdf"


# RColorBrewer Accent series.
ACCENT = {
    "minimizers": "#7FC97F",
    "L1": "#BEAED4",
    "L2": "#FDC086",
    "Distance": "#386CB0",
}


STAGES = [
    ("query_minimizers_sec", "minimizers"),
    ("l1_sec", "L1"),
    ("l2_sec", "L2"),
    ("l2_distance_sec", "Distance"),
]


KEEP = {
    "C++ FastANI instrumented": "C++ FastANI",
    "bitANI L1 optimized": "TurboANI",
}


def load_rows():
    if not SUMMARY.is_file():
        raise FileNotFoundError(f"Summary file not found:\n{SUMMARY}")

    with SUMMARY.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    selected_rows = [row for row in rows if row["label"] in KEEP]

    if len(selected_rows) != len(KEEP):
        found_labels = {row["label"] for row in selected_rows}
        missing_labels = set(KEEP) - found_labels
        raise ValueError(
            "Could not find all requested rows in the summary file.\n"
            f"Missing labels: {sorted(missing_labels)}"
        )

    # Ensure the plotted order matches the order in KEEP.
    return sorted(
        selected_rows,
        key=lambda row: list(KEEP).index(row["label"]),
    )


def fmt_seconds(value):
    if value >= 100:
        return f"{value:,.0f}s"

    if value >= 10:
        return f"{value:,.1f}s"

    return f"{value:,.2f}s"


def label_segment(ax, x, width, bottom, value, total, label):
    pct = 100.0 * value / total
    y_mid = bottom + value / 2.0
    text = f"{label}\n{fmt_seconds(value)}"

    if pct >= 5.5:
        ax.text(
            x,
            y_mid,
            text,
            ha="center",
            va="center",
            fontsize=10,
            color="black",
        )
    else:
        offset = total * (0.12 if bottom < total * 0.5 else -0.12)

        ax.annotate(
            text,
            xy=(x + width / 2.0, y_mid),
            xytext=(x + 0.62, y_mid + offset),
            ha="left",
            va="center",
            fontsize=10,
            arrowprops={
                "arrowstyle": "-",
                "color": "black",
                "linewidth": 0.8,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )


def draw_axis(ax, row):
    x = 0.0
    width = 0.5
    total = sum(float(row[key]) for key, _ in STAGES)
    bottom = 0.0

    for key, stage_label in STAGES:
        value = float(row[key])

        ax.bar(
            x,
            value,
            width=width,
            bottom=bottom,
            color=ACCENT[stage_label],
            edgecolor="black",
            linewidth=0.8,
        )

        label_segment(
            ax=ax,
            x=x,
            width=width,
            bottom=bottom,
            value=value,
            total=total,
            label=stage_label,
        )

        bottom += value

    wall = float(row["elapsed_sec"])

    ax.text(
        x,
        total * 1.035,
        f"Total {fmt_seconds(total)}\n"
        f"Wall {fmt_seconds(wall)} (16T)",
        ha="center",
        va="bottom",
        fontsize=14,
    )

    ax.set_ylabel("CPU-stage time (s)")
    ax.set_xlim(-0.65, 1.35)
    ax.set_ylim(0, total * 1.24)
    ax.set_xticks([])

    # Force scientific notation on the y-axis.
    # Tick labels will appear as compact values such as 0, 1, 2, 3,
    # with a shared multiplier such as ×10³ shown near the axis.
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    formatter.set_useOffset(False)

    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.get_offset_text().set_fontsize(14)

    ax.tick_params(
        axis="both",
        width=1.0,
        length=5,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def main():
    rows = load_rows()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.2, 5.2),
        constrained_layout=True,
        gridspec_kw={"wspace": 0.12},
    )

    for ax, row in zip(axes, rows):
        draw_axis(ax, row)

    handles = [
        Patch(
            facecolor=ACCENT[name],
            edgecolor="black",
            label=name,
        )
        for _, name in STAGES
    ]

    # Separate legend for the left subplot.
    axes[0].legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.47, 0.60),
        ncol=1,
        frameon=False,
        borderaxespad=0.0,
    )

    # Separate legend for the right subplot.
    axes[1].legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.60),
        ncol=1,
        frameon=False,
        borderaxespad=0.0,
    )

    fig.savefig(
        OUT,
        bbox_inches="tight",
    )

    print(OUT)


if __name__ == "__main__":
    main()
