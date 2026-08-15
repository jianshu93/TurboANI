#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ACCENT = {
    "green": "#7FC97F",
    "purple": "#BEAED4",
    "orange": "#FDC086",
    "yellow": "#FFFF99",
    "blue": "#386CB0",
    "red": "#F0027F",
    "brown": "#BF5B17",
    "gray": "#666666",
}

BIN_WIDTH = 0.5


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
            "font.size": 20,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 17,
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


def load_orthoani(path: Path) -> np.ndarray:
    df = pd.read_csv(path, sep="\t")
    if "orthoANI_value" not in df.columns:
        raise ValueError(f"{path} does not contain column 'orthoANI_value'")
    values = pd.to_numeric(df["orthoANI_value"], errors="coerce").dropna().to_numpy(float)
    values = values[np.isfinite(values)]
    return values


def write_summary(values: np.ndarray, out_tsv: Path) -> None:
    summary = {
        "n_pairs": len(values),
        "min": np.min(values),
        "q01": np.quantile(values, 0.01),
        "q05": np.quantile(values, 0.05),
        "q25": np.quantile(values, 0.25),
        "median": np.median(values),
        "mean": np.mean(values),
        "q75": np.quantile(values, 0.75),
        "q95": np.quantile(values, 0.95),
        "q99": np.quantile(values, 0.99),
        "max": np.max(values),
    }
    pd.DataFrame([summary]).to_csv(out_tsv, sep="\t", index=False, float_format="%.6f")


def plot_distribution(values: np.ndarray, out_pdf: Path, out_png: Path | None) -> None:
    configure_matplotlib()

    bins = np.arange(np.floor(values.min()), np.ceil(values.max()) + BIN_WIDTH, BIN_WIDTH)
    if bins[0] > 75:
        bins = np.insert(bins, 0, 75.0)
    if bins[-1] < 100:
        bins = np.append(bins, 100.0)

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    counts, _, patches = ax.hist(
        values,
        bins=bins,
        color=ACCENT["green"],
        edgecolor="black",
        linewidth=0.45,
        alpha=0.92,
    )

    median = float(np.median(values))
    mean = float(np.mean(values))
    ax.axvline(median, color=ACCENT["blue"], linewidth=2.2, label=f"Median = {median:.2f}%")
    ax.axvline(mean, color=ACCENT["red"], linewidth=2.2, linestyle="--", label=f"Mean = {mean:.2f}%")

    ax.set_title("OrthoANIu Distribution")
    ax.set_xlabel("OrthoANIu (%)")
    ax.set_ylabel("Number of genome pairs")
    ax.set_xlim(np.floor(values.min()), 100)
    ax.set_xticks(np.arange(75, 101, 5))
    ax.set_ylim(0, max(counts) * 1.12)

    ax.text(
        0.03,
        0.95,
        f"n = {len(values):,} non-self pairs",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=17,
    )

    ax.legend(frameon=False, loc="upper right", handlelength=2.2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    if out_png is not None:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the 300-genome OrthoANIu pairwise distribution.")
    parser.add_argument(
        "--truth",
        type=Path,
        default=Path(
            "/Users/jianshuzhao/Library/Mobile Documents/com~apple~CloudDocs/"
            "TurboANI_paper/accuracy/orthoani_all_44850_pairs.results.tsv"
        ),
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path(
            "/Users/jianshuzhao/Documents/Codex/2026-07-23/for/outputs/"
            "orthoani_300_distribution/orthoani_300_distribution"
        ),
    )
    args = parser.parse_args()

    values = load_orthoani(args.truth)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_summary(values, args.out_prefix.with_suffix(".summary.tsv"))
    plot_distribution(values, args.out_prefix.with_suffix(".pdf"), args.out_prefix.with_suffix(".png"))


if __name__ == "__main__":
    main()
