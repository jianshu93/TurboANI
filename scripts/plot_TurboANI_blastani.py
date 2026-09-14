#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def genome_key(value: object) -> str:
    name = Path(str(value)).name
    if name.endswith(".gz"):
        name = name[:-3]
    for suffix in (".fna", ".fa", ".fasta"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def pair_key(a: object, b: object) -> tuple[str, str]:
    x = genome_key(a)
    y = genome_key(b)
    return (x, y) if x <= y else (y, x)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
            "font.size": 18,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 15,
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


def load_blastani_truth(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t")
    rows: list[tuple[str, str, float]] = []
    for row in raw.itertuples(index=False):
        ani = float(row.rbm_ani)
        if ani < 0 or math.isnan(ani):
            continue
        g1 = genome_key(row.genome1)
        g2 = genome_key(row.genome2)
        a, b = pair_key(g1, g2)
        rows.append((a, b, ani))

    truth = pd.DataFrame(rows, columns=["genome1", "genome2", "truth_ani"])
    return truth.drop_duplicates(["genome1", "genome2"], keep="first")


def load_turboani_average(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        usecols=[0, 1, 2],
        names=["query", "reference", "ani"],
        engine="python",
    )
    raw["ani"] = raw["ani"].astype(float)
    raw = raw[raw["ani"] >= 0.0].copy()
    raw[["genome1", "genome2"]] = pd.DataFrame(
        [pair_key(q, r) for q, r in zip(raw["query"], raw["reference"])],
        index=raw.index,
    )
    return (
        raw.groupby(["genome1", "genome2"], as_index=False)
        .agg(ani_avg=("ani", "mean"), direction_count=("ani", "size"))
        .sort_values(["genome1", "genome2"])
    )


def compute_summary(df: pd.DataFrame, truth_pairs: int) -> dict[str, float]:
    delta = df["ani_avg"].to_numpy(dtype=float) - df["truth_ani"].to_numpy(dtype=float)
    x = df["truth_ani"].to_numpy(dtype=float)
    y = df["ani_avg"].to_numpy(dtype=float)
    return {
        "truth_pairs": truth_pairs,
        "matched_pairs": len(df),
        "computed_fraction": len(df) / truth_pairs if truth_pairs else math.nan,
        "pearson": float(np.corrcoef(x, y)[0, 1]) if len(df) > 1 else math.nan,
        "mae": float(np.mean(np.abs(delta))) if len(df) else math.nan,
        "rmse": float(np.sqrt(np.mean(delta * delta))) if len(df) else math.nan,
        "bias": float(np.mean(delta)) if len(df) else math.nan,
        "recovery_0_5pct": float(np.mean(np.abs(delta) <= 0.5)) if len(df) else math.nan,
        "recovery_1pct": float(np.mean(np.abs(delta) <= 1.0)) if len(df) else math.nan,
        "recovery_2pct": float(np.mean(np.abs(delta) <= 2.0)) if len(df) else math.nan,
    }


def plot_joint(df: pd.DataFrame, summary: dict[str, float], out_pdf: Path, out_png: Path) -> None:
    configure_matplotlib()
    x = df["truth_ani"].to_numpy(dtype=float)
    y = df["ani_avg"].to_numpy(dtype=float)

    lo = max(70.0, math.floor(min(float(np.min(x)), float(np.min(y))) / 5.0) * 5.0)
    hi = 100.0
    bins = np.linspace(lo, hi, int((hi - lo) * 4) + 1)

    fig = plt.figure(figsize=(6.4, 5.9), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        left=0.14,
        right=0.92,
        bottom=0.13,
        top=0.90,
        width_ratios=[4.2, 1.0],
        height_ratios=[1.0, 4.2],
        hspace=0.07,
        wspace=0.06,
    )
    ax_top = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(grid[1, 1], sharey=ax)
    fig.add_subplot(grid[0, 1]).axis("off")

    h, xedges, yedges = np.histogram2d(x, y, bins=[bins, bins])
    masked = np.ma.masked_where(h.T <= 0, h.T)
    cmap = mpl.colormaps["Purples"].copy()
    cmap.set_bad("white")
    mesh = ax.pcolormesh(
        xedges,
        yedges,
        masked,
        cmap=cmap,
        norm=LogNorm(vmin=1, vmax=max(2, float(h.max()))),
        shading="auto",
    )
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.1, zorder=3)

    top_counts, top_edges = np.histogram(x, bins=bins)
    top_x = (top_edges[:-1] + top_edges[1:]) / 2.0
    right_counts, right_edges = np.histogram(y, bins=bins)
    right_y = (right_edges[:-1] + right_edges[1:]) / 2.0
    fill = mpl.colormaps["Purples"](0.62)
    ax_top.bar(
        top_x,
        top_counts,
        width=np.diff(top_edges),
        color=fill,
        alpha=0.72,
        edgecolor="none",
        linewidth=0,
    )
    ax_right.barh(
        right_y,
        right_counts,
        height=np.diff(right_edges),
        color=fill,
        alpha=0.72,
        edgecolor="none",
        linewidth=0,
    )

    ax_top.set_title("FastANI-style L1 hybrid", fontweight="bold", pad=10)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = [t for t in [70, 75, 80, 85, 90, 95, 100] if lo <= t <= hi]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("BLASTANI truth (%)")
    ax.set_ylabel("Estimated ANI (%)")
    ax.text(
        0.035,
        0.965,
        f"n={summary['matched_pairs']:,.0f}/{summary['truth_pairs']:,.0f}\n"
        f"r={summary['pearson']:.4f}\n"
        f"MAE={summary['mae']:.2f}, RMSE={summary['rmse']:.2f}\n"
        f"bias={summary['bias']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=3),
    )
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.set_ylabel("Pair count", fontsize=12)
    ax_top.set_yticks([])
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.set_xlabel("Pair count", fontsize=12)
    ax_right.set_xticks([])
    for panel_ax in (ax, ax_top, ax_right):
        for spine in panel_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.9)

    cax = inset_axes(
        ax,
        width="3.2%",
        height="28%",
        loc="lower right",
        bbox_to_anchor=(-0.055, 0.0, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=1.0,
    )
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("Pair count", fontsize=9)
    cb.ax.tick_params(labelsize=8, width=0.7, length=2)

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the experimental TurboANI hybrid "
            "(SIMD minimizers + twisted tabulation + FastANI-style L1 + bitset L2) "
            "against BLASTANI truth."
        )
    )
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--turbo", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    truth = load_blastani_truth(args.truth)
    hybrid = load_turboani_average(args.hybrid)
    joined = truth.merge(hybrid, how="left", on=["genome1", "genome2"])
    matched = joined.dropna(subset=["ani_avg"]).copy()

    summary = compute_summary(matched, truth_pairs=len(truth))
    joined.to_csv(args.out_dir / "hybrid_fastani_l1_vs_blastani_joined.tsv", sep="\t", index=False)
    pd.DataFrame([summary]).to_csv(
        args.out_dir / "hybrid_fastani_l1_vs_blastani_summary.tsv",
        sep="\t",
        index=False,
        float_format="%.8f",
    )
    plot_joint(
        matched,
        summary,
        args.out_dir / "hybrid_fastani_l1_vs_blastani_correlation.pdf",
        args.out_dir / "hybrid_fastani_l1_vs_blastani_correlation.png",
    )
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
