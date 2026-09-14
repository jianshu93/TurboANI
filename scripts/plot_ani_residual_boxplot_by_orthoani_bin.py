#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 15,
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


def load_common_residuals(base: Path) -> pd.DataFrame:
    long = pd.read_csv(base / "orthoani_vs_fastani_turboani_modes_long.tsv", sep="\t")
    long = long.rename(columns={"method_ani_avg": "ani_avg", "delta_vs_orthoani": "delta"})[
        ["genome1", "genome2", "method", "orthoani", "ani_avg", "delta", "direction_count"]
    ]
    long["method"] = long["method"].replace(
        {
            "TurboANI default 80": "TurboANI default 80",
            "TurboANI sensitive 77.5": "TurboANI sensitive 77.5",
            "TurboANI fast 82.5": "TurboANI fast 82.5",
        }
    )
    long["status"] = "computed"

    skani = pd.read_csv(base / "orthoani_vs_skani_joined.tsv", sep="\t")
    skani_comp = skani[skani["status"] == "computed"].copy()
    skani_long = pd.DataFrame(
        {
            "genome1": skani_comp["genome1"],
            "genome2": skani_comp["genome2"],
            "method": "skani",
            "orthoani": skani_comp["orthoani"],
            "ani_avg": skani_comp["skani_avg"],
            "delta": skani_comp["skani_delta_vs_orthoani"],
            "direction_count": skani_comp["skani_computed_direction_count"],
            "status": "computed",
        }
    )

    methods = ["FastANI", "TurboANI sensitive 77.5", "TurboANI default 80", "TurboANI fast 82.5", "skani"]
    df_all = pd.concat([long, skani_long], ignore_index=True)
    df_all = df_all[df_all["genome1"] != df_all["genome2"]].copy()

    pair_methods = df_all.groupby(["genome1", "genome2"])["method"].agg(lambda values: set(values))
    common_pairs = pair_methods[pair_methods.apply(lambda values: set(methods).issubset(values))].index
    common_index = pd.MultiIndex.from_tuples(common_pairs, names=["genome1", "genome2"])
    df = df_all.set_index(["genome1", "genome2"]).loc[common_index].reset_index()
    return df[df["method"].isin(methods)].copy()


def make_summary(df: pd.DataFrame, labels: list[str], methods: list[str]) -> pd.DataFrame:
    rows = []
    for label in labels:
        for method in methods:
            sub = df[(df["orthoani_bin"].astype(str) == label) & (df["method"] == method)]
            if sub.empty:
                continue
            q1 = sub["delta"].quantile(0.25)
            q3 = sub["delta"].quantile(0.75)
            rows.append(
                {
                    "orthoani_bin": label,
                    "method": method,
                    "n_pairs": int(len(sub)),
                    "mean_delta": sub["delta"].mean(),
                    "median_delta": sub["delta"].median(),
                    "std_delta": sub["delta"].std(ddof=1),
                    "var_delta": sub["delta"].var(ddof=1),
                    "iqr_delta": q3 - q1,
                    "mae": sub["delta"].abs().mean(),
                    "median_abs_delta": sub["delta"].abs().median(),
                    "p05_delta": sub["delta"].quantile(0.05),
                    "p95_delta": sub["delta"].quantile(0.95),
                }
            )

    for method in methods:
        sub = df[(df["orthoani"] < 90) & (df["method"] == method)]
        rows.append(
            {
                "orthoani_bin": "<90 aggregate",
                "method": method,
                "n_pairs": int(len(sub)),
                "mean_delta": sub["delta"].mean(),
                "median_delta": sub["delta"].median(),
                "std_delta": sub["delta"].std(ddof=1),
                "var_delta": sub["delta"].var(ddof=1),
                "iqr_delta": sub["delta"].quantile(0.75) - sub["delta"].quantile(0.25),
                "mae": sub["delta"].abs().mean(),
                "median_abs_delta": sub["delta"].abs().median(),
                "p05_delta": sub["delta"].quantile(0.05),
                "p95_delta": sub["delta"].quantile(0.95),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot residual distributions against OrthoANIu bins for common computed pairs."
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the correlation intermediate TSV files.",
    )
    args = parser.parse_args()
    base = args.base

    methods = ["FastANI", "TurboANI sensitive 77.5", "TurboANI default 80", "TurboANI fast 82.5", "skani"]
    labels = ["75-80", "80-85", "85-90", "90-95", "95-100"]
    edges = [75, 80, 85, 90, 95, 100.000001]

    df = load_common_residuals(base)
    df["orthoani_bin"] = pd.cut(
        df["orthoani"],
        bins=edges,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    df = df.dropna(subset=["orthoani_bin"]).copy()
    summary = make_summary(df, labels, methods)

    out_prefix = base / "ani_residual_boxplot_by_orthoani_bin"
    summary.to_csv(out_prefix.with_name(out_prefix.name + "_summary.tsv"), sep="\t", index=False, float_format="%.6f")
    df.to_csv(base / "ani_residual_boxplot_common_pairs_long.tsv", sep="\t", index=False, float_format="%.6f")

    configure_matplotlib()
    colors = {
        "FastANI": "#7FC97F",
        "TurboANI sensitive 77.5": "#E41A1C",
        "TurboANI default 80": "#BEAED4",
        "TurboANI fast 82.5": "#FDC086",
        "skani": "#386CB0",
    }

    fig = plt.figure(figsize=(12.2, 7.8 * 2.0 / 3.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.15], hspace=0.16)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)

    centers = np.arange(len(labels))
    offsets = np.linspace(-0.34, 0.34, len(methods))
    width = 0.12

    for mi, method in enumerate(methods):
        data = []
        positions = []
        for bi, label in enumerate(labels):
            vals = df[(df["method"] == method) & (df["orthoani_bin"].astype(str) == label)]["delta"].to_numpy()
            data.append(vals)
            positions.append(centers[bi] + offsets[mi])
        box = ax.boxplot(
            data,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
            manage_ticks=False,
            medianprops={"color": "black", "linewidth": 1.15},
            whiskerprops={"color": "black", "linewidth": 0.8},
            capprops={"color": "black", "linewidth": 0.8},
            boxprops={"color": "black", "linewidth": 0.8},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(colors[method])
            patch.set_alpha(0.82)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.8)
    ax.set_ylabel("ANI estimate - OrthoANIu (%)")
    ax.set_xlim(-0.58, len(labels) - 0.42)
    q_low = df["delta"].quantile(0.01)
    q_high = df["delta"].quantile(0.99)
    ax.set_ylim(min(-1.2, q_low - 0.4), max(5.8, q_high + 0.45))
    ax.tick_params(axis="x", labelbottom=False)
    ax.legend(
        handles=[Patch(facecolor=colors[method], edgecolor="black", label=method) for method in methods],
        loc="upper left",
        ncol=1,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.0,
        borderaxespad=0.2,
    )

    for method in methods:
        values = []
        for label in labels:
            row = summary[(summary["orthoani_bin"] == label) & (summary["method"] == method)]
            values.append(float(row["std_delta"].iloc[0]) if not row.empty else np.nan)
        ax2.plot(centers, values, marker="o", linewidth=2.0, markersize=5.5, color=colors[method])

    bin_counts = [
        int(df[(df["orthoani_bin"].astype(str) == label) & (df["method"] == methods[0])].shape[0])
        for label in labels
    ]
    ax2.set_ylabel("Residual SD (%)")
    ax2.set_xlabel("OrthoANIu truth bin (%)")
    ax2.set_xticks(centers)
    ax2.set_xticklabels([f"{label}\nn={count:,}" for label, count in zip(labels, bin_counts)])
    ax2.set_ylim(bottom=0)

    ax.text(
        0.995,
        0.98,
        "Common computed non-self pairs; box whiskers = 5th-95th percentile",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        color="black",
    )
    for target_ax in (ax, ax2):
        target_ax.spines["top"].set_visible(False)
        target_ax.spines["right"].set_visible(False)

    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
