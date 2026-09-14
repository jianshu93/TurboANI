#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ["FastANI", "TurboANI", "TurboANI id77.5", "skani"]
COLORS = {
    # RColorBrewer Accent-inspired colors.
    "FastANI": "#7FC97F",
    "TurboANI": "#BEAED4",
    "TurboANI id77.5": "#FDC086",
    "skani": "#386CB0",
}
MARKERS = {
    "FastANI": "o",
    "TurboANI": "s",
    "TurboANI id77.5": "^",
    "skani": "D",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
            "font.size": 19,
            "axes.titlesize": 19,
            "axes.labelsize": 19,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
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


def load_common_residuals(base: Path) -> pd.DataFrame:
    long = pd.read_csv(base / "orthoani_vs_fastani_turboani_id775_long.tsv", sep="\t")
    long = long.rename(
        columns={
            "method_ani_avg": "ani_avg",
            "delta_vs_orthoani": "delta",
        }
    )[
        [
            "genome1",
            "genome2",
            "method",
            "orthoani",
            "ani_avg",
            "delta",
            "direction_count",
        ]
    ]
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

    df_all = pd.concat([long, skani_long], ignore_index=True)
    df_all = df_all[df_all["genome1"] != df_all["genome2"]].copy()

    pair_methods = df_all.groupby(["genome1", "genome2"])["method"].agg(lambda values: set(values))
    common_pairs = pair_methods[pair_methods.apply(lambda values: set(METHODS).issubset(values))].index
    common_index = pd.MultiIndex.from_tuples(common_pairs, names=["genome1", "genome2"])
    df = df_all.set_index(["genome1", "genome2"]).loc[common_index].reset_index()
    return df[df["method"].isin(METHODS)].copy()


def summarize_binned_rmse(
    df: pd.DataFrame,
    min_ani: float,
    max_ani: float,
    bin_width: float,
    min_pairs: int,
) -> pd.DataFrame:
    edges = np.arange(min_ani, max_ani + bin_width, bin_width)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = df[(df["orthoani"] >= low) & (df["orthoani"] < high)]
        bin_name = f"{low:g}-{high:g}"
        midpoint = (low + high) / 2.0
        for method in METHODS:
            sub = in_bin[in_bin["method"] == method]
            delta = sub["delta"].to_numpy()
            n_pairs = int(len(delta))
            if n_pairs < min_pairs:
                rmse = np.nan
                mae = np.nan
                mse = np.nan
                residual_sd = np.nan
            else:
                mse = float(np.mean(delta * delta))
                rmse = float(np.sqrt(mse))
                mae = float(np.mean(np.abs(delta)))
                residual_sd = float(np.std(delta, ddof=1)) if n_pairs > 1 else 0.0
            rows.append(
                {
                    "orthoani_bin": bin_name,
                    "bin_low": low,
                    "bin_high": high,
                    "bin_midpoint": midpoint,
                    "method": method,
                    "n_pairs": n_pairs,
                    "rmse": rmse,
                    "mae": mae,
                    "mse": mse,
                    "residual_sd": residual_sd,
                }
            )
    summary = pd.DataFrame(rows)
    skani = summary[summary["method"] == "skani"][["orthoani_bin", "rmse", "mae", "mse"]].rename(
        columns={"rmse": "skani_rmse", "mae": "skani_mae", "mse": "skani_mse"}
    )
    summary = summary.merge(skani, on="orthoani_bin", how="left")
    summary["skani_over_method_rmse"] = summary["skani_rmse"] / summary["rmse"]
    summary["skani_over_method_mae"] = summary["skani_mae"] / summary["mae"]
    summary["skani_over_method_mse"] = summary["skani_mse"] / summary["mse"]
    return summary


def plot_binned_rmse(summary: pd.DataFrame, out_prefix: Path) -> None:
    fig = plt.figure(figsize=(15.5, 8.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.45, 1.15], hspace=0.11)
    ax = fig.add_subplot(gs[0, 0])
    ax_ratio = fig.add_subplot(gs[1, 0], sharex=ax)

    for method in METHODS:
        sub = summary[summary["method"] == method]
        ax.plot(
            sub["bin_midpoint"],
            sub["rmse"],
            marker=MARKERS[method],
            markersize=5.5,
            linewidth=2.0,
            color=COLORS[method],
            label=method,
        )

    ratio_methods = [method for method in METHODS if method != "skani"]
    for method in ratio_methods:
        sub = summary[summary["method"] == method]
        ax_ratio.plot(
            sub["bin_midpoint"],
            sub["skani_over_method_rmse"],
            marker=MARKERS[method],
            markersize=5.5,
            linewidth=2.0,
            color=COLORS[method],
            label=f"skani/{method}",
        )

    ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("RMSE vs OrthoANIu (%)")
    ax_ratio.set_ylabel("skani / method RMSE")
    ax_ratio.set_xlabel("OrthoANIu truth bin midpoint (%)")

    x_ticks = sorted(summary["bin_midpoint"].unique())
    ax_ratio.set_xticks(x_ticks)
    ax_ratio.set_xticklabels([f"{x:g}" for x in x_ticks], rotation=45, ha="right")

    y_max = float(summary["rmse"].max(skipna=True))
    ratio_max = float(summary["skani_over_method_rmse"].replace([np.inf, -np.inf], np.nan).max(skipna=True))
    ax.set_ylim(0, y_max * 1.12)
    ax_ratio.set_ylim(0, max(1.25, ratio_max * 1.16))
    ax.tick_params(axis="x", labelbottom=False)

    ax.legend(loc="upper right", frameon=False, ncol=4, handlelength=1.4, columnspacing=1.1)
    ax_ratio.legend(loc="upper right", frameon=False, ncol=3, handlelength=1.4, columnspacing=1.1)

    # Pair counts are common across methods because the input is the shared computed pair set.
    count_rows = summary[summary["method"] == METHODS[0]]
    for _, row in count_rows.iterrows():
        ax.text(
            row["bin_midpoint"],
            y_max * 0.035,
            f"n={int(row['n_pairs']):,}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=10,
            color="black",
        )

    for target_ax in (ax, ax_ratio):
        target_ax.spines["top"].set_visible(False)
        target_ax.spines["right"].set_visible(False)

    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot binned RMSE by OrthoANIu truth bin.")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the correlation intermediate TSV files.",
    )
    parser.add_argument("--min-ani", type=float, default=75.0)
    parser.add_argument("--max-ani", type=float, default=95.0)
    parser.add_argument("--bin-width", type=float, default=1.0)
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=10,
        help="Bins with fewer common pairs are written to TSV but omitted from plotted RMSE.",
    )
    args = parser.parse_args()

    base = args.base
    configure_matplotlib()
    df = load_common_residuals(base)
    summary = summarize_binned_rmse(df, args.min_ani, args.max_ani, args.bin_width, args.min_pairs)

    width_label = f"{args.bin_width:g}pct".replace(".", "p")
    range_label = f"{args.min_ani:g}_{args.max_ani:g}".replace(".", "p")
    out_prefix = base / f"ani_binned_rmse_by_orthoani_{width_label}_{range_label}"
    summary.to_csv(out_prefix.with_name(out_prefix.name + "_summary.tsv"), sep="\t", index=False, float_format="%.6f")
    plot_binned_rmse(summary, out_prefix)

    compact = summary[
        (summary["method"].isin(["TurboANI", "TurboANI id77.5", "skani"]))
        & (summary["bin_low"] >= 80)
        & (summary["bin_high"] <= 85)
    ][
        [
            "orthoani_bin",
            "method",
            "n_pairs",
            "rmse",
            "mae",
            "mse",
            "skani_over_method_rmse",
            "skani_over_method_mse",
        ]
    ]
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
