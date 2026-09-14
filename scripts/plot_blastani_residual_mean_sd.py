#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ["FastANI", "TurboANI default", "skani sensitive"]
COLORS = {
    "FastANI": "#7FC97F",
    "TurboANI default": "#BEAED4",
    "skani sensitive": "#386CB0",
}
MARKERS = {
    "FastANI": "s",
    "TurboANI default": "s",
    "skani sensitive": "s",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
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


def genome_key(value: object) -> str:
    name = Path(str(value)).name
    if name.endswith(".gz"):
        name = name[:-3]
    return name


def pair_key(a: object, b: object) -> tuple[str, str]:
    x = genome_key(a)
    y = genome_key(b)
    return (x, y) if x <= y else (y, x)


def load_blastani_truth(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t")
    rows = []
    for row in raw.itertuples(index=False):
        ani = float(row.rbm_ani)
        if ani < 0:
            continue
        g1, g2 = pair_key(row.genome1, row.genome2)
        rows.append((g1, g2, ani))
    truth = pd.DataFrame(rows, columns=["genome1", "genome2", "blastani"])
    return truth.drop_duplicates(["genome1", "genome2"], keep="first").sort_values(["genome1", "genome2"])


def load_ani_table(path: Path, *, scale: float = 1.0, has_header: bool = False) -> pd.DataFrame:
    values: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    with open(path, newline="") as handle:
        if has_header:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                ref = row.get("Ref_file") or row.get("reference")
                qry = row.get("Query_file") or row.get("query")
                ani_value = row.get("ANI") or row.get("ani")
                if ref is None or qry is None or ani_value is None:
                    continue
                if genome_key(ref) == genome_key(qry):
                    continue
                ani = float(ani_value)
                if ani < 0:
                    continue
                values[pair_key(ref, qry)].append(ani * scale)
        else:
            for line in handle:
                if not line.strip():
                    continue
                cols = line.rstrip("\n").split()
                if len(cols) < 3:
                    continue
                query, reference = cols[0], cols[1]
                if genome_key(query) == genome_key(reference):
                    continue
                ani = float(cols[2])
                if ani < 0:
                    continue
                values[pair_key(query, reference)].append(ani * scale)

    rows = [(key[0], key[1], float(np.mean(vals)), len(vals)) for key, vals in values.items()]
    return pd.DataFrame(rows, columns=["genome1", "genome2", "ani_avg", "direction_count"]).sort_values(
        ["genome1", "genome2"]
    )


def build_long_table(
    truth: pd.DataFrame,
    fastani: pd.DataFrame,
    turboani: pd.DataFrame,
    skani_sensitive: pd.DataFrame,
    *,
    common_only: bool,
) -> pd.DataFrame:
    method_tables = {
        "FastANI": fastani,
        "TurboANI default": turboani,
        "skani sensitive": skani_sensitive,
    }
    long_rows = []
    for method, table in method_tables.items():
        merged = truth.merge(table, how="left", on=["genome1", "genome2"])
        merged["method"] = method
        merged["delta"] = merged["ani_avg"] - merged["blastani"]
        long_rows.append(
            merged[["genome1", "genome2", "blastani", "method", "ani_avg", "direction_count", "delta"]]
        )
    long = pd.concat(long_rows, ignore_index=True)

    if common_only:
        complete = (
            long.dropna(subset=["ani_avg"])
            .groupby(["genome1", "genome2"])["method"]
            .agg(lambda values: set(values))
        )
        complete_pairs = complete[complete.apply(lambda values: set(METHODS).issubset(values))].index
        common_index = pd.MultiIndex.from_tuples(complete_pairs, names=["genome1", "genome2"])
        long = long.set_index(["genome1", "genome2"]).loc[common_index].reset_index()

    return long.dropna(subset=["ani_avg"]).copy()


def make_summary(df: pd.DataFrame, bin_labels: list[str]) -> pd.DataFrame:
    rows = []
    for label in bin_labels:
        for method in METHODS:
            sub = df[(df["blastani_bin"].astype(str) == label) & (df["method"] == method)]
            if sub.empty:
                rows.append(
                    {
                        "blastani_bin": label,
                        "method": method,
                        "n_pairs": 0,
                        "mean_delta": np.nan,
                        "median_delta": np.nan,
                        "std_delta": np.nan,
                        "mae": np.nan,
                        "rmse": np.nan,
                        "p05_delta": np.nan,
                        "p95_delta": np.nan,
                    }
                )
                continue
            delta = sub["delta"].to_numpy(dtype=float)
            rows.append(
                {
                    "blastani_bin": label,
                    "method": method,
                    "n_pairs": int(len(sub)),
                    "mean_delta": float(np.mean(delta)),
                    "median_delta": float(np.median(delta)),
                    "std_delta": float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0,
                    "mae": float(np.mean(np.abs(delta))),
                    "rmse": float(np.sqrt(np.mean(delta * delta))),
                    "p05_delta": float(np.quantile(delta, 0.05)),
                    "p95_delta": float(np.quantile(delta, 0.95)),
                }
            )
    return pd.DataFrame(rows)


def plot_absolute_error_box_sd(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    bin_counts: dict[str, int],
    out_pdf: Path,
) -> None:
    configure_matplotlib()

    labels = list(dict.fromkeys(summary["blastani_bin"].astype(str)))
    x = np.arange(len(labels), dtype=float) * 1.12
    offsets = np.linspace(-0.34, 0.34, len(METHODS))
    width = 0.17

    fig = plt.figure(figsize=(12.2, 7.8 * 2.0 / 3.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.2], hspace=0.20)
    ax_abs = fig.add_subplot(gs[0, 0])
    ax_sd = fig.add_subplot(gs[1, 0], sharex=ax_abs)

    for offset, method in zip(offsets, METHODS):
        box_data = []
        positions = []
        for xi, label in zip(x, labels):
            vals = df[(df["method"] == method) & (df["blastani_bin"].astype(str) == label)]["delta"]
            box_data.append(vals.abs().to_numpy(dtype=float))
            positions.append(xi + offset)
        box = ax_abs.boxplot(
            box_data,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
            manage_ticks=False,
            medianprops={"color": "black", "linewidth": 1.1},
            whiskerprops={"color": "black", "linewidth": 0.8},
            capprops={"color": "black", "linewidth": 0.8},
            boxprops={"color": "black", "linewidth": 0.8},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.82)

        sub = summary[summary["method"] == method].set_index("blastani_bin").loc[labels].reset_index()
        y_sd = sub["std_delta"].to_numpy(dtype=float)
        ax_sd.plot(
            x + offset,
            y_sd,
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=7.5,
            linewidth=2.1,
            label=method,
        )

    ax_abs.set_ylabel("Absolute\nerror (%)", labelpad=8)
    ax_sd.set_ylabel("Residual\nSD (%)", labelpad=8)
    ax_sd.set_xlabel("BLASTANI truth bin (%)")
    ax_sd.set_xticks(x)
    ax_sd.set_xticklabels([f"{label}\nn={bin_counts.get(label, 0):,}" for label in labels])

    ax_abs.set_ylim(0.0, 2.0)
    ax_sd.set_ylim(0, max(0.8, float(summary["std_delta"].max()) * 1.22))

    from matplotlib.patches import Patch

    ax_abs.legend(
        handles=[Patch(facecolor=COLORS[method], edgecolor="black", label=method, alpha=0.82) for method in METHODS],
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        handlelength=1.4,
        columnspacing=1.2,
        handletextpad=0.45,
    )
    ax_abs.tick_params(axis="x", labelbottom=False)

    for ax in (ax_abs, ax_sd):
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
        ax.tick_params(length=4.0, width=1.0)

    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.17, top=0.82)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot mean residual and residual SD using BLASTANI truth.")
    parser.add_argument("--truth", type=Path, default=Path("/Users/jianshuzhao/blastANI_300.all_results.tsv"))
    parser.add_argument(
        "--fastani",
        type=Path,
        default=Path(
            "/Users/jianshuzhao/Library/Mobile Documents/com~apple~CloudDocs/TurboANI_paper/accuracy/strep_300_fastANI_local_t16.txt"
        ),
    )
    parser.add_argument(
        "--turboani-default",
        type=Path,
        default=Path(
            "/Users/jianshuzhao/Library/Mobile Documents/com~apple~CloudDocs/TurboANI_paper/accuracy/strep_300_turboani_min80.txt"
        ),
    )
    parser.add_argument(
        "--skani-sensitive",
        type=Path,
        default=Path(
            "/Users/jianshuzhao/Library/Mobile Documents/com~apple~CloudDocs/TurboANI_paper/accuracy/strep_300_superani.txt"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/jianshuzhao/Documents/Codex/2026-07-23/for/outputs/strep300_blastani_truth_20260901"),
    )
    parser.add_argument(
        "--all-computed",
        action="store_true",
        help="Use every computed pair per method instead of the pair-matched common subset.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    truth = load_blastani_truth(args.truth)
    fastani = load_ani_table(args.fastani, scale=1.0)
    turboani = load_ani_table(args.turboani_default, scale=1.0)
    skani_sensitive = load_ani_table(args.skani_sensitive, scale=100.0)

    long = build_long_table(
        truth,
        fastani,
        turboani,
        skani_sensitive,
        common_only=not args.all_computed,
    )

    bin_edges = [75, 80, 85, 90, 95, 100.000001]
    bin_labels = ["75-80", "80-85", "85-90", "90-95", "95-100"]
    long["blastani_bin"] = pd.cut(
        long["blastani"],
        bins=bin_edges,
        labels=bin_labels,
        right=False,
        include_lowest=True,
    )
    long = long.dropna(subset=["blastani_bin"]).copy()

    summary = make_summary(long, bin_labels)
    count_df = long[long["method"] == METHODS[0]].groupby("blastani_bin", observed=True).size()
    bin_counts = {str(label): int(count_df.get(label, 0)) for label in bin_labels}

    suffix = "all_computed" if args.all_computed else "common_pairs"
    long_out = args.out_dir / f"blastani_residual_mean_sd_{suffix}_long.tsv"
    summary_out = args.out_dir / f"blastani_residual_mean_sd_{suffix}_summary.tsv"
    plot_out = args.out_dir / f"blastani_absolute_error_boxplot_residual_sd_{suffix}.pdf"

    long.to_csv(long_out, sep="\t", index=False, float_format="%.6f")
    summary.to_csv(summary_out, sep="\t", index=False, float_format="%.6f")
    plot_absolute_error_box_sd(long, summary, bin_counts, plot_out)

    print(f"truth_pairs\t{len(truth)}")
    print(f"common_or_plotted_rows\t{len(long)}")
    print(f"plot\t{plot_out}")
    print(f"summary\t{summary_out}")
    print(f"long_table\t{long_out}")


if __name__ == "__main__":
    main()
