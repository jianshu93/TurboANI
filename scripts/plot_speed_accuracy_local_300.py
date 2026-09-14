#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
GROUPS = [
    ("All pairs", None, None),
    ("75-90", 75.0, 90.0),
    ("75-85", 75.0, 85.0),
]


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


def genome_key(value: object) -> str:
    name = Path(str(value)).name
    if name.endswith(".gz"):
        name = name[:-3]
    return name


def pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def load_orthoani(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t")
    raw["genome1"] = raw["genome1"].map(genome_key)
    raw["genome2"] = raw["genome2"].map(genome_key)

    rows = []
    genomes: set[str] = set()
    for row in raw.itertuples(index=False):
        g1 = row.genome1
        g2 = row.genome2
        genomes.add(g1)
        genomes.add(g2)
        a, b = pair_key(g1, g2)
        rows.append((a, b, float(row.orthoANI_value)))

    ortho = pd.DataFrame(rows, columns=["genome1", "genome2", "orthoani"])
    ortho = ortho.drop_duplicates(["genome1", "genome2"], keep="first")
    self_rows = pd.DataFrame(
        [(g, g, 100.0) for g in sorted(genomes)],
        columns=["genome1", "genome2", "orthoani"],
    )
    ortho = pd.concat([ortho, self_rows], ignore_index=True)
    ortho = ortho.drop_duplicates(["genome1", "genome2"], keep="first")
    return ortho.sort_values(["genome1", "genome2"]).reset_index(drop=True)


def average_directed(path: Path, scale: float) -> pd.DataFrame:
    raw = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        usecols=[0, 1, 2],
        names=["query", "reference", "ani"],
        engine="python",
    )
    raw["query"] = raw["query"].map(genome_key)
    raw["reference"] = raw["reference"].map(genome_key)
    raw["ani"] = raw["ani"].astype(float)
    raw = raw[raw["ani"] >= 0.0].copy()
    raw["ani"] *= scale
    raw[["genome1", "genome2"]] = pd.DataFrame(
        [pair_key(q, r) for q, r in zip(raw["query"], raw["reference"])],
        index=raw.index,
    )
    return (
        raw.groupby(["genome1", "genome2"], as_index=False)
        .agg(ani_avg=("ani", "mean"), direction_count=("ani", "size"))
        .sort_values(["genome1", "genome2"])
    )


def parse_real_seconds(path: Path) -> float:
    text = path.read_text()
    match = re.search(r"^real\s+([0-9.]+)", text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"could not find real time in {path}")
    return float(match.group(1))


def load_method_outputs(out_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "fastANI": average_directed(out_dir / "strep_300_fastANI_local_t16.txt", scale=1.0),
        "superani": average_directed(out_dir / "strep_300_superani.txt", scale=100.0),
        "TurboANI-sensitive": average_directed(out_dir / "strep_300_turboani_min77_5.txt", scale=1.0),
        "TurboANI-default": average_directed(out_dir / "strep_300_turboani_min80.txt", scale=1.0),
        "TurboANI-fast": average_directed(out_dir / "strep_300_turboani_min82_5.txt", scale=1.0),
    }


def load_runtimes(out_dir: Path) -> dict[str, float]:
    return {
        "fastANI": parse_real_seconds(out_dir / "strep_300_fastANI_local_t16.stderr"),
        "superani": parse_real_seconds(out_dir / "strep_300_superani.stderr"),
        "TurboANI-sensitive": parse_real_seconds(out_dir / "strep_300_turboani_min77_5.stderr"),
        "TurboANI-default": parse_real_seconds(out_dir / "strep_300_turboani_min80.stderr"),
        "TurboANI-fast": parse_real_seconds(out_dir / "strep_300_turboani_min82_5.stderr"),
    }


def summarize(
    ortho: pd.DataFrame,
    methods: dict[str, pd.DataFrame],
    runtimes: dict[str, float],
    tolerances: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = ortho.copy()
    long_rows = []
    for method, values in methods.items():
        renamed = values.rename(
            columns={
                "ani_avg": f"{method}_ani_avg",
                "direction_count": f"{method}_direction_count",
            }
        )
        joined = joined.merge(renamed, how="left", on=["genome1", "genome2"])
        method_df = ortho.merge(renamed, how="left", on=["genome1", "genome2"])
        method_df["method"] = method
        method_df["ani_avg"] = method_df[f"{method}_ani_avg"]
        method_df["direction_count"] = method_df[f"{method}_direction_count"]
        method_df["delta"] = method_df["ani_avg"] - method_df["orthoani"]
        long_rows.append(
            method_df[
                [
                    "genome1",
                    "genome2",
                    "orthoani",
                    "method",
                    "ani_avg",
                    "direction_count",
                    "delta",
                ]
            ]
        )

    long = pd.concat(long_rows, ignore_index=True)
    fastani_time = runtimes["fastANI"]
    rows = []
    for tolerance in tolerances:
        for group_name, low, high in GROUPS:
            if low is None:
                truth_group = ortho
            else:
                truth_group = ortho[(ortho["orthoani"] >= low) & (ortho["orthoani"] < high)]
            truth_pairs = len(truth_group)
            keys = pd.MultiIndex.from_frame(truth_group[["genome1", "genome2"]])
            for method in METHODS:
                sub = long[long["method"] == method].set_index(["genome1", "genome2"]).loc[keys]
                computed = sub.dropna(subset=["ani_avg"]).copy()
                delta = computed["delta"]
                abs_delta = delta.abs()
                within_pairs = int((abs_delta <= tolerance).sum())
                runtime = runtimes[method]
                rows.append(
                    {
                        "tolerance": tolerance,
                        "orthoani_group": group_name,
                        "method": method,
                        "truth_pairs": truth_pairs,
                        "computed_pairs": int(len(computed)),
                        "computed_fraction": float(len(computed) / truth_pairs),
                        "within_pairs": within_pairs,
                        "ani_recovery": float(within_pairs / truth_pairs),
                        "mae": float(abs_delta.mean()),
                        "mse": float((delta * delta).mean()),
                        "rmse": float(np.sqrt((delta * delta).mean())),
                        "runtime_s": runtime,
                        "speedup_vs_fastani": float(fastani_time / runtime),
                    }
                )
    return pd.DataFrame(rows), joined


def plot_tradeoff(summary: pd.DataFrame, tolerance: float, out_pdf: Path) -> None:
    sub = summary[summary["tolerance"] == tolerance].copy()
    configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.0), sharey=True)

    max_x = max(10.0, float(sub["ani_recovery"].max() * 100.0) * 1.12)
    max_y = float(sub["speedup_vs_fastani"].max())
    y_ticks = [1, 3, 10, 30]
    if max_y >= 60:
        y_ticks.append(60)

    for ax, (group_name, _, _) in zip(axes, GROUPS):
        panel = sub[sub["orthoani_group"] == group_name].set_index("method").loc[METHODS]
        truth_pairs = int(panel["truth_pairs"].iloc[0])
        for method, row in panel.iterrows():
            x = row["ani_recovery"] * 100.0
            y = row["speedup_vs_fastani"]
            ax.scatter(
                x,
                y,
                s=105,
                marker=MARKERS[method],
                color=COLORS[method],
                edgecolor="black",
                linewidth=0.8,
                zorder=3,
            )
            ax.annotate(
                method,
                xy=(x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10.5,
                color="black",
            )

        ax.set_title(f"{group_name}\nn={truth_pairs:,}")
        ax.set_xlim(0, max_x)
        ax.set_yscale("log")
        ax.set_ylim(0.8, max(65.0, max_y * 1.25))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{tick:g}x" for tick in y_ticks])
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_xlabel(f"ANI recovery within +/-{tolerance:g}% (%)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Speedup vs local fastANI")
    handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[method],
            color="none",
            markerfacecolor=COLORS[method],
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=9,
            label=method,
        )
        for method in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=5,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    fig.savefig(out_pdf, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orthoani",
        type=Path,
        default=Path(
            "/Users/jianshuzhao/Library/Mobile Documents/com~apple~CloudDocs/"
            "TurboANI_paper/accuracy/orthoani_all_44850_pairs.results.tsv"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--tolerances", type=float, nargs="+", default=[1.0, 2.0])
    args = parser.parse_args()

    ortho = load_orthoani(args.orthoani)
    methods = load_method_outputs(args.out_dir)
    runtimes = load_runtimes(args.out_dir)
    summary, joined = summarize(ortho, methods, runtimes, args.tolerances)
    joined.to_csv(
        args.out_dir / "ani_speed_accuracy_local_300_joined.tsv",
        sep="\t",
        index=False,
        float_format="%.6f",
    )
    summary.to_csv(
        args.out_dir / "ani_speed_accuracy_local_300_summary.tsv",
        sep="\t",
        index=False,
        float_format="%.6f",
    )
    for tolerance in args.tolerances:
        suffix = str(tolerance).replace(".", "p")
        plot_tradeoff(
            summary,
            tolerance,
            args.out_dir / f"ani_speed_accuracy_local_300_tol{suffix}.pdf",
        )


if __name__ == "__main__":
    main()
