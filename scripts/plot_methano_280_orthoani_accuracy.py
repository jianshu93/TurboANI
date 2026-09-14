#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ACCESSION_RE = re.compile(r"(GC[AF]_[0-9]+\.[0-9]+)")


def genome_key(value: object) -> str:
    text = str(value)
    match = ACCESSION_RE.search(text)
    if match:
        return match.group(1)
    name = Path(text).name
    for suffix in [".gz", ".fna", ".fa", ".fasta", ".fastq", ".fq"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def load_orthoani(path: Path, genome_paths: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t")
    raw["genome1"] = raw["genome1"].map(genome_key)
    raw["genome2"] = raw["genome2"].map(genome_key)

    rows = []
    genomes: set[str] = set()
    with genome_paths.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                genomes.add(genome_key(line))

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
    return (
        ortho.drop_duplicates(["genome1", "genome2"], keep="first")
        .sort_values(["genome1", "genome2"])
        .reset_index(drop=True)
    )


def read_directed_ani(path: Path, *, fraction: bool = False) -> pd.DataFrame:
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
    raw[["genome1", "genome2"]] = pd.DataFrame(
        [pair_key(q, r) for q, r in zip(raw["query"], raw["reference"])],
        index=raw.index,
    )
    raw["ani"] = pd.to_numeric(raw["ani"], errors="coerce")
    raw = raw[np.isfinite(raw["ani"]) & (raw["ani"] >= 0)].copy()
    if fraction:
        raw["ani"] *= 100.0
    return raw


def average_reciprocal_pairs(path: Path, *, fraction: bool = False) -> pd.DataFrame:
    raw = read_directed_ani(path, fraction=fraction)
    return (
        raw.groupby(["genome1", "genome2"], as_index=False)
        .agg(method_ani_avg=("ani", "mean"), direction_count=("ani", "size"))
        .sort_values(["genome1", "genome2"])
    )


def skani_seen_counts(path: Path) -> pd.DataFrame:
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
    raw[["genome1", "genome2"]] = pd.DataFrame(
        [pair_key(q, r) for q, r in zip(raw["query"], raw["reference"])],
        index=raw.index,
    )
    raw["ani"] = pd.to_numeric(raw["ani"], errors="coerce")
    raw["computed"] = np.isfinite(raw["ani"]) & (raw["ani"] >= 0)
    seen = raw.groupby(["genome1", "genome2"], as_index=False).size()
    seen = seen.rename(columns={"size": "seen_direction_count"})
    skipped = raw[~raw["computed"]].groupby(["genome1", "genome2"], as_index=False).size()
    skipped = skipped.rename(columns={"size": "skipped_direction_count"})
    return seen.merge(skipped, how="left", on=["genome1", "genome2"]).fillna(
        {"skipped_direction_count": 0}
    )


def correlation_stats(x: pd.Series, y: pd.Series) -> dict[str, float]:
    arr = pd.DataFrame({"x": x, "y": y}).dropna()
    xval = arr["x"].to_numpy()
    yval = arr["y"].to_numpy()
    pearson = float(np.corrcoef(xval, yval)[0, 1]) if len(arr) > 1 else float("nan")
    xr = arr["x"].rank(method="average").to_numpy()
    yr = arr["y"].rank(method="average").to_numpy()
    spearman = float(np.corrcoef(xr, yr)[0, 1]) if len(arr) > 1 else float("nan")
    delta = yval - xval
    return {
        "n_pairs": int(len(arr)),
        "pearson": pearson,
        "spearman": spearman,
        "mae": float(np.mean(np.abs(delta))),
        "median_abs_delta": float(np.median(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "mean_delta": float(np.mean(delta)),
        "min_delta": float(np.min(delta)),
        "max_delta": float(np.max(delta)),
        "min_truth": float(np.min(xval)),
        "max_truth": float(np.max(xval)),
    }


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
            "legend.fontsize": 16,
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


def add_method(
    ortho: pd.DataFrame,
    method: str,
    path: Path,
    *,
    fraction: bool = False,
) -> tuple[pd.DataFrame, dict[str, float]]:
    averaged = average_reciprocal_pairs(path, fraction=fraction)
    method_df = ortho.merge(averaged, how="left", on=["genome1", "genome2"])
    method_df = method_df.dropna(subset=["method_ani_avg"]).copy()
    method_df["method"] = method
    method_df["delta_vs_orthoani"] = method_df["method_ani_avg"] - method_df["orthoani"]
    summary = {"method": method, **correlation_stats(method_df["orthoani"], method_df["method_ani_avg"])}
    summary["directional_rows"] = int(len(read_directed_ani(path, fraction=fraction)))
    summary["orthoani_pairs_with_self"] = int(len(ortho))
    summary["missing_or_skipped_pairs"] = int(len(ortho) - len(method_df))
    return method_df, summary


def plot_methods(
    long: pd.DataFrame,
    summary: pd.DataFrame,
    methods: list[str],
    out_prefix: Path,
    *,
    ylabel: str,
    colors: dict[str, str],
    markers: dict[str, str],
) -> None:
    configure_matplotlib()
    low = math.floor(min(long["orthoani"].min(), long["method_ani_avg"].min()) / 5.0) * 5.0
    high = 100.5
    low = min(low, 65.0)

    fig, ax = plt.subplots(figsize=(10.4, 8.8))
    for method in methods:
        sub = long[long["method"] == method]
        ax.scatter(
            sub["orthoani"],
            sub["method_ani_avg"],
            s=12,
            marker=markers[method],
            color=colors[method],
            alpha=0.38,
            linewidths=0,
            rasterized=True,
        )

    ax.plot([low, high], [low, high], color="black", linewidth=1.0, linestyle="--")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("OrthoANIu (%)")
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")

    handles = []
    for method in methods:
        row = summary[summary["method"] == method].iloc[0]
        label = f"{method}  r={row.pearson:.4f}, MAE={row.mae:.3f}%"
        handles.append(
            Line2D(
                [0],
                [0],
                marker=markers[method],
                color="none",
                markerfacecolor=colors[method],
                markeredgewidth=0,
                markersize=8,
                label=label,
            )
        )
    ax.legend(handles=handles, loc="upper left", frameon=False, handletextpad=0.4)
    fig.savefig(out_prefix.with_name(out_prefix.name + ".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_name(out_prefix.name + ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    base = args.base

    ortho = load_orthoani(
        Path("/Users/jianshuzhao/Documents/Methanobacteriaceae_orthoANIu_280.all_results.tsv"),
        base / "methano_280_paths.txt",
    )

    main_inputs = [
        ("FastANI", base / "methano_280_fastani.tsv", False),
        ("TurboANI sensitive 77.5", base / "methano_280_turboani_sensitive_id775.tsv", False),
        ("TurboANI default 80", base / "methano_280_turboani_default_id80.tsv", False),
        ("TurboANI fast 82.5", base / "methano_280_turboani_fast_id825.tsv", False),
    ]
    long_rows = []
    summary_rows = []
    for method, path, fraction in main_inputs:
        method_df, summary = add_method(ortho, method, path, fraction=fraction)
        long_rows.append(method_df)
        summary_rows.append(summary)
    main_long = pd.concat(long_rows, ignore_index=True)
    main_summary = pd.DataFrame(summary_rows)
    main_long.to_csv(base / "methano_280_orthoani_vs_fastani_turboani_long.tsv", sep="\t", index=False, float_format="%.6f")
    main_summary.to_csv(base / "methano_280_orthoani_vs_fastani_turboani_summary.tsv", sep="\t", index=False, float_format="%.6f")

    colors = {
        "FastANI": "#7FC97F",
        "TurboANI sensitive 77.5": "#E41A1C",
        "TurboANI default 80": "#BEAED4",
        "TurboANI fast 82.5": "#FDC086",
    }
    markers = {
        "FastANI": "o",
        "TurboANI sensitive 77.5": "^",
        "TurboANI default 80": "s",
        "TurboANI fast 82.5": "v",
    }
    plot_methods(
        main_long,
        main_summary,
        [item[0] for item in main_inputs],
        base / "methano_280_orthoani_vs_fastani_turboani_correlation",
        ylabel="ANI estimate (%)",
        colors=colors,
        markers=markers,
    )

    skani_long, skani_summary = add_method(
        ortho,
        "skani",
        base / "methano_280_skani_superani.tsv",
        fraction=True,
    )
    seen = skani_seen_counts(base / "methano_280_skani_superani.tsv")
    skani_joined = ortho.merge(
        skani_long[
            [
                "genome1",
                "genome2",
                "method_ani_avg",
                "direction_count",
                "delta_vs_orthoani",
            ]
        ],
        how="left",
        on=["genome1", "genome2"],
    ).merge(seen, how="left", on=["genome1", "genome2"])
    for col in ["direction_count", "seen_direction_count", "skipped_direction_count"]:
        skani_joined[col] = skani_joined[col].fillna(0).astype(int)
    skani_joined["status"] = np.where(skani_joined["method_ani_avg"].notna(), "computed", "skipped")
    skani_summary["seen_directional_rows"] = int(skani_joined["seen_direction_count"].sum())
    skani_summary["skipped_directional_rows"] = int(skani_joined["skipped_direction_count"].sum())
    skani_summary["pairs_with_any_skipped_direction"] = int((skani_joined["skipped_direction_count"] > 0).sum())
    pd.DataFrame([skani_summary]).to_csv(base / "methano_280_orthoani_vs_skani_summary.tsv", sep="\t", index=False, float_format="%.6f")
    skani_joined.to_csv(base / "methano_280_orthoani_vs_skani_joined.tsv", sep="\t", index=False, float_format="%.6f")
    skani_long.to_csv(base / "methano_280_orthoani_vs_skani_long.tsv", sep="\t", index=False, float_format="%.6f")
    plot_methods(
        skani_long,
        pd.DataFrame([skani_summary]),
        ["skani"],
        base / "methano_280_orthoani_vs_skani_correlation",
        ylabel="skani ANI (%)",
        colors={"skani": "#386CB0"},
        markers={"skani": "s"},
    )


if __name__ == "__main__":
    main()
