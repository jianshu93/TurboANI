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


def genome_key(value: object) -> str:
    name = Path(str(value)).name
    if name.endswith(".gz"):
        name = name[:-3]
    return name


def pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def load_path_keys(path_list: Path) -> set[str]:
    return {genome_key(line.strip()) for line in path_list.read_text().splitlines() if line.strip()}


def load_orthoani(path: Path, keep: set[str]) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t")
    raw["genome1"] = raw["genome1"].map(genome_key)
    raw["genome2"] = raw["genome2"].map(genome_key)
    raw = raw[raw["genome1"].isin(keep) & raw["genome2"].isin(keep)].copy()
    raw[["genome1", "genome2"]] = pd.DataFrame(
        [pair_key(a, b) for a, b in zip(raw["genome1"], raw["genome2"])],
        index=raw.index,
    )
    ortho = raw[["genome1", "genome2", "orthoANI_value"]].rename(
        columns={"orthoANI_value": "orthoani"}
    )
    self_rows = pd.DataFrame(
        [(g, g, 100.0) for g in sorted(keep)],
        columns=["genome1", "genome2", "orthoani"],
    )
    return (
        pd.concat([ortho, self_rows], ignore_index=True)
        .drop_duplicates(["genome1", "genome2"], keep="first")
        .sort_values(["genome1", "genome2"])
        .reset_index(drop=True)
    )


def average_directed(path: Path, keep: set[str]) -> pd.DataFrame:
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
    raw = raw[raw["query"].isin(keep) & raw["reference"].isin(keep) & (raw["ani"] >= 0)].copy()
    raw[["genome1", "genome2"]] = pd.DataFrame(
        [pair_key(q, r) for q, r in zip(raw["query"], raw["reference"])],
        index=raw.index,
    )
    return raw.groupby(["genome1", "genome2"], as_index=False).agg(
        method_ani=("ani", "mean"),
        direction_count=("ani", "size"),
    )


def parse_debug_timing(stderr_path: Path) -> dict[str, float]:
    text = stderr_path.read_text(errors="replace")
    values: dict[str, float] = {}
    patterns = {
        "total_s": r"timing total=([0-9.]+)s",
        "reference_s": r"reference=([0-9.]+)s",
        "query_map_wall_sum_s": r"query_map_wall_sum=([0-9.]+)s",
        "query_minimizer_s": r"query_minimizers=([0-9.]+)s",
        "query_sketch_s": r"query_sketch=([0-9.]+)s",
        "l1_s": r"l1=([0-9.]+)s",
        "l2_s": r"l2=([0-9.]+)s",
        "l2_distance_s": r"l2_distance=([0-9.]+)s",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        values[key] = float(match.group(1)) if match else math.nan
    count_line = re.search(
        r"timing fragments=(\d+) query_minimizers=(\d+) seed_hits=(\d+) "
        r"l1_candidates=(\d+) l2_candidates=(\d+) l2_windows=(\d+) "
        r"l2_ref_sketches=(\d+) mappings=(\d+)",
        text,
    )
    if count_line:
        for key, value in zip(
            [
                "fragments",
                "query_minimizers",
                "seed_hits",
                "l1_candidates",
                "l2_candidates",
                "l2_windows",
                "l2_ref_sketches",
                "mappings",
            ],
            count_line.groups(),
        ):
            values[key] = float(value)
    return values


def stats(truth: pd.Series, estimate: pd.Series) -> dict[str, float]:
    delta = estimate.to_numpy() - truth.to_numpy()
    return {
        "n_pairs": float(len(delta)),
        "pearson": float(np.corrcoef(truth, estimate)[0, 1]) if len(delta) > 1 else math.nan,
        "mae": float(np.mean(np.abs(delta))) if len(delta) else math.nan,
        "rmse": float(np.sqrt(np.mean(delta * delta))) if len(delta) else math.nan,
        "mean_delta": float(np.mean(delta)) if len(delta) else math.nan,
        "median_abs_delta": float(np.median(np.abs(delta))) if len(delta) else math.nan,
        "max_abs_delta": float(np.max(np.abs(delta))) if len(delta) else math.nan,
        "recovery_1pct": float(np.mean(np.abs(delta) <= 1.0)) if len(delta) else math.nan,
        "recovery_2pct": float(np.mean(np.abs(delta) <= 2.0)) if len(delta) else math.nan,
    }


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 13,
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


METHODS = [
    ("Minimizer K=16", "default_syncmer60.tsv", "default_syncmer60.stderr", "#386CB0", "s"),
    ("Minimizer K=15", "default_k15.tsv", "default_k15.stderr", "#7FC97F", "o"),
    ("Open syncmer K=15", "open_syncmer_simd_k15.tsv", "open_syncmer_simd_k15.stderr", "#FDC086", "^"),
    ("Closed syncmer K=15", "closed_syncmer_simd_k15.tsv", "closed_syncmer_simd_k15.stderr", "#E41A1C", "D"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--path-list",
        type=Path,
        default=Path(
            "/Users/jianshuzhao/Documents/Codex/2026-07-23/for/outputs/"
            "turboani_minmer_scalar_exp_20260820_001916/results/strep_60_paths.txt"
        ),
    )
    parser.add_argument(
        "--truth",
        type=Path,
        default=Path(
            "/Users/jianshuzhao/Library/Mobile Documents/com~apple~CloudDocs/"
            "TurboANI_paper/accuracy/orthoani_all_44850_pairs.results.tsv"
        ),
    )
    args = parser.parse_args()

    keep = load_path_keys(args.path_list)
    truth = load_orthoani(args.truth, keep)

    long_rows = []
    summary_rows = []
    for method, table_name, stderr_name, _color, _marker in METHODS:
        table_path = args.out_dir / table_name
        if not table_path.exists():
            continue
        method_df = truth.merge(average_directed(table_path, keep), how="left", on=["genome1", "genome2"])
        computed = method_df.dropna(subset=["method_ani"]).copy()
        computed["method"] = method
        computed["delta"] = computed["method_ani"] - computed["orthoani"]
        long_rows.append(computed)
        row = {"method": method, **stats(computed["orthoani"], computed["method_ani"])}
        row["truth_pairs_with_self"] = float(len(truth))
        row["missing_pairs"] = float(len(truth) - len(computed))
        row["computed_fraction"] = float(len(computed) / len(truth)) if len(truth) else math.nan
        row["directional_rows"] = float(computed["direction_count"].sum()) if len(computed) else 0.0
        row.update(parse_debug_timing(args.out_dir / stderr_name))
        summary_rows.append(row)

    long = pd.concat(long_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    long.to_csv(args.out_dir / "correct_syncmer_60_vs_orthoani_long.tsv", sep="\t", index=False)
    summary.to_csv(args.out_dir / "correct_syncmer_60_vs_orthoani_summary.tsv", sep="\t", index=False)

    setup_style()
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    for method, _table_name, _stderr_name, color, marker in METHODS:
        sub = long[long["method"] == method]
        if sub.empty:
            continue
        ax.scatter(
            sub["orthoani"],
            sub["method_ani"],
            s=18 if "Minimizer" in method else 32,
            marker=marker,
            color=color,
            alpha=0.5 if "Minimizer" in method else 0.65,
            linewidths=0,
            label=method,
        )
    ax.plot([70, 100], [70, 100], color="black", lw=1)
    ax.set_xlim(70, 100.5)
    ax.set_ylim(70, 100.5)
    ax.set_xlabel("OrthoANIu (%)")
    ax.set_ylabel("Estimated ANI (%)")
    ax.legend(frameon=False, loc="lower right", handletextpad=0.25)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.0)
    fig.tight_layout()
    fig.savefig(args.out_dir / "correct_syncmer_60_vs_orthoani_correlation.pdf")
    fig.savefig(args.out_dir / "correct_syncmer_60_vs_orthoani_correlation.png", dpi=320)

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    timing = summary.set_index("method").loc[[m[0] for m in METHODS if m[0] in set(summary["method"])]]
    x = np.arange(len(timing))
    ax.bar(x, timing["total_s"], color=[m[3] for m in METHODS if m[0] in timing.index], width=0.68)
    ax.set_xticks(x)
    ax.set_xticklabels(timing.index, rotation=35, ha="right")
    ax.set_ylabel("Wall time (s)")
    for xpos, value in zip(x, timing["total_s"]):
        ax.text(xpos, value, f"{value:.1f}", ha="center", va="bottom", fontsize=13)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.0)
    fig.tight_layout()
    fig.savefig(args.out_dir / "correct_syncmer_60_runtime.pdf")
    fig.savefig(args.out_dir / "correct_syncmer_60_runtime.png", dpi=320)


if __name__ == "__main__":
    main()
