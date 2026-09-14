#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path("/Users/jianshuzhao/Documents/Codex/2026-07-23/for/outputs/paired_anchor_filter_60_orthoani")
RUNS = OUT / "runs"
ORTHO = Path(
    "/Users/jianshuzhao/Library/Mobile Documents/com~apple~CloudDocs/TurboANI_paper/accuracy/orthoani_all_44850_pairs.results.tsv"
)

METHODS = [
    ("default", "Default", "#386CB0", "o"),
    ("compound", "Paired-minimizer filter", "#E41A1C", "s"),
]

BINS = [(70, 75), (75, 80), (80, 85), (85, 90), (90, 95), (95, 100.000001)]


def genome_key(value: str) -> str:
    name = Path(value).name
    if name.endswith(".gz"):
        name = name[:-3]
    for suffix in (".fna", ".fa", ".fasta"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def read_directed_ani(path: Path) -> dict[tuple[str, str], float]:
    rows: dict[tuple[str, str], list[float]] = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            ani = float(parts[2])
            if ani < 0:
                continue
            rows[pair_key(genome_key(parts[0]), genome_key(parts[1]))].append(ani)
    return {key: sum(values) / len(values) for key, values in rows.items()}


def read_truth(keep: set[str]) -> dict[tuple[str, str], float]:
    truth: dict[tuple[str, str], float] = {}
    with ORTHO.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            a = genome_key(row["genome1"])
            b = genome_key(row["genome2"])
            if a in keep and b in keep:
                truth[pair_key(a, b)] = float(row["orthoANI_value"])
    for sample in keep:
        truth[(sample, sample)] = 100.0
    return truth


def metric_summary(truth: dict[tuple[str, str], float], estimate: dict[tuple[str, str], float]) -> dict[str, float]:
    common = sorted(set(truth) & set(estimate))
    deltas = np.array([estimate[key] - truth[key] for key in common], dtype=float)
    truth_values = np.array([truth[key] for key in common], dtype=float)
    est_values = np.array([estimate[key] for key in common], dtype=float)
    return {
        "truth_pairs": len(truth),
        "estimated_pairs": len(estimate),
        "common_pairs": len(common),
        "missing_pairs": len(set(truth) - set(estimate)),
        "pearson": float(np.corrcoef(truth_values, est_values)[0, 1]) if len(common) > 1 else math.nan,
        "mae": float(np.mean(np.abs(deltas))),
        "rmse": float(np.sqrt(np.mean(deltas * deltas))),
        "bias": float(np.mean(deltas)),
        "recovery_1pct_common": float(np.mean(np.abs(deltas) <= 1.0)),
        "recovery_2pct_common": float(np.mean(np.abs(deltas) <= 2.0)),
        "recovery_1pct_all": float(sum(abs(estimate.get(key, math.inf) - value) <= 1.0 for key, value in truth.items()) / len(truth)),
        "recovery_2pct_all": float(sum(abs(estimate.get(key, math.inf) - value) <= 2.0 for key, value in truth.items()) / len(truth)),
    }


def binned_rows(truth: dict[tuple[str, str], float], estimates: dict[str, dict[tuple[str, str], float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for lo, hi in BINS:
        keys = [key for key, value in truth.items() if lo <= value < hi]
        if not keys:
            continue
        label = f"{lo}-{int(hi) if hi <= 100 else 100}"
        for method_key, method_label, _color, _marker in METHODS:
            est = estimates[method_key]
            common = [key for key in keys if key in est]
            deltas = np.array([est[key] - truth[key] for key in common], dtype=float)
            recovered_all = sum(
                key in est and abs(est[key] - truth[key]) <= 1.0
                for key in keys
            )
            rows.append(
                {
                    "bin": label,
                    "method_key": method_key,
                    "method": method_label,
                    "truth_pairs": len(keys),
                    "computed_pairs": len(common),
                    "computed_fraction": len(common) / len(keys),
                    "mae": float(np.mean(np.abs(deltas))) if len(deltas) else math.nan,
                    "rmse": float(np.sqrt(np.mean(deltas * deltas))) if len(deltas) else math.nan,
                    "bias": float(np.mean(deltas)) if len(deltas) else math.nan,
                    "recovery_1pct_all": recovered_all / len(keys),
                }
            )
    return rows


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
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


def write_tables(summary: list[dict[str, object]], binned: list[dict[str, object]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "paired_minimizer_filter_60_orthoani_summary.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    with (OUT / "paired_minimizer_filter_60_orthoani_binned.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(binned[0].keys()))
        writer.writeheader()
        writer.writerows(binned)


def plot_binned(binned: list[dict[str, object]]) -> None:
    setup_style()
    bins = []
    for lo, hi in BINS:
        label = f"{lo}-{int(hi) if hi <= 100 else 100}"
        if any(row["bin"] == label for row in binned):
            bins.append(label)
    x = np.arange(len(bins), dtype=float)
    offsets = np.linspace(-0.18, 0.18, len(METHODS))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharex=True)
    row_lookup = {(row["bin"], row["method_key"]): row for row in binned}

    for offset, (method_key, method_label, color, marker) in zip(offsets, METHODS):
        mae = [row_lookup[(label, method_key)]["mae"] for label in bins]
        recovery = [100.0 * row_lookup[(label, method_key)]["recovery_1pct_all"] for label in bins]
        axes[0].plot(x + offset, mae, marker=marker, ms=6.5, lw=1.8, color=color, label=method_label)
        axes[1].plot(x + offset, recovery, marker=marker, ms=6.5, lw=1.8, color=color, label=method_label)

    axes[0].set_ylabel("MAE vs OrthoANIu (%)")
    axes[1].set_ylabel("ANI recovery within 1% (%)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=35, ha="right")
        ax.set_xlabel("OrthoANIu bin (%)")
        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(1.0)
        ax.tick_params(width=1.0, length=4)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(0, 102)
    axes[1].legend(frameon=False, loc="lower right", handletextpad=0.4)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "paired_minimizer_filter_60_orthoani_accuracy.pdf")
    fig.savefig(OUT / "paired_minimizer_filter_60_orthoani_accuracy.png", dpi=300)
    plt.close(fig)


def main() -> None:
    estimates = {
        method_key: read_directed_ani(RUNS / f"{method_key}_60truth.tsv")
        for method_key, *_ in METHODS
    }
    keep = {sample for estimate in estimates.values() for pair in estimate for sample in pair}
    truth = read_truth(keep)
    summary = [
        {"method_key": method_key, "method": method_label, **metric_summary(truth, estimates[method_key])}
        for method_key, method_label, _color, _marker in METHODS
    ]
    binned = binned_rows(truth, estimates)
    write_tables(summary, binned)
    plot_binned(binned)


if __name__ == "__main__":
    main()
