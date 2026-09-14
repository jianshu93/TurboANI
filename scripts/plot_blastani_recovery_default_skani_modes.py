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


METHODS = [
    ("FastANI", "#7FC97F"),
    ("skani default", "#386CB0"),
    ("skani sensitive", "#A6CEE3"),
    ("TurboANI default", "#BEAED4"),
]


def genome_key(value: object) -> str:
    name = Path(str(value)).name
    if name.endswith(".gz"):
        name = name[:-3]
    return name


def pair_key(a: object, b: object) -> tuple[str, str]:
    x = genome_key(a)
    y = genome_key(b)
    return (x, y) if x <= y else (y, x)


def read_blastani_truth(path: str) -> dict[tuple[str, str], float]:
    truth = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            ani = float(row["rbm_ani"])
            if ani >= 0:
                truth[pair_key(row["genome1"], row["genome2"])] = ani
    return truth


def read_ani_table(path: str, scale: float = 1.0, has_header: bool = False) -> dict[tuple[str, str], float]:
    values: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    with open(path, newline="") as handle:
        if has_header:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                ref = row.get("Ref_file") or row.get("reference")
                qry = row.get("Query_file") or row.get("query")
                ani = float(row.get("ANI") or row.get("ani"))
                if ref is None or qry is None or genome_key(ref) == genome_key(qry) or ani < 0:
                    continue
                values[pair_key(ref, qry)].append(ani * scale)
        else:
            for line in handle:
                if not line.strip():
                    continue
                cols = line.rstrip("\n").split()
                if len(cols) < 3:
                    continue
                a, b = cols[0], cols[1]
                ani = float(cols[2])
                if genome_key(a) == genome_key(b) or ani < 0:
                    continue
                values[pair_key(a, b)].append(ani * scale)
    return {key: float(np.mean(vals)) for key, vals in values.items()}


def recovery_summary(truth: dict[tuple[str, str], float], predictions: dict[str, dict[tuple[str, str], float]]) -> list[dict[str, object]]:
    rows = []
    for method, pred in predictions.items():
        matched = 0
        abs_errors = []
        for key, truth_ani in truth.items():
            if key not in pred:
                continue
            matched += 1
            abs_errors.append(abs(pred[key] - truth_ani))
        abs_errors_arr = np.asarray(abs_errors, dtype=float)
        for tolerance in (0.5, 1.0, 2.0):
            within = int(np.sum(abs_errors_arr <= tolerance))
            rows.append(
                {
                    "method": method,
                    "tolerance": tolerance,
                    "truth_pairs": len(truth),
                    "matched_pairs": matched,
                    "matched_fraction": matched / len(truth),
                    "within_pairs": within,
                    "ani_recovery": within / len(truth),
                }
            )
    return rows


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
            "font.size": 20,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 18,
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


def plot_bar(rows: list[dict[str, object]], out_pdf: str) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    tolerances = sorted({float(row["tolerance"]) for row in rows})
    x = np.arange(len(tolerances))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(METHODS))
    row_map = {(row["method"], row["tolerance"]): row for row in rows}
    for offset, (method, color) in zip(offsets, METHODS):
        values = [100.0 * float(row_map[(method, tol)]["ani_recovery"]) for tol in tolerances]
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=color,
            edgecolor="none",
            linewidth=0,
            label=method,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.0,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=10.5,
                rotation=90,
                clip_on=False,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"±{tol:g}%" for tol in tolerances])
    ax.set_ylabel("ANI recovery (%)")
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        handlelength=1.0,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.78)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--fastani", required=True)
    parser.add_argument("--skani-default", required=True)
    parser.add_argument("--skani-sensitive", required=True)
    parser.add_argument("--turboani-default", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    truth = read_blastani_truth(args.truth)
    predictions = {
        "FastANI": read_ani_table(args.fastani, scale=1.0),
        "skani default": read_ani_table(args.skani_default, scale=1.0, has_header=True),
        "skani sensitive": read_ani_table(args.skani_sensitive, scale=100.0),
        "TurboANI default": read_ani_table(args.turboani_default, scale=1.0),
    }
    rows = recovery_summary(truth, predictions)
    out_summary = os.path.join(args.out_dir, "blastani_recovery_default_skani_modes_summary.tsv")
    with open(out_summary, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "tolerance",
                "truth_pairs",
                "matched_pairs",
                "matched_fraction",
                "within_pairs",
                "ani_recovery",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    plot_bar(rows, os.path.join(args.out_dir, "blastani_recovery_default_skani_modes.pdf"))
    for row in rows:
        print(
            row["method"],
            f"tol={row['tolerance']}",
            f"matched={row['matched_pairs']}/{row['truth_pairs']}",
            f"within={row['within_pairs']}",
            f"recovery={100 * row['ani_recovery']:.2f}%",
            sep="\t",
        )


if __name__ == "__main__":
    main()
