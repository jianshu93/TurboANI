#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


BASE = Path("/Users/jianshuzhao/Documents/Codex/2026-07-23/for/phage_oau_benchmark")
OUT = BASE / "best_virus_correlation"
OAU_TRUTH = BASE / "viral_85_100/oau_pairwise_full/teseptimavirus_60_oau_all_pairs.tsv"

METHOD_FILES = {
    "TurboANI fast": OUT / "teseptimavirus60_blast_turboani_fast_frag3000_ref40000.tsv",
    "TurboANI default": OUT / "teseptimavirus60_blast_turboani_default_frag3000_ref40000.tsv",
    "TurboANI slow": OUT / "teseptimavirus60_blast_turboani_slow_frag3000_ref40000.tsv",
    "FastANI": OUT / "teseptimavirus60_blast_fastani_frag3000.tsv",
    "skani": OUT / "teseptimavirus60_blast_skani.tsv",
}

COLORS = {
    "TurboANI fast": "#e41a1c",
    "TurboANI default": "#377eb8",
    "TurboANI slow": "#4daf4a",
    "FastANI": "#984ea3",
    "skani": "#ff7f00",
}
MARKERS = {
    "TurboANI fast": "^",
    "TurboANI default": "o",
    "TurboANI slow": "s",
    "FastANI": "D",
    "skani": "P",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 12,
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


def genome_name(value: str) -> str:
    return Path(value).name


def pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((genome_name(a), genome_name(b))))


def load_oau_truth(path: Path, min_ani: float | None = None) -> dict[tuple[str, str], float]:
    truth: dict[tuple[str, str], float] = {}
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            ani = float(row["orthoANI_value"])
            if ani < 0:
                continue
            if min_ani is not None and ani < min_ani:
                continue
            truth[pair_key(row["genome_a"], row["genome_b"])] = ani
    return truth


def load_predictions(path: Path, method: str) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = {}
    with path.open() as handle:
        for raw in handle:
            if not raw.strip() or raw.startswith("#"):
                continue
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            a = genome_name(parts[0])
            b = genome_name(parts[1])
            if a == b:
                continue
            try:
                ani = float(parts[2])
            except ValueError:
                continue
            if not math.isfinite(ani):
                continue
            if method == "skani" and ani <= 1.5:
                ani *= 100.0
            grouped.setdefault(tuple(sorted((a, b))), []).append(ani)
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def pearson(xs: list[float], ys: list[float]) -> float:
    if not xs:
        return math.nan
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    denom = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    if denom == 0.0:
        return math.nan
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_joined(truth: dict[tuple[str, str], float]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    joined: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for method, path in METHOD_FILES.items():
        pred = load_predictions(path, "skani" if method == "skani" else method)
        xs: list[float] = []
        ys: list[float] = []
        diffs: list[float] = []
        for key, true_ani in truth.items():
            if key not in pred:
                continue
            estimate = pred[key]
            xs.append(true_ani)
            ys.append(estimate)
            diffs.append(estimate - true_ani)
            joined.append(
                {
                    "method": method,
                    "genome_a": key[0],
                    "genome_b": key[1],
                    "orthoani": true_ani,
                    "estimated_ani": estimate,
                    "delta": estimate - true_ani,
                }
            )
        absdiff = [abs(d) for d in diffs]
        n = len(diffs)
        summary.append(
            {
                "method": method,
                "truth_pairs": len(truth),
                "computed_pairs": n,
                "missing_pairs": len(truth) - n,
                "pearson": pearson(xs, ys),
                "mae": sum(absdiff) / n if n else math.nan,
                "rmse": math.sqrt(sum(d * d for d in diffs) / n) if n else math.nan,
                "bias": sum(diffs) / n if n else math.nan,
                "within_0_5": 100.0 * sum(d <= 0.5 for d in absdiff) / n if n else 0.0,
                "within_1_0": 100.0 * sum(d <= 1.0 for d in absdiff) / n if n else 0.0,
                "within_2_0": 100.0 * sum(d <= 2.0 for d in absdiff) / n if n else 0.0,
            }
        )
    return joined, summary


def plot(joined: list[dict[str, object]], out_pdf: Path, axis_min: float, axis_max: float) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([axis_min, axis_max], [axis_min, axis_max], color="0.25", linewidth=1.1, zorder=0)
    for method in ("TurboANI fast", "TurboANI default", "TurboANI slow", "FastANI", "skani"):
        points = [row for row in joined if row["method"] == method]
        if not points:
            continue
        ax.scatter(
            [float(row["orthoani"]) for row in points],
            [float(row["estimated_ani"]) for row in points],
            s=22,
            marker=MARKERS[method],
            color=COLORS[method],
            alpha=0.55,
            edgecolors="none",
            label=method,
        )
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("OrthoANIu ANI (%)")
    ax.set_ylabel("Estimated ANI (%)")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    ax.legend(frameon=False, loc="lower right", handletextpad=0.2, borderpad=0.1)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    full_truth = load_oau_truth(OAU_TRUTH, min_ani=None)
    full_joined, full_summary = build_joined(full_truth)
    write_table(OUT / "teseptimavirus60_oau_successful_vs_methods_joined.tsv", full_joined)
    write_table(OUT / "teseptimavirus60_oau_successful_vs_methods_summary.tsv", full_summary)
    plot(full_joined, OUT / "teseptimavirus60_oau_successful_vs_methods_axis60_100.pdf", 60.0, 100.0)

    high_truth = load_oau_truth(OAU_TRUTH, min_ani=85.0)
    high_joined, high_summary = build_joined(high_truth)
    write_table(OUT / "teseptimavirus60_oau_successful_85_100_vs_methods_joined.tsv", high_joined)
    write_table(OUT / "teseptimavirus60_oau_successful_85_100_vs_methods_summary.tsv", high_summary)
    plot(high_joined, OUT / "teseptimavirus60_oau_successful_85_100_vs_methods_axis85_100.pdf", 85.0, 100.0)

    print(OUT / "teseptimavirus60_oau_successful_85_100_vs_methods_axis85_100.pdf")
    print(OUT / "teseptimavirus60_oau_successful_vs_methods_axis60_100.pdf")


if __name__ == "__main__":
    main()
