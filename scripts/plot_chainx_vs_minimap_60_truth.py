#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
for env_name, dirname in [("MPLCONFIGDIR", "mplconfig"), ("XDG_CACHE_HOME", "xdg-cache")]:
    cache_dir = SCRIPT_DIR / dirname
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(env_name, str(cache_dir))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        "font.size": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
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


ACCESSION_RE = re.compile(r"(GC[AF]_[0-9]+\.[0-9]+)")
TIME_RE = re.compile(r"timing total=([0-9.]+)s")
COUNTER_RE = re.compile(
    r"timing fragments=(?P<fragments>\d+) query_minimizers=(?P<query_minimizers>\d+) "
    r"seed_hits=(?P<seed_hits>\d+) l1_candidates=(?P<l1_candidates>\d+) "
    r"l2_candidates=(?P<l2_candidates>\d+) l2_windows=(?P<l2_windows>\d+)"
)
CPU_RE = re.compile(
    r"timing cpu-stage-sum query_minimizers=(?P<query_minimizers>[0-9.]+)s "
    r"query_sketch=(?P<query_sketch>[0-9.]+)s l1=(?P<l1>[0-9.]+)s "
    r"l2=(?P<l2>[0-9.]+)s .* l2_distance=(?P<l2_distance>[0-9.]+)s"
)
RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s+(\d+)")
USER_RE = re.compile(r"User time \(seconds\):\s+([0-9.]+)")
SYS_RE = re.compile(r"System time \(seconds\):\s+([0-9.]+)")
WALL_RE = re.compile(r"Elapsed \(wall clock\) time .*:\s+(.+)")


def accession(path_like: str) -> str:
    match = ACCESSION_RE.search(Path(path_like).name)
    if not match:
        match = ACCESSION_RE.search(path_like)
    if not match:
        raise ValueError(f"Could not parse accession from {path_like}")
    return match.group(1)


def pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


def read_path_ids(path_file: Path) -> list[str]:
    ids = []
    with path_file.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                ids.append(accession(line))
    return ids


def read_orthoani(path: Path, ids: set[str]) -> dict[tuple[str, str], float]:
    truth = {}
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            a = accession(row["genome1"])
            b = accession(row["genome2"])
            if a in ids and b in ids:
                truth[pair_key(a, b)] = float(row["orthoANI_value"])
    for genome_id in ids:
        truth[(genome_id, genome_id)] = 100.0
    return truth


def read_turboani(path: Path):
    values = {}
    directional_rows = 0
    with path.open() as handle:
        for raw in handle:
            if not raw.strip():
                continue
            fields = raw.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            a = accession(fields[0])
            b = accession(fields[1])
            ani = float(fields[2])
            values.setdefault(pair_key(a, b), []).append(ani)
            directional_rows += 1
    averaged = {key: sum(vals) / len(vals) for key, vals in values.items()}
    direction_counts = {key: len(vals) for key, vals in values.items()}
    return averaged, direction_counts, directional_rows


def parse_timing(stderr_path: Path) -> dict[str, float | int | str]:
    text = stderr_path.read_text(errors="replace")
    out = {}
    if match := TIME_RE.search(text):
        out["timing_total_sec"] = float(match.group(1))
    if match := COUNTER_RE.search(text):
        out.update({key: int(value) for key, value in match.groupdict().items()})
    if match := CPU_RE.search(text):
        out.update({f"cpu_{key}_sec": float(value) for key, value in match.groupdict().items()})
    if match := RSS_RE.search(text):
        out["max_rss_gib"] = int(match.group(1)) / 1024 / 1024
    if match := USER_RE.search(text):
        out["user_sec"] = float(match.group(1))
    if match := SYS_RE.search(text):
        out["sys_sec"] = float(match.group(1))
    if match := WALL_RE.search(text):
        out["gtime_wall"] = match.group(1).strip()
    return out


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def summarize(label, estimates, direction_counts, truth, directional_rows, stderr_path):
    rows = []
    for key, estimate in estimates.items():
        if key not in truth:
            continue
        rows.append((key[0], key[1], truth[key], estimate, estimate - truth[key], direction_counts[key]))
    diffs = np.array([row[4] for row in rows], dtype=float)
    x = [row[2] for row in rows]
    y = [row[3] for row in rows]
    timing = parse_timing(stderr_path)
    summary = {
        "method": label,
        "directional_rows": directional_rows,
        "averaged_pairs": len(estimates),
        "truth_matched_pairs": len(rows),
        "missing_truth_pairs": len(estimates) - len(rows),
        "one_direction_pairs": sum(1 for count in direction_counts.values() if count == 1),
        "two_direction_pairs": sum(1 for count in direction_counts.values() if count == 2),
        "mae": float(np.mean(np.abs(diffs))),
        "rmse": float(math.sqrt(np.mean(diffs * diffs))),
        "bias": float(np.mean(diffs)),
        "pearson": pearson(x, y),
        "max_abs_error": float(np.max(np.abs(diffs))),
    }
    summary.update(timing)
    return rows, summary


def write_table(path: Path, header, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--minimap", type=Path, required=True)
    parser.add_argument("--minimap-stderr", type=Path, required=True)
    parser.add_argument("--chainx", type=Path, required=True)
    parser.add_argument("--chainx-stderr", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ids = set(read_path_ids(args.paths))
    truth = read_orthoani(args.truth, ids)

    minimap_values, minimap_counts, minimap_directional = read_turboani(args.minimap)
    chainx_values, chainx_counts, chainx_directional = read_turboani(args.chainx)

    minimap_rows, minimap_summary = summarize(
        "minimap2-style chaining",
        minimap_values,
        minimap_counts,
        truth,
        minimap_directional,
        args.minimap_stderr,
    )
    chainx_rows, chainx_summary = summarize(
        "ChainX chaining",
        chainx_values,
        chainx_counts,
        truth,
        chainx_directional,
        args.chainx_stderr,
    )

    long_rows = []
    for method, rows in [
        ("minimap2-style chaining", minimap_rows),
        ("ChainX chaining", chainx_rows),
    ]:
        for a, b, truth_value, estimate, delta, directions in rows:
            long_rows.append((method, a, b, truth_value, estimate, delta, directions))
    write_table(
        args.out_dir / "chainx_vs_minimap_60truth_orthoani_long.tsv",
        ["method", "genome1", "genome2", "orthoani", "ani", "delta", "direction_count"],
        long_rows,
    )

    summary_rows = []
    summary_keys = sorted(set(minimap_summary) | set(chainx_summary))
    for summary in [minimap_summary, chainx_summary]:
        summary_rows.append([summary.get(key, "") for key in summary_keys])
    write_table(args.out_dir / "chainx_vs_minimap_60truth_summary.tsv", summary_keys, summary_rows)

    colors = {
        "minimap2-style chaining": "#386CB0",
        "ChainX chaining": "#E41A1C",
    }
    markers = {
        "minimap2-style chaining": "o",
        "ChainX chaining": "^",
    }

    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    stat_lines = []
    for method, rows, summary in [
        ("minimap2-style chaining", minimap_rows, minimap_summary),
        ("ChainX chaining", chainx_rows, chainx_summary),
    ]:
        x = [row[2] for row in rows]
        y = [row[3] for row in rows]
        stat_lines.append(
            f"{method}: MAE={summary['mae']:.3f}, RMSE={summary['rmse']:.3f}, r={summary['pearson']:.4f}"
        )
        ax.scatter(
            x,
            y,
            s=24,
            marker=markers[method],
            facecolor=colors[method],
            edgecolor="none",
            linewidth=0.0,
            alpha=0.42,
            label=method,
        )

    low = 74.5
    high = 100.6
    ax.plot([low, high], [low, high], color="black", linewidth=1.0, linestyle="--")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("OrthoANIu ANI (%)")
    ax.set_ylabel("TurboANI ANI (%)")
    ax.set_xticks([75, 80, 85, 90, 95, 100])
    ax.set_yticks([75, 80, 85, 90, 95, 100])
    ax.legend(
        frameon=False,
        loc="upper left",
        handletextpad=0.4,
        borderaxespad=0.2,
        fontsize=12,
    )
    ax.text(
        0.04,
        0.86,
        "\n".join(stat_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    fig.tight_layout()
    fig.savefig(args.out_dir / "chainx_vs_minimap_60truth_orthoani_correlation.pdf")
    fig.savefig(args.out_dir / "chainx_vs_minimap_60truth_orthoani_correlation.png", dpi=300)


if __name__ == "__main__":
    main()
