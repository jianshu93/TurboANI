#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_DIR / "cache"))
(SCRIPT_DIR / "mplconfig").mkdir(parents=True, exist_ok=True)
(SCRIPT_DIR / "cache").mkdir(parents=True, exist_ok=True)

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


METHODS = [
    "FastANI",
    "skani",
    "TurboANI sensitive 77.5",
    "TurboANI default 80",
    "TurboANI fast 82.5",
]
LABELS = {
    "FastANI": "fastANI",
    "skani": "skani",
    "TurboANI sensitive 77.5": "TurboANI sensitive",
    "TurboANI default 80": "TurboANI default",
    "TurboANI fast 82.5": "TurboANI fast",
}
COLORS = {
    "FastANI": "#7FC97F",
    "skani": "#386CB0",
    "TurboANI sensitive 77.5": "#E41A1C",
    "TurboANI default 80": "#BEAED4",
    "TurboANI fast 82.5": "#FDC086",
}
MARKERS = {
    "FastANI": "o",
    "skani": "D",
    "TurboANI sensitive 77.5": "^",
    "TurboANI default 80": "s",
    "TurboANI fast 82.5": "v",
}
GROUPS = [
    ("All pairs", None, None),
    ("70-90", 70.0, 90.0),
    ("75-90", 75.0, 90.0),
]
RUNTIME_FILES = {
    "FastANI": "methano_280_fastani.stderr",
    "skani": "methano_280_skani_superani.stderr",
    "TurboANI sensitive 77.5": "methano_280_turboani_sensitive_id775.stderr",
    "TurboANI default 80": "methano_280_turboani_default_id80.stderr",
    "TurboANI fast 82.5": "methano_280_turboani_fast_id825.stderr",
}


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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def parse_elapsed_seconds(path: Path) -> float:
    text = path.read_text(errors="ignore")
    match = re.search(r"Elapsed \(wall clock\) time .*?:\s*([0-9:.]+)", text)
    if match:
        parts = [float(part) for part in match.group(1).split(":")]
        if len(parts) == 3:
            return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60.0 + parts[1]
        return parts[0]
    match = re.search(r"\b([0-9.]+)\s+real\b", text)
    if match:
        return float(match.group(1))
    raise ValueError(f"could not parse elapsed time from {path}")


def load_runtime(base: Path) -> pd.DataFrame:
    rows = []
    for method, filename in RUNTIME_FILES.items():
        runtime_s = parse_elapsed_seconds(base / filename)
        rows.append({"method": method, "runtime_s": runtime_s})
    runtime = pd.DataFrame(rows)
    fastani_runtime = float(runtime.loc[runtime["method"].eq("FastANI"), "runtime_s"].iloc[0])
    runtime["speedup_vs_fastani"] = fastani_runtime / runtime["runtime_s"]
    return runtime


def load_truth(base: Path) -> pd.DataFrame:
    joined = pd.read_csv(base / "methano_280_orthoani_vs_skani_joined.tsv", sep="\t")
    return joined[["genome1", "genome2", "orthoani"]].drop_duplicates().copy()


def load_method_values(base: Path) -> pd.DataFrame:
    main = pd.read_csv(base / "methano_280_orthoani_vs_fastani_turboani_long.tsv", sep="\t")
    skani = pd.read_csv(base / "methano_280_orthoani_vs_skani_long.tsv", sep="\t")
    values = pd.concat([main, skani], ignore_index=True)
    return values[["genome1", "genome2", "method", "method_ani_avg"]].copy()


def summarize(base: Path, tolerances: list[float]) -> pd.DataFrame:
    truth = load_truth(base)
    values = load_method_values(base)
    runtime = load_runtime(base)
    rows = []
    for tolerance in tolerances:
        for group_name, low, high in GROUPS:
            group_truth = truth.copy()
            if low is not None:
                group_truth = group_truth[group_truth["orthoani"] >= low]
            if high is not None:
                group_truth = group_truth[group_truth["orthoani"] < high]
            truth_pairs = len(group_truth)

            for method in METHODS:
                method_values = values[values["method"].eq(method)]
                joined = group_truth.merge(
                    method_values,
                    how="left",
                    on=["genome1", "genome2"],
                )
                computed = joined["method_ani_avg"].notna()
                delta = joined.loc[computed, "method_ani_avg"] - joined.loc[computed, "orthoani"]
                within = int((delta.abs() <= tolerance).sum())
                runtime_row = runtime[runtime["method"].eq(method)].iloc[0]
                rows.append(
                    {
                        "tolerance": tolerance,
                        "orthoani_group": group_name,
                        "method": method,
                        "truth_pairs": truth_pairs,
                        "computed_pairs": int(computed.sum()),
                        "computed_fraction": float(computed.sum() / truth_pairs) if truth_pairs else np.nan,
                        "within_pairs": within,
                        "ani_recovery": float(within / truth_pairs) if truth_pairs else np.nan,
                        "mae_computed": float(delta.abs().mean()) if len(delta) else np.nan,
                        "rmse_computed": float(np.sqrt((delta * delta).mean())) if len(delta) else np.nan,
                        "runtime_s": float(runtime_row.runtime_s),
                        "speedup_vs_fastani": float(runtime_row.speedup_vs_fastani),
                    }
                )
    return pd.DataFrame(rows)


def plot_tradeoff(summary: pd.DataFrame, tolerance: float, out_prefix: Path) -> None:
    configure_matplotlib()
    sub = summary[summary["tolerance"].eq(tolerance)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.0), sharey=True)

    max_x = max(5.0, float(sub["ani_recovery"].max() * 100.0) * 1.18)
    max_y = float(sub["speedup_vs_fastani"].max())
    y_ticks = [1, 3, 10, 30]

    for ax, (group_name, _low, _high) in zip(axes, GROUPS):
        panel = sub[sub["orthoani_group"].eq(group_name)].set_index("method").loc[METHODS]
        truth_pairs = int(panel["truth_pairs"].iloc[0])
        for method, row in panel.iterrows():
            x = float(row["ani_recovery"]) * 100.0
            y = float(row["speedup_vs_fastani"])
            ax.scatter(
                x,
                y,
                s=130,
                marker=MARKERS[method],
                color=COLORS[method],
                edgecolor="black",
                linewidth=0.9,
                zorder=3,
            )
            ax.annotate(
                LABELS[method],
                xy=(x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10.5,
                color="black",
            )

        ax.set_title(f"{group_name}\nn={truth_pairs:,}")
        ax.set_xlim(0, max_x)
        ax.set_yscale("log")
        ax.set_ylim(0.75, max(32.0, max_y * 1.25))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{tick:g}x" for tick in y_ticks])
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_xlabel(f"ANI recovery within +/-{tolerance:g}% (%)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Speedup vs fastANI C++")
    handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[method],
            color="none",
            markerfacecolor=COLORS[method],
            markeredgecolor="black",
            markeredgewidth=0.9,
            markersize=9,
            label=LABELS[method],
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
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--tolerances", type=float, nargs="+", default=[1.0, 2.0])
    args = parser.parse_args()
    summary = summarize(args.base, args.tolerances)
    summary.to_csv(
        args.base / "methano_280_speed_accuracy_summary.tsv",
        sep="\t",
        index=False,
        float_format="%.6f",
    )
    for tolerance in args.tolerances:
        suffix = str(tolerance).replace(".", "p")
        plot_tradeoff(
            summary,
            tolerance,
            args.base / f"methano_280_speed_accuracy_tol{suffix}",
        )


if __name__ == "__main__":
    main()
