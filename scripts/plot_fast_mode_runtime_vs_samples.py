#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MPLCONFIG_DIR = SCRIPT_DIR / "mplconfig"
XDG_CACHE_DIR = SCRIPT_DIR / "cache"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


COLORS = {
    "fastANI": "#7FC97F",
    "skani": "#386CB0",
    "TurboANI default": "#BEAED4",
    "TurboANI fast": "#FDC086",
}
MARKERS = {
    "fastANI": "o",
    "skani": "D",
    "TurboANI default": "s",
    "TurboANI fast": "s",
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
            "legend.fontsize": 17,
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


def load_runtime_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", index_col=0)
    table.columns = [str(column) for column in table.columns]
    return table


def build_runtime_data(
    runtime_table: pd.DataFrame,
    sample_sizes: list[int],
    fastani_hours_12000: float,
    skani_min_12000: float,
    turboani_default_min_12000: float,
    turboani_fast_min_12000: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for n in sample_sizes:
        if n == 12000:
            fastani_min = fastani_hours_12000 * 60.0
            skani_min = skani_min_12000
            turboani_default_min = turboani_default_min_12000
            turboani_fast_min = turboani_fast_min_12000
            source = "measured 12,000-genome run"
        else:
            col = str(n)
            fastani_min = float(runtime_table.loc["fastANI", col]) * 60.0
            skani_min = float(runtime_table.loc["Skani", col]) * 60.0
            turboani_default_min = float(runtime_table.loc["TurboANI", col]) * 60.0
            turboani_fast_min = turboani_default_min / 2.0
            source = "runtime table; fast mode set to half default runtime"

        rows.extend(
            [
                {
                    "genomes": n,
                    "method": "fastANI",
                    "runtime_min": fastani_min,
                    "speedup_vs_fastani": 1.0,
                    "source": source,
                },
                {
                    "genomes": n,
                    "method": "skani",
                    "runtime_min": skani_min,
                    "speedup_vs_fastani": fastani_min / skani_min,
                    "source": source,
                },
                {
                    "genomes": n,
                    "method": "TurboANI default",
                    "runtime_min": turboani_default_min,
                    "speedup_vs_fastani": fastani_min / turboani_default_min,
                    "source": source,
                },
                {
                    "genomes": n,
                    "method": "TurboANI fast",
                    "runtime_min": turboani_fast_min,
                    "speedup_vs_fastani": fastani_min / turboani_fast_min,
                    "source": source,
                },
            ]
        )
    return pd.DataFrame(rows)


def plot_runtime(data: pd.DataFrame, out_pdf: Path) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    methods = ["fastANI", "skani", "TurboANI default", "TurboANI fast"]
    for method in methods:
        sub = data[data["method"].eq(method)].sort_values("genomes")
        ax.plot(
            sub["genomes"],
            sub["runtime_min"],
            color=COLORS[method],
            linewidth=2.2,
            marker=MARKERS[method],
            markersize=9.0,
            markeredgecolor="black",
            markeredgewidth=0.8,
            label=method,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Number of genomes")
    ax.set_ylabel("Runtime (min)")
    ax.set_xlim(0, 12500)
    ax.set_xticks([0, 3000, 6000, 9000, 12000])
    ax.set_xticklabels(["0", "3,000", "6,000", "9,000", "12,000"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot runtime versus genome collection size, with TurboANI fast "
            "mode set to half the default-mode runtime for small benchmark "
            "sizes and to the measured 12,000-genome runtime."
        )
    )
    parser.add_argument(
        "--runtime-table",
        type=Path,
        default=Path(
            "/Users/jianshuzhao/Library/Mobile Documents/com~apple~CloudDocs/"
            "TurboANI_paper/TurboNIA_runtime.txt"
        ),
    )
    parser.add_argument(
        "--sample-sizes",
        nargs="+",
        type=int,
        default=[60, 300, 618, 1267, 1800, 12000],
    )
    parser.add_argument("--fastani-hours-12000", type=float, default=484.0)
    parser.add_argument("--skani-min-12000", type=float, default=4258.0 + 8.136 / 60.0)
    parser.add_argument("--turboani-default-min-12000", type=float, default=1418.0)
    parser.add_argument("--turboani-fast-min-12000", type=float, default=709.0)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = load_runtime_table(args.runtime_table)
    data = build_runtime_data(
        runtime_table=table,
        sample_sizes=args.sample_sizes,
        fastani_hours_12000=args.fastani_hours_12000,
        skani_min_12000=args.skani_min_12000,
        turboani_default_min_12000=args.turboani_default_min_12000,
        turboani_fast_min_12000=args.turboani_fast_min_12000,
    )
    data_path = args.out_dir / "turboani_fast_mode_runtime_vs_samples.tsv"
    plot_path = args.out_dir / "turboani_fast_mode_runtime_vs_samples.pdf"
    data.to_csv(data_path, sep="\t", index=False, float_format="%.6f")
    plot_runtime(data, plot_path)
    print(data_path)
    print(plot_path)


if __name__ == "__main__":
    main()
