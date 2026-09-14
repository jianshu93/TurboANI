#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def genome_key(value: object) -> str:
    return Path(str(value)).name


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


def read_skani(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        usecols=[0, 1, 2],
        names=["query", "reference", "ani_raw"],
        engine="python",
    )
    raw["query"] = raw["query"].map(genome_key)
    raw["reference"] = raw["reference"].map(genome_key)
    raw[["genome1", "genome2"]] = pd.DataFrame(
        [pair_key(q, r) for q, r in zip(raw["query"], raw["reference"])],
        index=raw.index,
    )
    raw["computed"] = raw["ani_raw"] >= 0
    raw["skani_ani"] = np.where(raw["computed"], raw["ani_raw"] * 100.0, np.nan)
    return raw


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
        "min_orthoani": float(np.min(xval)),
        "max_orthoani": float(np.max(xval)),
        "min_skani_ani": float(np.min(yval)),
        "max_skani_ani": float(np.max(yval)),
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot skani reciprocal-averaged ANI against OrthoANIu; skani -1 means skipped."
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the accuracy result files.",
    )
    args = parser.parse_args()
    base = args.base

    ortho = load_orthoani(base / "orthoani_all_44850_pairs.results.tsv")
    raw = read_skani(base / "strep_300_skani.txt")

    computed = raw[raw["computed"]].copy()
    averaged = (
        computed.groupby(["genome1", "genome2"], as_index=False)
        .agg(
            skani_avg=("skani_ani", "mean"),
            skani_computed_direction_count=("skani_ani", "size"),
        )
        .sort_values(["genome1", "genome2"])
    )
    seen = raw.groupby(["genome1", "genome2"], as_index=False).size().rename(
        columns={"size": "skani_seen_direction_count"}
    )
    skipped = (
        raw[~raw["computed"]]
        .groupby(["genome1", "genome2"], as_index=False)
        .size()
        .rename(columns={"size": "skani_skipped_direction_count"})
    )

    joined = ortho.merge(averaged, how="left", on=["genome1", "genome2"])
    joined = joined.merge(seen, how="left", on=["genome1", "genome2"])
    joined = joined.merge(skipped, how="left", on=["genome1", "genome2"])
    for col in [
        "skani_computed_direction_count",
        "skani_seen_direction_count",
        "skani_skipped_direction_count",
    ]:
        joined[col] = joined[col].fillna(0).astype(int)
    joined["skani_delta_vs_orthoani"] = joined["skani_avg"] - joined["orthoani"]
    joined["status"] = np.where(joined["skani_avg"].notna(), "computed", "skipped")

    plotted = joined[joined["status"] == "computed"].copy()
    stats = correlation_stats(plotted["orthoani"], plotted["skani_avg"])
    stats.update(
        {
            "orthoani_total_pairs_with_self": int(len(ortho)),
            "skani_directed_rows": int(len(raw)),
            "skani_computed_directed_rows": int(raw["computed"].sum()),
            "skani_skipped_directed_rows": int((~raw["computed"]).sum()),
            "skani_averaged_computed_pairs": int(len(averaged)),
            "skani_pairs_with_any_skipped_direction": int((joined["skani_skipped_direction_count"] > 0).sum()),
            "missing_or_skipped_orthoani_pairs": int((joined["status"] != "computed").sum()),
        }
    )
    summary = pd.DataFrame([{"method": "skani", **stats}])

    out_prefix = base / "orthoani_vs_skani"
    joined.to_csv(out_prefix.with_name(out_prefix.name + "_joined.tsv"), sep="\t", index=False, float_format="%.6f")
    summary.to_csv(out_prefix.with_name(out_prefix.name + "_summary.tsv"), sep="\t", index=False, float_format="%.6f")

    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(9.8, 8.8))
    ax.scatter(
        plotted["orthoani"],
        plotted["skani_avg"],
        s=13,
        marker="s",
        color="#386CB0",
        alpha=0.43,
        linewidths=0,
        rasterized=True,
    )
    ax.plot([75, 100.5], [75, 100.5], color="black", linewidth=1.0, linestyle="--")
    ax.set_xlim(75, 100.5)
    ax.set_ylim(75, 100.5)
    ax.set_xlabel("OrthoANIu (%)")
    ax.set_ylabel("skani ANI (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_aspect("equal", adjustable="box")

    label = f"skani  r={stats['pearson']:.4f}, MAE={stats['mae']:.3f}%, skipped={stats['missing_or_skipped_orthoani_pairs']:,} pairs"
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor="#386CB0",
                markeredgewidth=0,
                markersize=8,
                label=label,
            )
        ],
        loc="upper left",
        frameon=False,
        handletextpad=0.4,
    )

    fig.savefig(out_prefix.with_name(out_prefix.name + "_correlation.pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_name(out_prefix.name + "_correlation.svg"), bbox_inches="tight")


if __name__ == "__main__":
    main()
