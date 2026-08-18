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
    name = Path(str(value)).name
    if name.endswith(".gz"):
        name = name[:-3]
    return name


def pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def configure_matplotlib() -> None:
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
            "grid.color": "0.7",
            "grid.linestyle": "--",
            "grid.linewidth": 0.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_orthoani(path: Path) -> tuple[list[str], dict[tuple[str, str], float]]:
    raw = pd.read_csv(path, sep="\t")
    raw["genome1"] = raw["genome1"].map(genome_key)
    raw["genome2"] = raw["genome2"].map(genome_key)

    genomes: set[str] = set()
    ani_by_pair: dict[tuple[str, str], float] = {}
    for row in raw.itertuples(index=False):
        genome1 = row.genome1
        genome2 = row.genome2
        genomes.add(genome1)
        genomes.add(genome2)
        ani_by_pair[pair_key(genome1, genome2)] = float(row.orthoANI_value)

    for genome in genomes:
        ani_by_pair[(genome, genome)] = 100.0
    return sorted(genomes), ani_by_pair


def read_directed_ani(path: Path, genomes: list[str], ani_scale: float) -> dict[tuple[str, str], float]:
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
    raw = raw[raw["ani"] >= 0].copy()
    raw["ani"] = raw["ani"] * ani_scale
    raw[["genome1", "genome2"]] = pd.DataFrame(
        [pair_key(query, reference) for query, reference in zip(raw["query"], raw["reference"])],
        index=raw.index,
    )
    averaged = (
        raw.groupby(["genome1", "genome2"], as_index=False)
        .agg(ani=("ani", "mean"), direction_count=("ani", "size"))
        .sort_values(["genome1", "genome2"])
    )
    ani_by_pair = {(row.genome1, row.genome2): float(row.ani) for row in averaged.itertuples(index=False)}
    for genome in genomes:
        ani_by_pair[(genome, genome)] = 100.0
    return ani_by_pair


def top_k_neighbors(
    query: str,
    database: list[str],
    ani_by_pair: dict[tuple[str, str], float],
    k: int,
) -> list[str]:
    ranked = []
    for genome in database:
        value = ani_by_pair.get(pair_key(query, genome), float("-inf"))
        ranked.append((value, genome))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [genome for value, genome in ranked[:k] if np.isfinite(value)]


def evaluate_splits(
    genomes: list[str],
    truth_ani: dict[tuple[str, str], float],
    method_ani_maps: dict[str, dict[tuple[str, str], float]],
    n_repeats: int,
    query_fraction: float,
    top_ks: list[int],
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_queries = max(1, int(round(len(genomes) * query_fraction)))
    rows = []
    for repeat in range(1, n_repeats + 1):
        query_genomes = sorted(rng.choice(genomes, size=n_queries, replace=False).tolist())
        query_set = set(query_genomes)
        database = sorted(genome for genome in genomes if genome not in query_set)

        for query in query_genomes:
            truth_top = {k: set(top_k_neighbors(query, database, truth_ani, k)) for k in top_ks}
            truth_best_ani = max(truth_ani.get(pair_key(query, genome), float("-inf")) for genome in database)
            for method, ani_by_pair in method_ani_maps.items():
                for k in top_ks:
                    method_top = set(top_k_neighbors(query, database, ani_by_pair, k))
                    recovered = len(method_top & truth_top[k])
                    rows.append(
                        {
                            "repeat": repeat,
                            "query": query,
                            "method": method,
                            "k": k,
                            "n_query_genomes": n_queries,
                            "n_database_genomes": len(database),
                            "truth_top_k_size": len(truth_top[k]),
                            "method_top_k_size": len(method_top),
                            "recovered": recovered,
                            "recall": recovered / k,
                            "truth_best_database_ani": truth_best_ani,
                        }
                    )
    return pd.DataFrame(rows)


def summarize(details: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_repeat = (
        details.groupby(["repeat", "method", "k"], as_index=False)
        .agg(
            mean_recall=("recall", "mean"),
            n_queries=("query", "nunique"),
            mean_truth_best_database_ani=("truth_best_database_ani", "mean"),
        )
        .sort_values(["k", "method", "repeat"])
    )
    summary = (
        per_repeat.groupby(["method", "k"], as_index=False)
        .agg(
            mean_recall=("mean_recall", "mean"),
            sd_recall=("mean_recall", "std"),
            min_recall=("mean_recall", "min"),
            max_recall=("mean_recall", "max"),
            n_repeats=("repeat", "nunique"),
            n_queries_per_repeat=("n_queries", "first"),
            mean_truth_best_database_ani=("mean_truth_best_database_ani", "mean"),
        )
        .sort_values(["k", "method"])
    )
    summary["se_recall"] = summary["sd_recall"] / np.sqrt(summary["n_repeats"])
    return per_repeat, summary


def plot_summary(summary: pd.DataFrame, per_repeat: pd.DataFrame, out_path: Path) -> None:
    configure_matplotlib()
    colors = {
        "FastANI": "#7FC97F",
        "TurboANI sensitive 77.5": "#E41A1C",
        "TurboANI default 80": "#BEAED4",
        "TurboANI fast 82.5": "#FDC086",
        "skani": "#386CB0",
    }
    methods = list(colors)
    ks = sorted(summary["k"].unique())
    x = np.arange(len(methods))

    markers = {5: "o", 10: "s"}
    offsets = {5: -0.12, 10: 0.12}

    fig, ax = plt.subplots(figsize=(6.1, 7.8 * 2.0 / 3.0))
    for k in ks:
        sub = summary[summary["k"] == k].set_index("method").loc[methods].reset_index()
        means = sub["mean_recall"].to_numpy() * 100.0
        ses = sub["se_recall"].fillna(0.0).to_numpy() * 100.0
        xpos = x + offsets.get(k, 0.0)
        for xi, method, mean, se in zip(xpos, methods, means, ses):
            ax.errorbar(
                xi,
                mean,
                yerr=se,
                fmt=markers.get(k, "o"),
                markersize=8.0,
                markerfacecolor=colors[method],
                markeredgecolor="black",
                markeredgewidth=0.8,
                ecolor="black",
                elinewidth=0.9,
                capsize=3,
                linestyle="none",
                zorder=4,
            )
            repeat_values = (
                per_repeat[(per_repeat["k"] == k) & (per_repeat["method"] == method)]
                .sort_values("repeat")["mean_recall"]
                .to_numpy()
                * 100.0
            )
            jitter = np.linspace(-0.035, 0.035, len(repeat_values)) if len(repeat_values) else []
            ax.scatter(
                xi + jitter,
                repeat_values,
                s=10,
                color="black",
                alpha=0.32,
                linewidths=0,
                zorder=3,
            )

    ax.set_ylabel("Nearest-neighbor recall (%)")
    ax.set_xlabel("Method")
    ax.set_ylim(70, 92)
    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(methods))

    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[method],
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=7.5,
            label=method,
        )
        for method in methods
    ]
    shape_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[k],
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            linestyle="none",
            markersize=7.5,
            label=f"Recall@{k}",
        )
        for k in ks
    ]
    fig.legend(
        handles=method_handles + shape_handles,
        loc="center left",
        bbox_to_anchor=(0.80, 0.50),
        ncol=1,
        frameon=False,
        handlelength=1.0,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(right=0.78)

    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
    ax.tick_params(direction="out", length=4, width=0.8)
    fig.savefig(out_path, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nearest-neighbor recall against OrthoANIu top-k neighbors."
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the 300-genome accuracy result files.",
    )
    parser.add_argument("--repeats", type=int, default=10, help="Number of random query/database splits.")
    parser.add_argument("--query-fraction", type=float, default=0.10, help="Fraction of genomes sampled as queries.")
    parser.add_argument("--top-k", default="5,10", help="Comma-separated top-k recall values.")
    parser.add_argument("--seed", type=int, default=20260818, help="Random seed for reproducible splits.")
    args = parser.parse_args()

    base = args.base
    top_ks = [int(value) for value in args.top_k.split(",") if value.strip()]
    genomes, truth_ani = load_orthoani(base / "orthoani_all_44850_pairs.results.tsv")
    method_inputs = [
        ("FastANI", base / "strep_300_fastANI_local_t16.txt", 1.0),
        ("TurboANI sensitive 77.5", base / "strep_300_turboani_min77_5.txt", 1.0),
        ("TurboANI default 80", base / "strep_300_turboani_min80.txt", 1.0),
        ("TurboANI fast 82.5", base / "strep_300_turboani_min82_5.txt", 1.0),
        ("skani", base / "strep_300_superani.txt", 100.0),
    ]
    method_ani_maps = {
        method: read_directed_ani(path, genomes, ani_scale) for method, path, ani_scale in method_inputs
    }

    details = evaluate_splits(
        genomes,
        truth_ani,
        method_ani_maps,
        n_repeats=args.repeats,
        query_fraction=args.query_fraction,
        top_ks=top_ks,
        seed=args.seed,
    )
    per_repeat, summary = summarize(details)

    out_prefix = base / "ani_nearest_neighbor_recall_300"
    details.to_csv(out_prefix.with_name(out_prefix.name + "_details.tsv"), sep="\t", index=False, float_format="%.6f")
    per_repeat.to_csv(
        out_prefix.with_name(out_prefix.name + "_per_repeat.tsv"),
        sep="\t",
        index=False,
        float_format="%.6f",
    )
    summary.to_csv(
        out_prefix.with_name(out_prefix.name + "_summary.tsv"),
        sep="\t",
        index=False,
        float_format="%.6f",
    )
    plot_summary(summary, per_repeat, out_prefix.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
