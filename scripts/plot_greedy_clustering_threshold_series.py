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


def load_orthoani(path: Path) -> tuple[list[str], dict[str, int], dict[tuple[str, str], float]]:
    raw = pd.read_csv(path, sep="\t")
    raw["genome1_key"] = raw["genome1"].map(genome_key)
    raw["genome2_key"] = raw["genome2"].map(genome_key)

    lengths: dict[str, int] = {}
    ani_by_pair: dict[tuple[str, str], float] = {}
    for row in raw.itertuples(index=False):
        g1 = row.genome1_key
        g2 = row.genome2_key
        lengths[g1] = max(lengths.get(g1, 0), int(row.query_length))
        lengths[g2] = max(lengths.get(g2, 0), int(row.subject_length))
        ani_by_pair[pair_key(g1, g2)] = float(row.orthoANI_value)

    for genome in lengths:
        ani_by_pair[(genome, genome)] = 100.0

    return sorted(lengths), lengths, ani_by_pair


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
        [pair_key(q, r) for q, r in zip(raw["query"], raw["reference"])],
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


def greedy_cluster(
    genomes: list[str],
    lengths: dict[str, int],
    ani_by_pair: dict[tuple[str, str], float],
    threshold: float,
) -> tuple[list[str], dict[str, set[str]], dict[str, str]]:
    order = sorted(genomes, key=lambda genome: (-lengths[genome], genome))
    unassigned = set(order)
    representatives: list[str] = []
    clusters: dict[str, set[str]] = {}
    assignment: dict[str, str] = {}

    for representative in order:
        if representative not in unassigned:
            continue
        members = []
        for genome in list(unassigned):
            if genome == representative or ani_by_pair.get(pair_key(representative, genome), -1.0) >= threshold:
                members.append(genome)
        for genome in members:
            unassigned.remove(genome)
            assignment[genome] = representative
        representatives.append(representative)
        clusters[representative] = set(members)

    return representatives, clusters, assignment


def pairwise_cluster_stats(
    genomes: list[str],
    truth_assignment: dict[str, str],
    method_assignment: dict[str, str],
) -> tuple[int, int, int, float, float, float]:
    tp = 0
    fp = 0
    fn = 0
    for i, genome1 in enumerate(genomes):
        for genome2 in genomes[i + 1 :]:
            truth_same = truth_assignment[genome1] == truth_assignment[genome2]
            method_same = method_assignment[genome1] == method_assignment[genome2]
            if truth_same and method_same:
                tp += 1
            elif not truth_same and method_same:
                fp += 1
            elif truth_same and not method_same:
                fn += 1

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return tp, fp, fn, precision, recall, f1


def summarize_method(
    method: str,
    threshold: float,
    genomes: list[str],
    truth_reps: list[str],
    truth_clusters: dict[str, set[str]],
    truth_assignment: dict[str, str],
    truth_ani: dict[tuple[str, str], float],
    reps: list[str],
    clusters: dict[str, set[str]],
    assignment: dict[str, str],
    ani_by_pair: dict[tuple[str, str], float],
) -> dict[str, object]:
    truth_edges = sum(1 for (g1, g2), ani in truth_ani.items() if g1 != g2 and ani >= threshold)
    method_edges = sum(1 for (g1, g2), ani in ani_by_pair.items() if g1 != g2 and ani >= threshold)
    rep_overlap = len(set(reps) & set(truth_reps))
    same_rep_genomes = sum(assignment[genome] == truth_assignment[genome] for genome in genomes)
    exact_truth_clusters = sum(
        representative in clusters and clusters[representative] == truth_clusters[representative]
        for representative in truth_reps
    )
    tp, fp, fn, pair_precision, pair_recall, pair_f1 = pairwise_cluster_stats(
        genomes, truth_assignment, assignment
    )
    expected_pairs = len(genomes) * (len(genomes) + 1) // 2

    return {
        "method": method,
        "threshold_ani": threshold,
        "n_genomes": len(genomes),
        "n_clusters": len(reps),
        "truth_n_clusters": len(truth_reps),
        "cluster_delta_vs_truth": len(reps) - len(truth_reps),
        "n_non_singleton_clusters": sum(len(members) > 1 for members in clusters.values()),
        "truth_n_non_singleton_clusters": sum(len(members) > 1 for members in truth_clusters.values()),
        "largest_cluster_size": max(len(members) for members in clusters.values()),
        "truth_largest_cluster_size": max(len(members) for members in truth_clusters.values()),
        "n_edges_ge_threshold": method_edges,
        "truth_edges_ge_threshold": truth_edges,
        "edge_delta_vs_truth": method_edges - truth_edges,
        "averaged_pairs_with_self": len(ani_by_pair),
        "expected_averaged_pairs_with_self": expected_pairs,
        "missing_averaged_pairs_with_self": expected_pairs - len(ani_by_pair),
        "representative_overlap": rep_overlap,
        "representative_precision": rep_overlap / len(reps),
        "representative_recall": rep_overlap / len(truth_reps),
        "same_representative_genomes": same_rep_genomes,
        "same_representative_fraction": same_rep_genomes / len(genomes),
        "exact_truth_clusters_recovered": exact_truth_clusters,
        "exact_truth_cluster_recovery": exact_truth_clusters / len(truth_reps),
        "pair_tp": tp,
        "pair_fp": fp,
        "pair_fn": fn,
        "pair_precision": pair_precision,
        "pair_recall": pair_recall,
        "pair_f1": pair_f1,
    }


def collect_edge_disagreements(
    threshold: float,
    genomes: list[str],
    truth_ani: dict[tuple[str, str], float],
    method_ani_maps: dict[str, dict[tuple[str, str], float]],
) -> list[dict[str, object]]:
    rows = []
    for method, ani_by_pair in method_ani_maps.items():
        for i, genome1 in enumerate(genomes):
            for genome2 in genomes[i + 1 :]:
                key = pair_key(genome1, genome2)
                truth_value = truth_ani.get(key, np.nan)
                method_value = ani_by_pair.get(key, np.nan)
                truth_edge = bool(truth_value >= threshold)
                method_edge = bool(method_value >= threshold) if not np.isnan(method_value) else False
                if truth_edge != method_edge:
                    rows.append(
                        {
                            "threshold_ani": threshold,
                            "method": method,
                            "genome1": genome1,
                            "genome2": genome2,
                            "truth_ani": truth_value,
                            "method_ani": method_value,
                            "truth_ge_threshold": truth_edge,
                            "method_ge_threshold": method_edge,
                        }
                    )
    return rows


def plot_threshold_series(summary: pd.DataFrame, out_path: Path) -> None:
    configure_matplotlib()
    colors = {
        "FastANI": "#7FC97F",
        "TurboANI sensitive 77.5": "#E41A1C",
        "TurboANI default 80": "#BEAED4",
        "TurboANI fast 82.5": "#FDC086",
        "skani": "#386CB0",
    }
    markers = {
        "FastANI": "o",
        "TurboANI sensitive 77.5": "^",
        "TurboANI default 80": "s",
        "TurboANI fast 82.5": "v",
        "skani": "D",
    }
    methods = list(colors)
    thresholds = sorted(summary["threshold_ani"].unique())
    truth_counts = (
        summary[summary["method"] == methods[0]]
        .sort_values("threshold_ani")["truth_n_clusters"]
        .to_numpy()
    )

    fig = plt.figure(figsize=(12.2, 7.8 * 2.0 / 3.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.05], hspace=0.42)
    ax_count = fig.add_subplot(gs[0, 0])
    ax_rep = fig.add_subplot(gs[1, 0], sharex=ax_count)

    ax_count.plot(
        thresholds,
        truth_counts,
        color="black",
        linestyle="--",
        linewidth=1.25,
        marker="x",
        markersize=6.0,
        label="OrthoANIu truth",
    )
    for method in methods:
        sub = summary[summary["method"] == method].sort_values("threshold_ani")
        ax_count.plot(
            sub["threshold_ani"],
            sub["n_clusters"],
            color=colors[method],
            marker=markers[method],
            markersize=5.6,
            linewidth=1.9,
            label=method,
        )
        ax_rep.plot(
            sub["threshold_ani"],
            sub["same_representative_fraction"] * 100.0,
            color=colors[method],
            marker=markers[method],
            markersize=5.6,
            linewidth=1.9,
            label=method,
        )

    ax_count.set_ylabel("Clusters")
    ax_count.tick_params(axis="x", labelbottom=False)
    ax_count.text(
        0.01,
        0.94,
        "Greedy clustering with longest genome as representative",
        transform=ax_count.transAxes,
        ha="left",
        va="top",
        fontsize=11,
    )

    ax_rep.axhline(100.0, color="black", linestyle="--", linewidth=1.0)
    ax_rep.set_ylabel("Same representative (%)")
    ax_rep.set_xlabel("ANI threshold (%)")
    ax_rep.set_xticks(thresholds)
    ax_rep.set_xticklabels([f"{threshold:g}" for threshold in thresholds])
    ymin = max(0.0, min(99.0, float((summary["same_representative_fraction"] * 100.0).min()) - 1.2))
    ax_rep.set_ylim(ymin, 100.4)

    legend_handles = [
        Line2D([0], [0], color="black", linestyle="--", marker="x", linewidth=1.25, label="OrthoANIu truth")
    ]
    legend_handles.extend(
        Line2D(
            [0],
            [0],
            color=colors[method],
            marker=markers[method],
            linewidth=1.9,
            markersize=5.6,
            label=method,
        )
        for method in methods
    )
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.80, 0.50),
        ncol=1,
        frameon=False,
        handlelength=1.35,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(right=0.78)

    for ax in (ax_count, ax_rep):
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
        ax.tick_params(direction="out", length=4, width=0.8)

    fig.savefig(out_path, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Greedy longest-genome representative clustering across ANI thresholds."
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the 300-genome accuracy result files.",
    )
    parser.add_argument(
        "--thresholds",
        default="85,90,92.5,95,97.5,99.5",
        help="Comma-separated ANI thresholds.",
    )
    args = parser.parse_args()

    base = args.base
    thresholds = [float(value) for value in args.thresholds.split(",") if value.strip()]
    genomes, lengths, truth_ani = load_orthoani(base / "orthoani_all_44850_pairs.results.tsv")

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

    summary_rows = []
    edge_disagreement_rows = []
    for threshold in thresholds:
        truth_reps, truth_clusters, truth_assignment = greedy_cluster(genomes, lengths, truth_ani, threshold)
        for method, ani_by_pair in method_ani_maps.items():
            reps, clusters, assignment = greedy_cluster(genomes, lengths, ani_by_pair, threshold)
            summary_rows.append(
                summarize_method(
                    method,
                    threshold,
                    genomes,
                    truth_reps,
                    truth_clusters,
                    truth_assignment,
                    truth_ani,
                    reps,
                    clusters,
                    assignment,
                    ani_by_pair,
                )
            )
        edge_disagreement_rows.extend(
            collect_edge_disagreements(threshold, genomes, truth_ani, method_ani_maps)
        )

    summary = pd.DataFrame(summary_rows)
    edge_disagreements = pd.DataFrame(edge_disagreement_rows)
    out_prefix = base / "ani_greedy_clustering_threshold_series"
    summary.to_csv(out_prefix.with_name(out_prefix.name + "_summary.tsv"), sep="\t", index=False, float_format="%.6f")
    edge_disagreements.to_csv(
        out_prefix.with_name(out_prefix.name + "_edge_disagreements.tsv"),
        sep="\t",
        index=False,
        float_format="%.6f",
    )
    plot_threshold_series(summary, out_prefix.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
