#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.patches import FancyArrowPatch


BASE = Path("/Users/jianshuzhao/Documents/Codex/2026-07-23/for/outputs/turboani_fastani_evolution_example")
BLAST_PATH = BASE / "results/blast/pTi1078_vs_pTiC57_TDNA_frag100_blastn.tsv"
BEST_PATH = BASE / "results/blast/pTi1078_vs_pTiC57_TDNA_frag100_blastn.best_qcov75.tsv"
OUT_PDF = BASE / "results/pTi1078_vs_pTiC57_TDNA_blast_fragment_best_validation.pdf"
OUT_PNG = BASE / "results/pTi1078_vs_pTiC57_TDNA_blast_fragment_best_validation.png"

QCOV_MIN = 75.0

ACCENT = {
    "green": "#7FC97F",
    "purple": "#BEAED4",
    "orange": "#FDC086",
    "blue": "#386CB0",
    "red": "#F0027F",
    "gray": "#666666",
}


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


def draw_gene(ax, start, end, y, label, color, strand="+", height=0.12):
    x0 = min(start, end)
    x1 = max(start, end)
    if strand == "-":
        patch = FancyArrowPatch(
            (x1, y),
            (x0, y),
            arrowstyle=f"Simple,head_length=8,head_width={height * 120},tail_width={height * 60}",
            color=color,
            linewidth=0.6,
            edgecolor="black",
            alpha=0.95,
        )
    else:
        patch = FancyArrowPatch(
            (x0, y),
            (x1, y),
            arrowstyle=f"Simple,head_length=8,head_width={height * 120},tail_width={height * 60}",
            color=color,
            linewidth=0.6,
            edgecolor="black",
            alpha=0.95,
        )
    ax.add_patch(patch)
    ax.text((x0 + x1) / 2, y + 0.16, label, ha="center", va="bottom", fontsize=12)


def load_best_fragment_hits() -> pd.DataFrame:
    cols = [
        "qseqid",
        "sseqid",
        "pident",
        "length",
        "mismatch",
        "gapopen",
        "qstart",
        "qend",
        "sstart",
        "send",
        "evalue",
        "bitscore",
        "qlen",
        "slen",
        "qcovhsp",
    ]
    df = pd.read_csv(BLAST_PATH, sep="\t", names=cols)
    ids = df["qseqid"].str.extract(r"frag_(?P<frag>\d+)\|qstart=(?P<frag_start>\d+)\|qend=(?P<frag_end>\d+)")
    df["frag"] = ids["frag"].astype(int)
    df["frag_start"] = ids["frag_start"].astype(int)
    df["frag_end"] = ids["frag_end"].astype(int)
    best = (
        df.sort_values(["frag", "bitscore", "length", "pident"], ascending=[True, False, False, False])
        .drop_duplicates("frag")
        .sort_values("frag")
    )
    best = best[best["qcovhsp"] >= QCOV_MIN].copy()
    best.to_csv(BEST_PATH, sep="\t", index=False)
    return best


def main() -> None:
    configure_matplotlib()
    df = load_best_fragment_hits()

    q_len = 15025
    r_len = 12418
    x_max = max(q_len, r_len)

    q_genes = [
        ("acs", 1, 1245, "-", ACCENT["green"]),
        ("5", 1853, 2536, "+", "#FDC086"),
        ("iaaH", 3046, 4449, "-", ACCENT["blue"]),
        ("iaaM", 4789, 7056, "+", ACCENT["blue"]),
        ("ipt", 7759, 8478, "+", ACCENT["purple"]),
        ("6b", 8978, 9385, "-", ACCENT["purple"]),
        ("IS21", 9501, 11842, "-", "#BDBDBD"),
        ("nos", 13787, 15025, "-", ACCENT["red"]),
    ]
    r_genes = [
        ("acs", 1, 1257, "-", ACCENT["green"]),
        ("5", 1872, 2555, "+", "#FDC086"),
        ("iaaH", 3064, 4467, "-", ACCENT["blue"]),
        ("iaaM", 4807, 7074, "+", ACCENT["blue"]),
        ("ipt", 7777, 8496, "+", ACCENT["purple"]),
        ("6b", 8996, 9622, "-", ACCENT["purple"]),
        ("3'", 10060, 10755, "+", ACCENT["purple"]),
        ("nos", 11180, 12418, "-", ACCENT["red"]),
    ]
    q_break = 2099
    q_insert_start = 9395
    q_insert_end = 12014

    fig = plt.figure(figsize=(12.2, 7.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.05, 1.15], hspace=0.42)
    ax = fig.add_subplot(gs[0])
    ax_id = fig.add_subplot(gs[1])

    norm = mpl.colors.Normalize(vmin=78, vmax=100)
    cmap = mpl.colormaps["RdYlGn"]
    yq, yr = 1.32, 0.28

    polys = []
    colors = []
    for row in df.itertuples(index=False):
        q0 = int(row.frag_start) + int(row.qstart) - 1
        q1 = int(row.frag_start) + int(row.qend)
        r0, r1 = int(row.sstart) - 1, int(row.send)
        if q0 > q1:
            q0, q1 = q1, q0
        if r0 > r1:
            r0, r1 = r1, r0
        polys.append([(q0, yq - 0.08), (q1, yq - 0.08), (r1, yr + 0.08), (r0, yr + 0.08)])
        colors.append(cmap(norm(float(row.pident))))
    ribbons = PolyCollection(polys, facecolors=colors, edgecolors="none", alpha=0.64, zorder=1)
    ax.add_collection(ribbons)

    ax.hlines([yq, yr], 0, [q_len, r_len], colors="black", linewidth=1.4, zorder=3)
    for gene in q_genes:
        draw_gene(ax, gene[1], gene[2], yq, gene[0], gene[4], gene[3])
    for gene in r_genes:
        draw_gene(ax, gene[1], gene[2], yr, gene[0], gene[4], gene[3])

    ax.axvline(q_break, color="black", linestyle="--", linewidth=1.2, zorder=4)
    ax.axvspan(q_insert_start, q_insert_end, color="#BDBDBD", alpha=0.16, zorder=0)
    ax.text(q_break - 110, 1.86, "candidate breakpoint\nwithin gene 5", ha="right", va="top", fontsize=13)
    ax.text(-250, yq, "pTi1078\nquery", ha="right", va="center", fontsize=15)
    ax.text(-250, yr, "pTiC5.7\nreference", ha="right", va="center", fontsize=15)
    ax.text(
        (q_insert_start + q_insert_end) / 2,
        1.02,
        "missing best-fragment BLAST hits\nacross query-specific IS21 interval",
        ha="center",
        va="center",
        fontsize=12,
    )

    ax.set_xlim(-850, x_max + 300)
    ax.set_ylim(-0.08, 1.96)
    ax.set_yticks([])
    ax.set_title("BLAST Best-Hit Validation of the TurboANI Fragment Map", pad=12)
    ax.set_xlabel("Position in extracted T-DNA interval (kb)")
    ax.set_xticks(np.arange(0, x_max + 1, 2500))
    ax.set_xticklabels([f"{x / 1000:g}" for x in np.arange(0, x_max + 1, 2500)])
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.028, pad=0.015)
    cbar.set_label("BLAST identity (%)", fontsize=14)
    cbar.ax.tick_params(labelsize=13)

    mids = (df["frag_start"].to_numpy(float) + df["frag_end"].to_numpy(float)) / 2000
    identities = df["pident"].to_numpy(float)
    sizes = np.clip(df["qcovhsp"].to_numpy(float) * 0.34, 24, 38)
    ax_id.scatter(
        mids,
        identities,
        s=sizes,
        c=identities,
        cmap=cmap,
        norm=norm,
        edgecolor="black",
        linewidth=0.35,
        zorder=3,
    )
    ax_id.axvspan(q_genes[1][1] / 1000, q_genes[1][2] / 1000, color="#FDC086", alpha=0.18, zorder=0)
    ax_id.axvspan(q_insert_start / 1000, q_insert_end / 1000, color="#BDBDBD", alpha=0.18, zorder=0)
    ax_id.axvline(q_break / 1000, color="black", linestyle="--", linewidth=1.1, zorder=2)
    ax_id.axhline(95, color=ACCENT["gray"], linestyle=":", linewidth=1.2)
    ax_id.text(q_break / 1000 + 0.12, 79.1, "breakpoint", ha="left", va="bottom", fontsize=12)
    ax_id.text((q_insert_start + q_insert_end) / 2000, 79.1, "IS21 interval", ha="center", va="bottom", fontsize=12)
    ax_id.text((q_genes[1][1] + q_genes[1][2]) / 2000, 100.6, "gene 5", ha="center", va="bottom", fontsize=12)
    ax_id.set_xlim(0, q_len / 1000)
    ax_id.set_ylim(77.5, 101.6)
    ax_id.set_xlabel("pTi1078 query position (kb)")
    ax_id.set_ylabel("Best BLAST identity (%)")
    ax_id.set_xticks(np.arange(0, q_len / 1000 + 0.01, 2.5))
    for spine in ax_id.spines.values():
        spine.set_linewidth(1.1)

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"best hits plotted: {len(df)}")
    print(BEST_PATH)
    print(OUT_PDF)
    print(OUT_PNG)


if __name__ == "__main__":
    main()
