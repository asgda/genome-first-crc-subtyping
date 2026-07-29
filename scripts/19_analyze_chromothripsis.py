#!/usr/bin/env python3
"""Summarize ShatterSeek chromothripsis calls across locked CRC subtypes.

Candidate and prespecified high-confidence calls are compared using Fisher
exact tests with multiplicity correction. The script reports a negative C4
enrichment result without relying on manual visual classification.
"""

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

##############################################################################
# CONFIG
##############################################################################
BASE = os.environ.get("CRC_BASE", str(Path(__file__).resolve().parents[1]))
OUTDIR = os.environ.get("CRC_M10_OUT", f"{BASE}/module10_chromothripsis_C1C4_ppt")
RAWDIR = os.path.join(OUTDIR, "raw")
TABDIR = os.path.join(OUTDIR, "tables")
FIGDIR = os.path.join(OUTDIR, "figures")
for d in (TABDIR, FIGDIR):
    os.makedirs(d, exist_ok=True)

CHROMSUMMARY = os.environ.get("CRC_M10_CHROMSUMMARY",
                               os.path.join(RAWDIR, "shatterseek_chromsummary_raw.csv"))
CLUSTER_FILE = os.environ.get("CRC_CLUSTER_FILE",
                               f"{BASE}/module05_06_loocv_results/labels/NMF_k4_LOOCV.csv")
GENES_BED = os.environ.get(
    "CRC_REF_GENES_BED",
    str(Path(BASE).parent / "reference" / "genes_protein_coding_hg38.bed"),
)

ALPHA = float(os.environ.get("CRC_M10_ALPHA", "0.05"))
C4_INTERNAL = 3

CLUSTER_COLORS = {"C1": "#E69F00", "C2": "#56B4E9", "C3": "#009E73", "C4": "#D55E00"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15, "axes.titlesize": 19,
    "axes.labelsize": 16, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 13, "savefig.dpi": 300, "axes.linewidth": 1.0,
    "figure.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
})

##############################################################################
# HELPERS
##############################################################################
def normalize_sample(x):
    m = re.search(r"(UM\d+|U\d+)", str(x))
    return m.group(1) if m else None

def display_cluster(c):
    return f"C{int(c) + 1}"

def normalize_cluster_values(series):
    s = pd.to_numeric(series, errors="raise").astype(int)
    vals = sorted(s.dropna().unique().tolist())
    if vals and min(vals) == 1 and max(vals) <= 8 and 0 not in vals:
        s = s - 1
    return s

def load_labels():
    lab = pd.read_csv(CLUSTER_FILE)
    lab["sid"] = lab["sample_id"].astype(str).map(normalize_sample)
    lab["cluster"] = normalize_cluster_values(lab["cluster"])
    lab = lab.dropna(subset=["sid"]).drop_duplicates("sid")
    return lab.set_index("sid")["cluster"]

def savefig(base):
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(f"{base}.{ext}", bbox_inches="tight")
    plt.close()
    print(f"  -> {base}.png/pdf")

##############################################################################
# LOAD + CLASSIFY
##############################################################################
REQUIRED_COLS = [
    "chrom", "sample_id", "number_DEL", "number_DUP", "number_h2hINV",
    "number_t2tINV", "number_TRA", "clusterSize_including_TRA",
    "pval_fragment_joins", "chr_breakpoint_enrichment", "pval_exp_chr",
    "max_number_oscillating_CN_segments_2_states",
]

def load_and_classify():
    if not os.path.exists(CHROMSUMMARY):
        raise FileNotFoundError(
            f"{CHROMSUMMARY} not found -- run 10a then 10b first.")
    df = pd.read_csv(CHROMSUMMARY)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Expected column(s) {missing} not found in {CHROMSUMMARY}. "
            f"Columns actually present: {list(df.columns)}. "
            f"This means the installed ShatterSeek version's chromSummary "
            f"layout differs from the one this script was written against "
            f"(github.com/parklab/ShatterSeek, commit 3b3f4cd, 2026-07-10) "
            f"-- fix the column names above rather than guessing.")

    df["intra_count"] = (df["number_DEL"] + df["number_DUP"]
                          + df["number_h2hINV"] + df["number_t2tINV"])
    osc2 = df["max_number_oscillating_CN_segments_2_states"]
    fj_sig = df["pval_fragment_joins"] < ALPHA
    enrich_or_exp_sig = (df["chr_breakpoint_enrichment"] < ALPHA) | (df["pval_exp_chr"] < ALPHA)

    high_A = (df["intra_count"] >= 6) & (osc2 >= 7) & fj_sig & enrich_or_exp_sig
    high_B = (df["intra_count"] >= 3) & (df["number_TRA"] >= 4) & (osc2 >= 7) & fj_sig
    df["high_confidence"] = (high_A | high_B).fillna(False)

    low_conf = ((df["intra_count"] >= 6) & (osc2 >= 4) & (osc2 <= 6)
                & fj_sig & enrich_or_exp_sig).fillna(False)
    df["low_confidence"] = low_conf & ~df["high_confidence"]
    df["candidate"] = df["high_confidence"] | df["low_confidence"]

    df["sample_id"] = df["sample_id"].astype(str).map(normalize_sample)
    return df

##############################################################################
# PER-SAMPLE ROLLUP
##############################################################################
def rollup_per_sample(chrom_df, labels):
    g = chrom_df.groupby("sample_id")
    roll = pd.DataFrame({
        "n_chroms_high_confidence": g["high_confidence"].sum(),
        "n_chroms_candidate": g["candidate"].sum(),
    })
    roll["high_confidence_positive"] = roll["n_chroms_high_confidence"] > 0
    roll["candidate_positive"] = roll["n_chroms_candidate"] > 0
    roll["cluster"] = labels.reindex(roll.index)
    roll = roll.dropna(subset=["cluster"])
    roll["cluster"] = roll["cluster"].astype(int)
    roll["cluster_display"] = roll["cluster"].map(display_cluster)
    return roll

##############################################################################
# STATISTICAL TESTS -- C4 vs rest, C4 vs C1
##############################################################################
def run_tests(roll):
    rows = []
    is_c4 = roll["cluster"] == C4_INTERNAL
    comparisons = {
        "C4_vs_rest": (is_c4, ~is_c4),
        "C4_vs_C1": (is_c4, roll["cluster"] == 0),
    }
    for tier, pos_col, burden_col in [
        ("candidate", "candidate_positive", "n_chroms_candidate"),
        ("high_confidence", "high_confidence_positive", "n_chroms_high_confidence"),
    ]:
        for comp_name, (mask_a, mask_b) in comparisons.items():
            a, b = roll.loc[mask_a], roll.loc[mask_b]
            table = [[a[pos_col].sum(), (~a[pos_col]).sum()],
                     [b[pos_col].sum(), (~b[pos_col]).sum()]]
            odds, p_fisher = fisher_exact(table)
            try:
                _, p_mw = mannwhitneyu(a[burden_col], b[burden_col], alternative="two-sided")
            except ValueError:
                p_mw = np.nan
            rows.append({
                "tier": tier, "comparison": comp_name,
                "n_group1": len(a), "n_group2": len(b),
                "prevalence_group1_pct": 100 * a[pos_col].mean(),
                "prevalence_group2_pct": 100 * b[pos_col].mean(),
                "odds_ratio": odds, "fisher_p": p_fisher,
                "burden_median_group1": a[burden_col].median(),
                "burden_median_group2": b[burden_col].median(),
                "mannwhitney_p": p_mw,
            })
    res = pd.DataFrame(rows)
    pvals = pd.concat([res["fisher_p"], res["mannwhitney_p"]]).values
    fdr = multipletests(pvals, method="fdr_bh")[1]
    res["fisher_FDR"] = fdr[:len(res)]
    res["mannwhitney_FDR"] = fdr[len(res):]
    return res

##############################################################################
# GENE ANNOTATION OF HIGH-CONFIDENCE REGIONS (interpretive, not detection)
##############################################################################
def annotate_genes(chrom_df):
    hi = chrom_df.loc[chrom_df["high_confidence"] & chrom_df["start"].notna()
                       & chrom_df["end"].notna()].copy()
    if hi.empty or not os.path.exists(GENES_BED):
        if not os.path.exists(GENES_BED):
            print(f"  [warn] gene BED not found at {GENES_BED} -- skipping gene annotation")
        return pd.DataFrame()

    genes = pd.read_csv(GENES_BED, sep="\t", header=None,
                         names=["chrom", "start", "end", "gene", "ensembl", "strand", "tss"],
                         usecols=[0, 1, 2, 3])
    genes["chrom"] = genes["chrom"].astype(str).str.replace("^chr", "", regex=True)

    rows = []
    for _, r in hi.iterrows():
        chrom = str(r["chrom"])
        overlap = genes.loc[(genes["chrom"] == chrom)
                             & (genes["start"] < r["end"]) & (genes["end"] > r["start"])]
        for g in overlap["gene"]:
            rows.append({"sample_id": r["sample_id"], "chrom": chrom,
                          "region_start": r["start"], "region_end": r["end"], "gene": g})
    return pd.DataFrame(rows)

##############################################################################
# FIGURE
##############################################################################
def make_figure(roll, tests):
    order = ["C1", "C2", "C3", "C4"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (tier, pos_col, burden_col, title) in zip(axes, [
        ("candidate", "candidate_positive", "n_chroms_candidate", "Candidate (pre-screen)"),
        ("high_confidence", "high_confidence_positive", "n_chroms_high_confidence", "High confidence"),
    ]):
        prev = (roll.groupby("cluster_display")[pos_col].mean() * 100).reindex(order).fillna(0)
        colors = [CLUSTER_COLORS[c] for c in order]
        ax.bar(order, prev.values, color=colors, edgecolor="black", linewidth=0.8)
        for i, v in enumerate(prev.values):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold")
        ax.set_ylabel("Chromothripsis-positive samples (%)")
        ax.set_title(title, fontweight="bold")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.set_ylim(0, max(prev.values.max() * 1.3, 10))  # prev is NaN-free after fillna(0)

        sub = tests.loc[(tests.tier == tier) & (tests.comparison == "C4_vs_rest")]
        if len(sub):
            p = sub["fisher_p"].values[0]
            ax.text(0.5, 0.97, f"C4 vs rest: Fisher p={p:.3g}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=11,
                    bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.85))

    fig.suptitle("Chromothripsis prevalence by subtype (ShatterSeek)", fontsize=18, fontweight="bold")
    savefig(os.path.join(FIGDIR, "chromothripsis_prevalence_composite"))

##############################################################################
# MAIN
##############################################################################
def main():
    print("=" * 70)
    print("MODULE 10c -- CHROMOTHRIPSIS CLASSIFICATION + C1-C4 ANALYSIS")
    print("=" * 70)

    chrom_df = load_and_classify()
    print(f"Loaded {len(chrom_df)} chromosome-level rows "
          f"({chrom_df['sample_id'].nunique()} samples)")
    print(f"  high_confidence rows: {chrom_df['high_confidence'].sum()}")
    print(f"  low_confidence rows : {chrom_df['low_confidence'].sum()}")
    print(f"  candidate rows      : {chrom_df['candidate'].sum()}")

    # Persist the full per-chromosome CLASSIFIED table (raw chromSummary +
    # intra_count/high_confidence/low_confidence/candidate columns computed
    # above). This is distinct from shatterseek_calls_persample.csv (the
    # per-sample rollup below) -- 10d_module10_review_plots_C1C4_ppt.R needs
    # per-chromosome classification to know which sample x chromosome pairs
    # to plot, and must not re-parse the raw (unclassified) chromSummary.
    chrom_classified_path = os.path.join(TABDIR, "shatterseek_chrom_classified.csv")
    chrom_df.to_csv(chrom_classified_path, index=False)
    print(f"Saved: {chrom_classified_path}")

    labels = load_labels()
    roll = rollup_per_sample(chrom_df, labels)
    print(f"\nPer-sample rollup: {len(roll)} samples matched to cluster labels")
    sizes = roll["cluster_display"].value_counts().reindex(["C1", "C2", "C3", "C4"])
    print(f"Cluster sizes in this rollup: {sizes.to_dict()}")
    if roll.empty:
        raise RuntimeError("No samples matched between ShatterSeek output and cluster labels "
                            "-- check sample ID normalization / that 10b ran on the full cohort.")

    calls_path = os.path.join(TABDIR, "shatterseek_calls_persample.csv")
    roll.reset_index().rename(columns={"index": "sample_id"}).to_csv(calls_path, index=False)
    print(f"Saved: {calls_path}")

    prevalence = roll.groupby("cluster_display").agg(
        n=("candidate_positive", "size"),
        candidate_positive_n=("candidate_positive", "sum"),
        candidate_positive_pct=("candidate_positive", lambda s: 100 * s.mean()),
        candidate_region_count_median=("n_chroms_candidate", "median"),
        high_confidence_positive_n=("high_confidence_positive", "sum"),
        high_confidence_positive_pct=("high_confidence_positive", lambda s: 100 * s.mean()),
        high_confidence_region_count_median=("n_chroms_high_confidence", "median"),
    ).reindex(["C1", "C2", "C3", "C4"])
    prev_path = os.path.join(TABDIR, "shatterseek_cluster_prevalence.csv")
    prevalence.to_csv(prev_path)
    print(f"\nCluster-level prevalence:\n{prevalence.to_string()}")
    print(f"Saved: {prev_path}")

    tests = run_tests(roll)
    tests_path = os.path.join(TABDIR, "shatterseek_cluster_tests.csv")
    tests.to_csv(tests_path, index=False)
    print(f"\nStatistical tests (C4 vs rest, C4 vs C1):\n{tests.to_string(index=False)}")
    print(f"Saved: {tests_path}")

    genes = annotate_genes(chrom_df)
    if len(genes):
        genes_path = os.path.join(TABDIR, "shatterseek_highconfidence_region_genes.csv")
        genes.to_csv(genes_path, index=False)
        print(f"\nGenes overlapping high-confidence regions: {len(genes)} rows -> {genes_path}")
        top_genes = genes["gene"].value_counts().head(15)
        print(f"Most recurrent genes in high-confidence regions:\n{top_genes.to_string()}")

    make_figure(roll, tests)

    caution_path = os.path.join(TABDIR, "WORDING_CAUTION.txt")
    with open(caution_path, "w") as f:
        f.write(
            "WORDING DISCIPLINE (carried forward from the original, deleted\n"
            "ShatterSeek pre-screen analysis -- see CRC_PROJECT_CONTEXT.md 9g):\n\n"
            "- 'candidate' / 'pre-screen' chromothripsis = low_confidence OR\n"
            "  high_confidence tier per this script. Do NOT call this\n"
            "  'chromothripsis' in the manuscript without qualification.\n\n"
            "- 'high-confidence' chromothripsis = the published ShatterSeek\n"
            "  cut-offs (tutorial.tex) are met. This is still NOT the same as\n"
            "  manually confirmed chromothripsis -- ShatterSeek's own authors\n"
            "  state that ~2,600 WGS samples analysed this way still needed\n"
            "  visual curation to remove false positives. Use\n"
            "  10d_module10_review_plots_C1C4_ppt.R to generate per-region\n"
            "  plots for manual review of C4's high-confidence calls before\n"
            "  writing 'confirmed chromothripsis' anywhere in the manuscript.\n"
        )
    print(f"\nSaved wording-discipline note: {caution_path}")
    print("\nMODULE 10c COMPLETE.")

if __name__ == "__main__":
    main()
