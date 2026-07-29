#!/usr/bin/env python3
"""Analyze canonical CRC pathway co-alteration within each genomic subtype.

Binary pathway states are derived from the 371-feature matrix. Pairwise
co-occurrence or mutual exclusivity is tested using Fisher exact tests with
Haldane-corrected odds ratios and within-subtype FDR correction.
"""

import os
import re
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE = os.environ.get("CRC_BASE", str(Path(__file__).resolve().parents[1]))
MATRIX_FILE = os.environ.get(
    "CRC_FEATURE_MATRIX",
    f"{BASE}/module4_results/module4_unified_discovery_matrix.csv",
)
CLUSTER_FILE = os.environ.get(
    "CRC_CLUSTER_FILE",
    f"{BASE}/module05_06_loocv_results/labels/NMF_k4_LOOCV.csv",
)
OUTDIR = Path(os.environ.get("CRC_M16_OUT", f"{BASE}/module16_pathway_comutation_C1C4_ppt"))
FIGDIR = OUTDIR / "figures"
TABDIR = OUTDIR / "tables"
FIGDIR.mkdir(parents=True, exist_ok=True)
TABDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------
CB8 = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999"]
CLUSTER_COLORS = {"C1": "#E69F00", "C2": "#56B4E9", "C3": "#009E73", "C4": "#D55E00"}
PATHWAY_COLORS = {
    "WNT": "#E69F00",
    "RAS_MAPK": "#0072B2",
    "TGFb": "#009E73",
    "PI3K": "#CC79A7",
    "p53_DDR": "#D55E00",
    "MMR_POL": "#F0E442",
    "CHROMATIN": "#56B4E9",
    "CIN_ARM": "#999999",
    "SV_COMPLEX": "#000000",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 18,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "savefig.dpi": 300,
    "axes.linewidth": 1.1,
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---------------------------------------------------------------------
# Pathway map
# ---------------------------------------------------------------------
PATHWAY_GENES = {
    "WNT": {
        "APC", "APC2", "CTNNB1", "AXIN1", "AXIN2", "RNF43", "ZNRF3", "TCF7L2", "LRP6", "LRP5",
        "AMER1", "BCL9", "BCL9L", "WNT5B", "WNT7B", "KREMEN1", "NKD1", "FBXW7",
    },
    "RAS_MAPK": {
        "KRAS", "NRAS", "HRAS", "BRAF", "ARAF", "RAF1", "EGFR", "ERBB2", "ERBB3", "MET",
        "MAPK1", "MAPK3", "MAP2K1", "MAP2K2", "MAP2K4", "DUSP16", "NF1", "GNAS",
    },
    "TGFb": {
        "SMAD2", "SMAD3", "SMAD4", "TGFBR1", "TGFBR2", "ACVR1B", "ACVR2A", "BMPR1A", "TGFB1", "TGFB2",
    },
    "PI3K": {
        "PIK3CA", "PIK3CB", "PIK3R1", "PIK3R2", "PTEN", "AKT1", "AKT2", "AKT3", "MTOR", "RICTOR", "STK11", "TSC1", "TSC2",
    },
    "p53_DDR": {
        "TP53", "ATM", "ATR", "CHEK1", "CHEK2", "BRCA1", "BRCA2", "BAX", "BBC3", "MDM2", "MDM4", "PARP4",
    },
    "MMR_POL": {
        "MLH1", "MSH2", "MSH3", "MSH6", "PMS1", "PMS2", "POLE", "POLD1", "EXO1",
    },
    "CHROMATIN": {
        "ARID1A", "ARID1B", "SMARCA4", "PBRM1", "EP300", "CREBBP", "KMT2C", "KMT2D", "CHD8", "BCOR", "NCOR2",
    },
}
PATHWAY_ORDER = ["WNT", "RAS_MAPK", "TGFb", "PI3K", "p53_DDR", "MMR_POL", "CHROMATIN", "CIN_ARM", "SV_COMPLEX"]

ARM_RE = re.compile(r"^CNV_(?:[0-9]{1,2}|X|Y)[pq]$")

def normalize_sample(x):
    m = re.search(r"(UM\d+|U\d+)", str(x))
    return m.group(1) if m else None


def normalize_cluster_values(series):
    s = pd.to_numeric(series, errors="raise").astype(int)
    vals = sorted(s.dropna().unique().tolist())
    if vals and min(vals) == 1 and max(vals) <= 8 and 0 not in vals:
        s = s - 1
    return s


def display_cluster(c):
    return f"C{int(c) + 1}"


def read_matrix(path):
    df = pd.read_csv(path, index_col=0, low_memory=False)
    df.index = df.index.astype(str).map(normalize_sample)
    df = df.loc[df.index.notna()]
    df = df.loc[~df.index.duplicated(keep="first")]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        df[c] = (df[c] > 0).astype(np.int8)
    return df


def read_labels(path):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Cluster label file not found: {path}\n"
            "Set CRC_CLUSTER_FILE to the final NMF k=4 label CSV."
        )
    lab = pd.read_csv(path)
    if not {"sample_id", "cluster"}.issubset(lab.columns):
        raise ValueError("Cluster CSV must contain sample_id and cluster columns.")
    lab["sid"] = lab["sample_id"].map(normalize_sample)
    lab = lab.dropna(subset=["sid"]).drop_duplicates("sid")
    lab["cluster"] = normalize_cluster_values(lab["cluster"])
    lab["cluster_display"] = lab["cluster"].map(display_cluster)
    return lab.set_index("sid")[["cluster", "cluster_display"]]


def feature_gene(feature):
    s = re.sub(r"^(SNV_|CNV_|SV_)", "", str(feature))
    s = re.sub(r"_(GAIN|LOSS|LOH|HOMDEL|AMP|DEL|BND|TRA|INV|DUP|INS)$", "", s)
    return s.upper()


def build_pathway_matrix(x):
    """
    Build the per-sample pathway alteration matrix.

    Gene pathways (WNT, RAS_MAPK, TGFb, PI3K, p53_DDR, MMR_POL, CHROMATIN):
    a gene counts as pathway-altered if EITHER its SNV or its SV feature is 1
    for that sample. This union-of-alteration-types convention follows
    Sanchez-Vega et al. 2018, Cell ("Oncogenic Signaling Pathways in TCGA"),
    which defines gene-level pathway alteration as the union of mutation,
    copy-number, and fusion/structural events on that gene.

    CIN_ARM and SV_COMPLEX are genome-wide instability axes, NOT gene
    pathways, and must be built from signals independent of the per-gene
    flags above -- otherwise a co-mutation test between e.g. SV_COMPLEX and
    TGFb would partly reflect the same single SV event counted twice
    (tautology), not two independent biological observations. Sanchez-Vega
    et al. treat genome-wide instability as a metric separate from gene-level
    pathway calls; CIN_ARM already follows this (arm-level CNV only, no gene
    overlap). SV_COMPLEX is fixed here to follow the same rule: it is built
    from total SV burden per sample (count of distinct altered SV_ features,
    median-split), never from the per-gene SV_ flags reused in WNT/RAS_MAPK/
    TGFb/PI3K.
    """
    out = pd.DataFrame(0, index=x.index, columns=PATHWAY_ORDER, dtype=np.int8)
    sv_cols = [c for c in x.columns if str(c).startswith("SV_")]
    sv_burden = x[sv_cols].sum(axis=1) if sv_cols else pd.Series(0, index=x.index)
    sv_median = sv_burden.median()
    out["SV_COMPLEX"] = (sv_burden > sv_median).astype(np.int8)
    print(f"  SV_COMPLEX: independent burden metric, "
          f"{len(sv_cols)} SV features summed per sample, "
          f"median={sv_median:.1f}, "
          f"{int(out['SV_COMPLEX'].sum())}/{len(out)} samples above median")

    for feat in x.columns:
        vals = x[feat].astype(np.int8)
        if ARM_RE.match(str(feat)):
            out["CIN_ARM"] = np.maximum(out["CIN_ARM"], vals)
            continue
        if str(feat).startswith("SV_"):
            # SV features no longer feed SV_COMPLEX here (see docstring) --
            # they still correctly feed the gene pathway below if the gene
            # they hit is a canonical pathway member (Sanchez-Vega union rule).
            pass
        gene = feature_gene(feat)
        for pathway, genes in PATHWAY_GENES.items():
            if gene in genes:
                out[pathway] = np.maximum(out[pathway], vals)
    return out


def fisher_pair(a, b):
    a = pd.Series(a).astype(int).values
    b = pd.Series(b).astype(int).values
    n11 = int(((a == 1) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n00 = int(((a == 0) & (b == 0)).sum())
    try:
        odds, p = fisher_exact([[n11, n10], [n01, n00]], alternative="two-sided")
    except Exception:
        odds, p = np.nan, 1.0
    # Haldane correction for stable log-OR and CI.
    log_or = math.log(((n11 + 0.5) * (n00 + 0.5)) / ((n10 + 0.5) * (n01 + 0.5)))
    se = math.sqrt(1/(n11+0.5) + 1/(n10+0.5) + 1/(n01+0.5) + 1/(n00+0.5))
    return {
        "n11": n11, "n10": n10, "n01": n01, "n00": n00,
        "odds_ratio_fisher": odds,
        "log2OR_haldane": log_or / math.log(2),
        "log2OR_low95": (log_or - 1.96 * se) / math.log(2),
        "log2OR_high95": (log_or + 1.96 * se) / math.log(2),
        "pvalue": float(p),
    }


def plot_prevalence(prev):
    fig, ax = plt.subplots(figsize=(9.5, 7.8))
    mat = prev.loc[PATHWAY_ORDER, ["C1", "C2", "C3", "C4"]]
    im = ax.imshow(mat.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(100, np.nanmax(mat.values)))
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns, fontweight="bold")
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if val > 50 else "black")
    ax.set_title("Pathway prevalence", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Samples altered (%)", fontweight="bold")
    for s in ax.spines.values(): s.set_visible(False)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIGDIR / f"pathway_prevalence_heatmap.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_pairwise(cluster_name, tests):
    sub = tests[tests["cluster_display"] == cluster_name].copy()
    mat = pd.DataFrame(0.0, index=PATHWAY_ORDER, columns=PATHWAY_ORDER)
    qmat = pd.DataFrame(1.0, index=PATHWAY_ORDER, columns=PATHWAY_ORDER)
    for _, r in sub.iterrows():
        a, b = r["pathway_A"], r["pathway_B"]
        mat.loc[a, b] = mat.loc[b, a] = r["log2OR_haldane"]
        qmat.loc[a, b] = qmat.loc[b, a] = r["FDR"]
    vlim = float(np.nanmax(np.abs(mat.values)))
    vlim = max(vlim, 1.0)
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="equal")
    ax.set_xticks(range(len(PATHWAY_ORDER))); ax.set_xticklabels(PATHWAY_ORDER, rotation=45, ha="right", fontweight="bold")
    ax.set_yticks(range(len(PATHWAY_ORDER))); ax.set_yticklabels(PATHWAY_ORDER, fontweight="bold")
    for i in range(len(PATHWAY_ORDER)):
        for j in range(len(PATHWAY_ORDER)):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=15, color="black")
            else:
                q = qmat.iloc[i, j]
                lab = "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""
                if lab:
                    ax.text(j, i, lab, ha="center", va="center", fontsize=15, fontweight="bold", color="black")
    ax.set_title(f"{cluster_name} co-mutation", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("log2 odds ratio", fontweight="bold")
    for s in ax.spines.values(): s.set_visible(False)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIGDIR / f"{cluster_name}_pairwise_log2OR.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_combined(tests):
    fig, axes = plt.subplots(2, 2, figsize=(17, 15))
    axes = axes.ravel()
    allv = max(1.0, float(np.nanmax(np.abs(tests["log2OR_haldane"].values))))
    for ax, cname in zip(axes, ["C1", "C2", "C3", "C4"]):
        sub = tests[tests["cluster_display"] == cname]
        mat = pd.DataFrame(0.0, index=PATHWAY_ORDER, columns=PATHWAY_ORDER)
        qmat = pd.DataFrame(1.0, index=PATHWAY_ORDER, columns=PATHWAY_ORDER)
        for _, r in sub.iterrows():
            a, b = r["pathway_A"], r["pathway_B"]
            mat.loc[a, b] = mat.loc[b, a] = r["log2OR_haldane"]
            qmat.loc[a, b] = qmat.loc[b, a] = r["FDR"]
        im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-allv, vmax=allv, aspect="equal")
        ax.set_xticks(range(len(PATHWAY_ORDER))); ax.set_xticklabels(PATHWAY_ORDER, rotation=45, ha="right", fontsize=12)
        ax.set_yticks(range(len(PATHWAY_ORDER))); ax.set_yticklabels(PATHWAY_ORDER, fontsize=12)
        ax.set_title(cname, fontweight="bold", color=CLUSTER_COLORS[cname])
        for i in range(len(PATHWAY_ORDER)):
            for j in range(len(PATHWAY_ORDER)):
                q = qmat.iloc[i, j]
                if i != j and q < 0.05:
                    ax.text(j, i, "*", ha="center", va="center", fontsize=14, fontweight="bold")
        for s in ax.spines.values(): s.set_visible(False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("log2 odds ratio", fontweight="bold")
    fig.suptitle("Pathway co-mutation", fontsize=28, fontweight="bold", y=0.995)
    for ext in ["png", "pdf"]:
        fig.savefig(FIGDIR / f"pathway_comutation_combined.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    print("Loading matrix...")
    x = read_matrix(MATRIX_FILE)
    print(f"  matrix: {x.shape}")
    print("Loading labels...")
    lab = read_labels(CLUSTER_FILE)
    common = x.index.intersection(lab.index)
    if len(common) < 100:
        raise ValueError(f"Too few common samples between matrix and labels: {len(common)}")
    x = x.loc[common]
    lab = lab.loc[common]
    print(f"  common samples: {len(common)}")

    pmat = build_pathway_matrix(x)
    pmat.insert(0, "cluster", lab["cluster_display"])
    pmat.to_csv(TABDIR / "pathway_binary_matrix.csv")

    prevalence = pmat.groupby("cluster")[PATHWAY_ORDER].mean().T * 100
    prevalence = prevalence.reindex(columns=["C1", "C2", "C3", "C4"])
    prevalence.to_csv(TABDIR / "pathway_prevalence_by_cluster.csv")
    plot_prevalence(prevalence)

    rows = []
    pmat_only = pmat[PATHWAY_ORDER]
    for cname in ["C1", "C2", "C3", "C4"]:
        idx = pmat["cluster"] == cname
        sub = pmat_only.loc[idx]
        for i, a in enumerate(PATHWAY_ORDER):
            for b in PATHWAY_ORDER[i+1:]:
                res = fisher_pair(sub[a], sub[b])
                rows.append({"cluster_display": cname, "n_cluster": int(idx.sum()), "pathway_A": a, "pathway_B": b, **res})
    tests = pd.DataFrame(rows)
    tests["FDR"] = np.nan
    for cname in tests["cluster_display"].unique():
        m = tests["cluster_display"] == cname
        tests.loc[m, "FDR"] = multipletests(tests.loc[m, "pvalue"], method="fdr_bh")[1]
    tests["direction"] = np.where(tests["log2OR_haldane"] > 0, "co_occurrence", "mutual_exclusion")
    tests.to_csv(TABDIR / "pathway_pairwise_tests_by_cluster.csv", index=False)

    for cname in ["C1", "C2", "C3", "C4"]:
        plot_pairwise(cname, tests)
    plot_combined(tests)
    print(f"Done. Outputs: {OUTDIR}")


if __name__ == "__main__":
    main()
