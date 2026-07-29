#!/usr/bin/env python3
"""Visualize subtype-specific SNV, directional CNV, arm-CNV and SV profiles.

The script uses the locked C1-C4 labels, reports global and C4-versus-rest
association statistics, and writes publication-resolution heatmaps and
occurrence plots.
"""

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

##############################################################################
# STYLE — compact but readable for PPT slides
##############################################################################
plt.rcParams.update({
    "font.family": "DejaVu Sans",     # broadly available; Arial-like in PPT
    "font.size": 18,
    "axes.titlesize": 24,
    "axes.labelsize": 20,
    "xtick.labelsize": 17,
    "ytick.labelsize": 18,
    "legend.fontsize": 17,
    "savefig.dpi": 300,
    "axes.linewidth": 1.0,
    "figure.facecolor": "white",
})

##############################################################################
# CONFIGURATION
##############################################################################
BASE = os.environ.get("CRC_BASE", str(Path(__file__).resolve().parents[1]))
M1_DIR = f"{BASE}/module1_results"
M2_DIR = f"{BASE}/module2_results"
M3_DIR = f"{BASE}/module3_results"

CLUSTER_LABELS = os.environ.get(
    "CRC_CLUSTER_FILE",
    f"{BASE}/module05_06_loocv_results/labels/NMF_k4_LOOCV.csv",
)
OUT_DIR = os.environ.get("CRC_MODULE7_OUT", f"{BASE}/module7g_pathway_heatmaps_C1C4_ppt")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
Path(OUT_DIR, "tables").mkdir(parents=True, exist_ok=True)
Path(OUT_DIR, "C4_vs_nonC4_barplots").mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
C4_INTERNAL = 3                       # locked high-risk old-C3 group displayed as C4
CNV_ARM_RE = re.compile(r"^CNV_(?:[0-9]{1,2}|X|Y)[pq]$")

# Concise row counts for slide use. Increase via env if needed.
TOP_N = {
    "SNV": int(os.environ.get("CRC_M7_TOP_SNV", "25")),
    "CNV": int(os.environ.get("CRC_M7_TOP_CNV", "25")),
    "CIN": int(os.environ.get("CRC_M7_TOP_CIN", "25")),
    "SV":  int(os.environ.get("CRC_M7_TOP_SV",  "25")),
}
BARPLOT_TOP_N = int(os.environ.get("CRC_M7_C4_BARPLOT_TOP", "12"))

# Okabe-Ito / colour-blind-safe subtype palette. C4 is vermillion/red.
CLUSTER_PALETTE = {
    0: "#0072B2",   # C1 blue
    1: "#009E73",   # C2 green
    2: "#E69F00",   # C3 orange
    3: "#D55E00",   # C4 vermillion/high-risk
}
NONC4_COLOR = "#9E9E9E"
C4_COLOR = CLUSTER_PALETTE[C4_INTERNAL]

# CRC drivers kept if present, even if not top-ranked, then capped to TOP_N.
CRC_DRIVER_GENES = [
    "APC", "TP53", "KRAS", "NRAS", "BRAF", "PIK3CA", "PTEN", "SMAD4", "SMAD2", "SMAD3",
    "TGFBR2", "ACVR2A", "RNF43", "FBXW7", "CTNNB1", "AXIN1", "AXIN2", "TCF7L2",
    "MSH2", "MSH3", "MSH6", "MLH1", "PMS2", "POLE", "POLD1", "ARID1A", "EP300",
    "ERBB2", "EGFR", "MET", "MYC", "CCND1", "CDK8", "MAPK1", "MAP2K4", "KREMEN1",
    "WNT7B", "CLDN5", "MYH9", "CEP89", "DUSP16", "COL4A1", "GNAS", "BBC3", "GDF15",
]

##############################################################################
# PATHWAY ANNOTATION — extended to reduce "Other"
##############################################################################
PATHWAY_MAP = {
    # WNT / beta-catenin
    "APC":"WNT", "CTNNB1":"WNT", "AXIN1":"WNT", "AXIN2":"WNT", "RNF43":"WNT", "ZNRF3":"WNT",
    "TCF7L2":"WNT", "KREMEN1":"WNT", "DVL2":"WNT", "DVL3":"WNT", "CSNK1E":"WNT",
    "WNT7B":"WNT", "WNT5B":"WNT", "LRP6":"WNT", "NKD1":"WNT", "FBXW7":"WNT",
    "AMER1":"WNT", "BCL9":"WNT", "BCL9L":"WNT", "NLK":"WNT", "PRKD1":"WNT",
    # RTK/RAS/MAPK
    "KRAS":"MAPK", "NRAS":"MAPK", "HRAS":"MAPK", "BRAF":"MAPK", "EGFR":"MAPK", "ERBB2":"MAPK",
    "ERBB3":"MAPK", "MAPK1":"MAPK", "MAPK11":"MAPK", "MAPK12":"MAPK", "NF1":"MAPK",
    "MAP2K4":"MAPK", "MET":"MAPK", "FGFR1":"MAPK", "FGFR2":"MAPK", "RAC2":"MAPK",
    "DUSP16":"MAPK", "GNAS":"MAPK", "CEP89":"MAPK",
    # PI3K / mTOR
    "PIK3CA":"PI3K", "PIK3R1":"PI3K", "PIK3R2":"PI3K", "PIK3R5":"PI3K", "PTEN":"PI3K",
    "AKT1":"PI3K", "AKT2":"PI3K", "AKT3":"PI3K", "STK11":"PI3K", "TSC1":"PI3K",
    "TSC2":"PI3K", "RICTOR":"PI3K", "TULP3":"PI3K",
    # TGF-beta / SMAD
    "SMAD2":"TGFβ", "SMAD3":"TGFβ", "SMAD4":"TGFβ", "TGFBR1":"TGFβ", "TGFBR2":"TGFβ",
    "ACVR2A":"TGFβ", "ACVR1B":"TGFβ", "TGFB1":"TGFβ", "STRAP":"TGFβ",
    # p53 / apoptosis / DDR
    "TP53":"p53/DDR", "MDM2":"p53/DDR", "MDM4":"p53/DDR", "ATM":"p53/DDR", "ATR":"p53/DDR",
    "CHEK2":"p53/DDR", "BAX":"p53/DDR", "BBC3":"p53/DDR", "PARP4":"p53/DDR",
    # Cell cycle
    "CCND1":"Cell cycle", "CDK4":"Cell cycle", "CDK6":"Cell cycle", "CDK8":"Cell cycle",
    "CDKN2A":"Cell cycle", "RB1":"Cell cycle", "CCNE1":"Cell cycle", "LATS2":"Cell cycle", "ING1":"Cell cycle",
    # NOTCH / Hippo
    "NOTCH1":"NOTCH", "NOTCH2":"NOTCH", "NOTCH3":"NOTCH", "YAP1":"NOTCH", "TAZ":"NOTCH",
    # Chromatin
    "ARID1A":"Chromatin", "ARID1B":"Chromatin", "EP300":"Chromatin", "CREBBP":"Chromatin",
    "KMT2C":"Chromatin", "KMT2D":"Chromatin", "CHD8":"Chromatin", "BCOR":"Chromatin",
    "NCOR2":"Chromatin", "SMARCA4":"Chromatin", "PBRM1":"Chromatin",
    # MMR / polymerase
    "MSH2":"MMR", "MSH3":"MMR", "MSH6":"MMR", "MLH1":"MMR", "PMS2":"MMR", "POLE":"MMR", "POLD1":"MMR",
    # Adhesion / ECM / junction / invasion
    "CLDN5":"Adhesion", "CLDN7":"Adhesion", "CDH1":"Adhesion", "COL4A1":"Adhesion", "COL4A2":"Adhesion",
    "PKP2":"Adhesion", "ANK1":"Adhesion", "ROBO2":"Adhesion", "NTN1":"Adhesion", "DLL3":"Adhesion",
    "GDF15":"Adhesion", "SNAI1":"Adhesion", "MMP9":"Adhesion", "MYH9":"Adhesion",
    # Immune / inflammation
    "USP6":"Immune", "ID1":"Immune", "RNF6":"Immune", "ACVR2A":"TGFβ",
}
PATHWAY_ORDER = ["WNT", "MAPK", "PI3K", "TGFβ", "p53/DDR", "Cell cycle", "NOTCH", "Chromatin", "MMR", "Adhesion", "Immune", "Other"]
PATHWAY_COLORS = dict(zip(PATHWAY_ORDER, [
    "#8B4513", "#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7",
    "#56B4E9", "#A6CEE3", "#F0E442", "#999999", "#6A3D9A", "#D0D0D0"
]))

##############################################################################
# GENERAL HELPERS
##############################################################################
def extract_base_id(x):
    m = re.search(r"((?:U|UM)\d+)", str(x))
    return m.group(1) if m else None

def display_cluster(c):
    return f"C{int(c) + 1}"

def normalize_cluster_values(series):
    """0-based internal labels regardless of source 0- or 1-indexing (matches
    Modules 16/17/19/20/21/22's convention)."""
    s = pd.to_numeric(series, errors="raise").astype(int)
    vals = sorted(s.dropna().unique().tolist())
    if vals and min(vals) == 1 and max(vals) <= 8 and 0 not in vals:
        s = s - 1
    return s

def clean_label(f):
    s = re.sub(r"^(SNV_|CNV_|SV_)", "", str(f))
    s = re.sub(r"_(GAIN|LOSS|LOH|HOMDEL|AMP|DEL|BND|TRA|INV|DUP)$", "", s)
    return s

def gene_pathway(gene):
    gene = clean_label(gene).upper()
    return PATHWAY_MAP.get(gene, "Other")

def load_binary(path):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str).map(extract_base_id)
    df = df[df.index.notna()]
    # force binary 0/1 where possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        df[c] = (df[c] > 0).astype(int)
    return df

def safe_get(df, col, index):
    if col in df.columns:
        return df.reindex(index)[col].fillna(0).astype(int)
    return pd.Series(0, index=index, dtype=int)

def fisher_c4_vs_rest(series, clab):
    y = pd.Series(series).astype(int)
    c4 = (clab == C4_INTERNAL)
    a = int(y[c4].sum())              # C4 event+
    b = int(c4.sum() - a)             # C4 event-
    c = int(y[~c4].sum())             # non-C4 event+
    d = int((~c4).sum() - c)          # non-C4 event-
    try:
        odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    except Exception:
        odds, p = np.nan, 1.0
    c4_prev = a / max(1, (a + b))
    non_prev = c / max(1, (c + d))
    return {
        "c4_event_n": a,
        "c4_total_n": a + b,
        "nonc4_event_n": c,
        "nonc4_total_n": c + d,
        "c4_prev_pct": 100 * c4_prev,
        "nonc4_prev_pct": 100 * non_prev,
        "odds_ratio": odds,
        "fisher_p": p,
    }

def chi2_global(series, clab, clusters_all):
    ct, prevs = [], []
    y = pd.Series(series).astype(int)
    for c in clusters_all:
        sub = y[clab == c]
        pos = int(sub.sum())
        neg = int(len(sub) - pos)
        ct.append([neg, pos])
        prevs.append(pos / max(1, len(sub)))
    ct = np.asarray(ct)
    if ct[:, 1].sum() == 0 or ct[:, 0].sum() == 0:
        p = 1.0
    else:
        try:
            _, p, _, _ = chi2_contingency(ct)
        except Exception:
            p = 1.0
    return p, prevs

def binary_feature_stats(mat, clab, modality, feature_labeler=None, min_prev=0.01):
    rows = []
    feature_labeler = feature_labeler or clean_label
    clusters_all = sorted(clab.unique())
    for f in mat.columns:
        y = mat[f].astype(int)
        if float(y.mean()) < min_prev:
            continue
        p_global, prevs = chi2_global(y, clab, clusters_all)
        fish = fisher_c4_vs_rest(y, clab)
        c4_prev = fish["c4_prev_pct"]
        max_cluster = clusters_all[int(np.argmax(prevs))]
        rows.append({
            "modality": modality,
            "feature": f,
            "display_feature": feature_labeler(f),
            "event_type": modality,
            "global_chi2_p": p_global,
            "effect_range_pct": 100 * (max(prevs) - min(prevs)),
            "c4_is_max": int(max_cluster == C4_INTERNAL),
            "max_cluster": display_cluster(max_cluster),
            **fish,
        })
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["global_chi2_fdr"] = multipletests(res["global_chi2_p"], method="fdr_bh")[1]
    res["fisher_fdr"] = multipletests(res["fisher_p"], method="fdr_bh")[1]
    res["c4_significant_fdr"] = (res["c4_is_max"].eq(1) & (res["fisher_fdr"] < ALPHA)).astype(int)
    # For slide annotations and C4-vs-non-C4 barplots, use raw Fisher p-value,
    # as requested; FDR values remain in the statistics tables.
    res["c4_significant"] = (res["c4_is_max"].eq(1) & (res["fisher_p"] < ALPHA)).astype(int)
    return res.sort_values(["c4_significant", "fisher_p", "global_chi2_fdr", "effect_range_pct"], ascending=[False, True, True, False])

def prevalence_matrix(mat, clab, feats):
    clusters_all = sorted(clab.unique())
    rows = []
    for f in feats:
        rows.append([100 * mat.loc[clab == c, f].mean() for c in clusters_all])
    return pd.DataFrame(rows, index=feats, columns=[display_cluster(c) for c in clusters_all])

def choose_features(stats, mat_cols, top_n, driver_genes=None, force_c4=True):
    """Priority: C4-significant, global-differential, CRC drivers; cap to top_n."""
    if stats.empty:
        return []
    chosen = []
    def add(seq):
        for x in seq:
            if x in mat_cols and x not in chosen:
                chosen.append(x)
    if force_c4:
        add(stats.loc[stats["c4_significant"] == 1].sort_values(["fisher_p", "effect_range_pct"], ascending=[True, False])["feature"].tolist())
    add(stats.sort_values(["global_chi2_fdr", "effect_range_pct"], ascending=[True, False])["feature"].tolist())
    if driver_genes:
        # columns are usually prefix_gene
        drivers_present = []
        for col in mat_cols:
            gene = clean_label(col).upper()
            if gene in set(driver_genes):
                drivers_present.append(col)
        add(drivers_present)
    return chosen[:top_n]

def order_rows_by_pathway_and_c4(prev_df):
    pathways = pd.Series([gene_pathway(i) for i in prev_df.index], index=prev_df.index)
    c4_col = display_cluster(C4_INTERNAL)
    z = prev_df.sub(prev_df.mean(axis=1), axis=0).div(prev_df.std(axis=1).replace(0, 1e-8), axis=0)
    best = z.idxmax(axis=1)
    order = pd.DataFrame(index=prev_df.index)
    order["pathway"] = pathways
    order["pathway_rank"] = pathways.map({p: i for i, p in enumerate(PATHWAY_ORDER)}).fillna(999)
    order["best_rank"] = best.map({display_cluster(c): i for i, c in enumerate(sorted(clust["cluster"].unique()))}).fillna(999)
    order["c4_z"] = z[c4_col] if c4_col in z.columns else 0
    order["max_z"] = z.max(axis=1)
    idx = order.sort_values(["pathway_rank", "best_rank", "c4_z", "max_z"], ascending=[True, True, False, False]).index
    return prev_df.loc[idx], pathways.loc[idx]

def savefig(out_prefix):
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(f"{out_prefix}.{ext}", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  → {out_prefix}.png/pdf")

##############################################################################
# LOAD LOCKED LABELS
##############################################################################
print("Loading C1-C4 locked subtype labels...")
clust = pd.read_csv(CLUSTER_LABELS)
if not {"sample_id", "cluster"}.issubset(clust.columns):
    raise ValueError(f"Cluster file must contain sample_id and cluster columns: {CLUSTER_LABELS}")
clust["sample_id"] = clust["sample_id"].astype(str).map(extract_base_id)
clust = clust.dropna(subset=["sample_id"]).drop_duplicates("sample_id").set_index("sample_id")
clust["cluster"] = normalize_cluster_values(clust["cluster"])
clusters_all = sorted(clust["cluster"].unique())
sizes = clust["cluster"].value_counts().sort_index()
print("  Internal counts:", sizes.to_dict())
print("  Display counts:", {display_cluster(c): int(sizes.loc[c]) for c in clusters_all})
if C4_INTERNAL not in clusters_all:
    raise ValueError(f"Expected internal cluster {C4_INTERNAL} for displayed C4, but labels contain {clusters_all}")
_expected_counts = {0: 426, 1: 274, 2: 268, 3: 94}
_observed_counts = {int(k): int(v) for k, v in sizes.to_dict().items()}
if _observed_counts != _expected_counts:
    raise ValueError(
        f"Cluster counts after 0/1-index normalization do not match the locked "
        f"solution (observed {_observed_counts}, expected {_expected_counts}). "
        f"This guards against a 1-indexed input silently mapping to the wrong "
        f"C1-C4 identity; check CLUSTER_LABELS={CLUSTER_LABELS}."
    )

##############################################################################
# PLOTTERS
##############################################################################
def draw_subtype_bar(ax, col_clusters):
    ax.set_xlim(0, len(col_clusters))
    ax.set_ylim(0, 1)
    for j, c in enumerate(col_clusters):
        ax.add_patch(plt.Rectangle((j, 0), 1, 1, color=CLUSTER_PALETTE.get(c, "#999999")))
    ax.set_yticks([])
    ax.set_xticks(np.arange(len(col_clusters)) + 0.5)
    ax.set_xticklabels([f"{display_cluster(c)}\nn={int(sizes.loc[c])}" for c in col_clusters], fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)

def pathway_blocks(pathway_series):
    vals = pathway_series.tolist()
    blocks = []
    start, prev = 0, None
    for i, pw in enumerate(vals + [None]):
        if pw != prev:
            if prev is not None:
                blocks.append((prev, start, i))
            start, prev = i, pw
    return blocks

def plot_standard_heatmap(prev_df, pathway_series, title, cbar_label, cmap, out_prefix,
                          stats=None, show_pathway=True, vmax=None):
    n_rows, n_cols = prev_df.shape
    col_clusters = clusters_all
    fig_h = max(5.0, 0.38 * n_rows + 1.9)
    fig_w = 7.5

    if show_pathway:
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = GridSpec(2, 3, width_ratios=[7.6, 0.18, 0.35], height_ratios=[n_rows, 1.05],
                      wspace=0.08, hspace=0.06, figure=fig)
        ax = fig.add_subplot(gs[0, 0])
        ax_path = fig.add_subplot(gs[0, 1])
        cax = fig.add_subplot(gs[0, 2])
        ax_sub = fig.add_subplot(gs[1, 0])
    else:
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = GridSpec(2, 2, width_ratios=[7.6, 0.35], height_ratios=[n_rows, 1.05],
                      wspace=0.08, hspace=0.06, figure=fig)
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        ax_sub = fig.add_subplot(gs[1, 0])
        ax_path = None

    vmax = vmax or max(float(np.nanmax(prev_df.values)), 1.0)
    im = ax.imshow(prev_df.values, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_title(title, fontweight="bold", pad=8)
    ax.set_xticks(range(n_cols))
    # Avoid duplicate C1-C4 labels: cluster labels are shown only in the subtype bar.
    ax.set_xticklabels([])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(prev_df.index, fontsize=17)
    ax.tick_params(axis="y", pad=6)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Mark C4-vs-non-C4 significant features where C4 is the highest cluster.
    if stats is not None and not stats.empty:
        stat_lu = {str(r["display_feature"]): r for _, r in stats.iterrows()}
        c4_col = prev_df.columns.tolist().index(display_cluster(C4_INTERNAL)) if display_cluster(C4_INTERNAL) in prev_df.columns else None
        if c4_col is not None:
            for i, feat in enumerate(prev_df.index):
                r = stat_lu.get(str(feat))
                if r is not None and int(r.get("c4_significant", 0)) == 1:
                    ax.text(c4_col, i, "★", ha="center", va="center", fontsize=21, color="black", fontweight="bold")

    cb = fig.colorbar(im, cax=cax)
    cb.set_label(cbar_label, fontweight="bold")
    cb.ax.tick_params(labelsize=15)

    if show_pathway and ax_path is not None:
        for pw, y0, y1 in pathway_blocks(pathway_series):
            ax_path.axhspan(y0 - 0.5, y1 - 0.5, color=PATHWAY_COLORS.get(pw, PATHWAY_COLORS["Other"]))
        ax_path.set_ylim(n_rows - 0.5, -0.5)
        ax_path.set_xlim(0, 1)
        ax_path.set_xticks([])
        ax_path.set_yticks([])
        for spine in ax_path.spines.values():
            spine.set_visible(False)

    draw_subtype_bar(ax_sub, col_clusters)
    savefig(out_prefix)

def plot_cnv_directional_heatmap(signed_df, row_pathways, event_stats, out_prefix):
    """
    CNV amplification/deletion-LOH heatmap.

    Layout-only update:
      - keeps the heatmap data, colour scale, C4 stars, pathway strip and labels unchanged
      - separates x tick labels, subtype bar and legend into distinct vertical zones
      - removes the previous overlap between Amp/Del tick labels and the C1-C4 subtype bar
    """
    n_rows, n_cols = signed_df.shape
    fig_h = max(5.8, 0.38 * n_rows + 3.0)
    fig_w = 10.2
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Three vertical rows:
    #   row 0 = heatmap + pathway strip + colourbar
    #   row 1 = subtype annotation strip
    #   row 2 = legend
    gs = GridSpec(
        3, 3,
        width_ratios=[8.6, 0.18, 0.38],
        height_ratios=[n_rows, 2, 0.75],
        wspace=0.08,
        hspace=0.34,
        figure=fig,
    )

    ax = fig.add_subplot(gs[0, 0])
    ax_path = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])
    ax_sub = fig.add_subplot(gs[1, 0])
    ax_leg = fig.add_subplot(gs[2, 0])

    vmax = max(5.0, float(np.nanmax(np.abs(signed_df.values))))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(signed_df.values, aspect="auto", cmap="RdBu_r", norm=norm)

    ax.set_title("CNV", fontweight="bold", pad=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(signed_df.index, fontsize=17)
    ax.tick_params(axis="y", pad=6)

    # Avoid duplicate C1-C4 labels: subtype labels are shown only in the bottom bar.
    # The heatmap x-axis shows only event direction.
    ax.set_xticks(range(n_cols))
    event_tick_labels = ["Amp" if str(c).endswith("Amp") else "Del" for c in signed_df.columns]
    ax.set_xticklabels(
        event_tick_labels,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontweight="bold",
        fontsize=15,
    )
    ax.tick_params(axis="x", pad=4)

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Star C4-enriched significant amp/del event cells.
    # signed_df columns follow: C1 Amp, C1 Del, C2 Amp, C2 Del, ...
    stat_lu = {(str(r["display_feature"]), str(r["event_type"])): r for _, r in event_stats.iterrows()}
    cols = signed_df.columns.tolist()
    for i, gene in enumerate(signed_df.index):
        for event_type, suffix in [("Amp", "Amp"), ("Del/LOH", "Del")]:
            r = stat_lu.get((str(gene), event_type))
            if r is not None and int(r.get("c4_significant", 0)) == 1:
                target_col = f"{display_cluster(C4_INTERNAL)} {suffix}"
                if target_col in cols:
                    ax.text(
                        cols.index(target_col), i, "★",
                        ha="center", va="center",
                        fontsize=21, color="black", fontweight="bold"
                    )

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("% event\nblue=Del/LOH\nred=Amp", fontweight="bold")
    cb.ax.tick_params(labelsize=14)

    # Pathway strip on the right of the heatmap.
    for pw, y0, y1 in pathway_blocks(row_pathways):
        ax_path.axhspan(y0 - 0.5, y1 - 0.5, color=PATHWAY_COLORS.get(pw, PATHWAY_COLORS["Other"]))
    ax_path.set_ylim(n_rows - 0.5, -0.5)
    ax_path.set_xlim(0, 1)
    ax_path.set_xticks([])
    ax_path.set_yticks([])
    for spine in ax_path.spines.values():
        spine.set_visible(False)

    # Bottom cluster bar: two columns per cluster, aligned to Amp/Del columns.
    ax_sub.set_xlim(0, n_cols)
    ax_sub.set_ylim(0, 1)
    x = 0
    for c in clusters_all:
        ax_sub.add_patch(
            plt.Rectangle((x, 0), 2, 1, color=CLUSTER_PALETTE.get(c, "#999999"))
        )
        ax_sub.text(
            x + 1, 0.5,
            f"{display_cluster(c)}\nn={int(sizes.loc[c])}",
            ha="center", va="center",
            fontweight="bold",
            color="white" if c in [0, 1, 3] else "black",
            fontsize=16,
        )
        x += 2
    ax_sub.set_xticks([])
    ax_sub.set_yticks([])
    for spine in ax_sub.spines.values():
        spine.set_visible(False)

    # Legend in its own row, below the subtype strip. This prevents overlap.
    ax_leg.axis("off")
    handles = [
        mpatches.Patch(color="#B2182B", label="Amplification"),
        mpatches.Patch(color="#2166AC", label="Deletion/LOH"),
        mpatches.Patch(color="white", ec="black", label="★ C4 p<0.05"),
    ]
    ax_leg.legend(
        handles=handles,
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=14,
        handlelength=1.8,
        columnspacing=1.8,
    )

    # Keep the colourbar/pathway columns from reserving blank lower rows.
    for empty_cell in [fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]),
                       fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[2, 2])]:
        empty_cell.axis("off")

    savefig(out_prefix)

def plot_c4_barplots(stats, modality, out_prefix):
    if stats is None or stats.empty:
        return
    # Use raw Fisher p-value for plotting/annotation; FDR remains in tables.
    sub = stats[(stats["c4_is_max"] == 1) & (stats["fisher_p"] < ALPHA)].copy()
    if sub.empty:
        print(f"  No C4-enriched significant {modality} features for barplot")
        return
    sub = sub.sort_values(["fisher_p", "c4_prev_pct"], ascending=[True, False]).head(BARPLOT_TOP_N)
    sub["label"] = sub["display_feature"].astype(str)
    if "event_type" in sub.columns:
        sub["label"] = sub.apply(
            lambda r: f"{r['display_feature']} {r['event_type']}"
            if r["event_type"] not in [modality, "CIN", "SNV", "SV"]
            else str(r["display_feature"]),
            axis=1,
        )

    y = np.arange(len(sub))
    fig_h = max(4.8, 0.56 * len(sub) + 2.0)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    ax.barh(y - 0.18, sub["nonc4_prev_pct"], height=0.34, color=NONC4_COLOR, label="non-C4")
    ax.barh(y + 0.18, sub["c4_prev_pct"], height=0.34, color=C4_COLOR, label="C4")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"], fontsize=16)
    ax.invert_yaxis()
    ax.set_xlabel("Samples with event (%)")
    ax.set_title(f"{modality} C4", fontweight="bold", pad=8)

    # Give a fixed right-side column for raw p-values so labels never overlap bars.
    xmax = 128.0
    ax.set_xlim(0, xmax)
    p_col_x = 126.0
    ax.axvline(100, color="#DDDDDD", lw=0.8, zorder=0)
    ax.text(p_col_x, -0.75, "Fisher p", ha="right", va="bottom", fontsize=13, fontweight="bold")

    for i, (_, r) in enumerate(sub.iterrows()):
        nc = float(r["nonc4_prev_pct"])
        c4 = float(r["c4_prev_pct"])
        ax.text(min(nc + 1.0, 98.0), i - 0.18, f"{nc:.1f}%", va="center", fontsize=12, color="black")
        ax.text(min(c4 + 1.0, 98.0), i + 0.18, f"{c4:.1f}%", va="center", fontsize=12, color="black", fontweight="bold")
        ax.text(p_col_x, i, f"p={float(r['fisher_p']):.1e}", va="center", ha="right", fontsize=12)

    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(0.97, 0.02))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(out_prefix)

##############################################################################
# MAIN ANALYSIS
##############################################################################
all_tests = []

# ───────────────────────── SNV ─────────────────────────
print("\n" + "="*76 + "\nSNV\n" + "="*76)
p_snv = os.path.join(M1_DIR, "module1_discovery_binary_matrix.csv")
if os.path.exists(p_snv):
    snv = load_binary(p_snv)
    common = snv.index.intersection(clust.index)
    snv_c = snv.loc[common]
    clab = clust.loc[common, "cluster"]
    stats_snv = binary_feature_stats(snv_c, clab, "SNV", feature_labeler=clean_label, min_prev=0.01)
    stats_snv.to_csv(Path(OUT_DIR, "tables", "SNV_statistics.csv"), index=False)
    all_tests.append(stats_snv)
    feats = choose_features(stats_snv, snv_c.columns.tolist(), TOP_N["SNV"], CRC_DRIVER_GENES)
    print(f"  SNV rows shown: {len(feats)} -> {[clean_label(f) for f in feats]}")
    prev = prevalence_matrix(snv_c, clab, feats)
    prev.index = [clean_label(f) for f in feats]
    prev_o, pw = order_rows_by_pathway_and_c4(prev)
    plot_standard_heatmap(prev_o, pw, "SNV", "Mutated (%)", "Reds",
                          os.path.join(OUT_DIR, "SNV_heatmap"), stats=stats_snv,
                          show_pathway=True, vmax=max(5, float(prev.values.max()) * 1.15))
    plot_c4_barplots(stats_snv, "SNV", os.path.join(OUT_DIR, "C4_vs_nonC4_barplots", "SNV_C4_barplots"))
else:
    print(f"  Missing: {p_snv}")

# ───────────────────────── CNV directional ─────────────────────────
print("\n" + "="*76 + "\nCNV\n" + "="*76)
p_disc = os.path.join(M2_DIR, "module2_discovery_binary_matrix.csv")
p_dir = os.path.join(M2_DIR, "module2_discovery_directional_matrix.csv")
if os.path.exists(p_disc) and os.path.exists(p_dir):
    disc = load_binary(p_disc)
    drc = load_binary(p_dir)
    common = disc.index.intersection(clust.index)
    clab = clust.loc[common, "cluster"]
    gene_feats = [f for f in disc.columns if not CNV_ARM_RE.match(f)]
    disc_c = disc.loc[common, gene_feats]

    rows = []
    amp_mat = pd.DataFrame(index=common)
    del_mat = pd.DataFrame(index=common)
    for f in gene_feats:
        gain = safe_get(drc, f"{f}_GAIN", common)
        loss = safe_get(drc, f"{f}_LOSS", common)
        loh = safe_get(drc, f"{f}_LOH", common)
        homdel = safe_get(drc, f"{f}_HOMDEL", common)
        dele = ((loss + loh + homdel) > 0).astype(int)
        amp_mat[f] = gain
        del_mat[f] = dele
        for event_type, y in [("Amp", gain), ("Del/LOH", dele)]:
            if float(y.mean()) < 0.01:
                continue
            p_global, prevs = chi2_global(y, clab, clusters_all)
            fish = fisher_c4_vs_rest(y, clab)
            max_cluster = clusters_all[int(np.argmax(prevs))]
            rows.append({
                "modality": "CNV",
                "feature": f,
                "display_feature": clean_label(f),
                "event_type": event_type,
                "global_chi2_p": p_global,
                "effect_range_pct": 100 * (max(prevs) - min(prevs)),
                "c4_is_max": int(max_cluster == C4_INTERNAL),
                "max_cluster": display_cluster(max_cluster),
                **fish,
            })
    stats_cnv = pd.DataFrame(rows)
    if not stats_cnv.empty:
        stats_cnv["global_chi2_fdr"] = multipletests(stats_cnv["global_chi2_p"], method="fdr_bh")[1]
        stats_cnv["fisher_fdr"] = multipletests(stats_cnv["fisher_p"], method="fdr_bh")[1]
        stats_cnv["c4_significant_fdr"] = (stats_cnv["c4_is_max"].eq(1) & (stats_cnv["fisher_fdr"] < ALPHA)).astype(int)
        # For slide annotations and C4-vs-non-C4 barplots, use raw Fisher p-value;
        # FDR values remain in the statistics table.
        stats_cnv["c4_significant"] = (stats_cnv["c4_is_max"].eq(1) & (stats_cnv["fisher_p"] < ALPHA)).astype(int)
        stats_cnv = stats_cnv.sort_values(["c4_significant", "fisher_p", "global_chi2_fdr", "effect_range_pct"], ascending=[False, True, True, False])
    stats_cnv.to_csv(Path(OUT_DIR, "tables", "CNV_directional_statistics.csv"), index=False)
    all_tests.append(stats_cnv)

    # Choose genes by event-level statistics + CRC driver retention.
    selected_genes = []
    if not stats_cnv.empty:
        for col in ["c4_significant", "global_chi2_fdr"]:
            if col == "c4_significant":
                seq = stats_cnv.loc[stats_cnv["c4_significant"] == 1].sort_values(["fisher_p", "effect_range_pct"], ascending=[True, False])["feature"].tolist()
            else:
                seq = stats_cnv.sort_values(["global_chi2_fdr", "effect_range_pct"], ascending=[True, False])["feature"].tolist()
            for g in seq:
                if g not in selected_genes:
                    selected_genes.append(g)
        for g in gene_feats:
            if clean_label(g).upper() in set(CRC_DRIVER_GENES) and g not in selected_genes:
                selected_genes.append(g)
    selected_genes = selected_genes[:TOP_N["CNV"]]
    print(f"  CNV rows shown: {len(selected_genes)} -> {[clean_label(f) for f in selected_genes]}")

    # Signed directional matrix: each cluster has Amp and Del/LOH columns.
    signed_rows = []
    for f in selected_genes:
        vals = []
        for c in clusters_all:
            vals.append(100 * amp_mat.loc[clab == c, f].mean())
            vals.append(-100 * del_mat.loc[clab == c, f].mean())
        signed_rows.append(vals)
    signed_cols = []
    for c in clusters_all:
        signed_cols.extend([f"{display_cluster(c)} Amp", f"{display_cluster(c)} Del"])
    signed_df = pd.DataFrame(signed_rows, index=[clean_label(f) for f in selected_genes], columns=signed_cols)
    row_pathways = pd.Series([gene_pathway(i) for i in signed_df.index], index=signed_df.index)
    # Order by pathway, then C4 direction magnitude.
    tmp = signed_df.copy()
    tmp["_path"] = row_pathways
    tmp["_path_rank"] = row_pathways.map({p: i for i, p in enumerate(PATHWAY_ORDER)}).fillna(999)
    tmp["_c4_mag"] = np.maximum(tmp[f"{display_cluster(C4_INTERNAL)} Amp"].abs(), tmp[f"{display_cluster(C4_INTERNAL)} Del"].abs())
    signed_df = tmp.sort_values(["_path_rank", "_c4_mag"], ascending=[True, False]).drop(columns=["_path", "_path_rank", "_c4_mag"])
    row_pathways = row_pathways.loc[signed_df.index]
    plot_cnv_directional_heatmap(signed_df, row_pathways, stats_cnv,
                                 os.path.join(OUT_DIR, "CNV_amp_del_heatmap"))
    plot_c4_barplots(stats_cnv, "CNV", os.path.join(OUT_DIR, "C4_vs_nonC4_barplots", "CNV_C4_barplots"))
else:
    print(f"  Missing: {p_disc} or {p_dir}")

# ───────────────────────── CIN arm ─────────────────────────
print("\n" + "="*76 + "\nCIN\n" + "="*76)
if os.path.exists(p_disc):
    disc = load_binary(p_disc)
    common = disc.index.intersection(clust.index)
    clab = clust.loc[common, "cluster"]
    arm_feats = [f for f in disc.columns if CNV_ARM_RE.match(f)]
    if arm_feats:
        arm_c = disc.loc[common, arm_feats]
        stats_cin = binary_feature_stats(arm_c, clab, "CIN", feature_labeler=clean_label, min_prev=0.01)
        stats_cin.to_csv(Path(OUT_DIR, "tables", "CIN_arm_statistics.csv"), index=False)
        all_tests.append(stats_cin)
        feats = choose_features(stats_cin, arm_c.columns.tolist(), TOP_N["CIN"], driver_genes=None)
        # sort arms by chromosome position after selection, but keep C4 significant as priority? use stats order is more informative
        print(f"  CIN rows shown: {len(feats)} -> {[clean_label(f) for f in feats]}")
        prev = prevalence_matrix(arm_c, clab, feats)
        prev.index = [clean_label(f) for f in feats]
        # Order by C4 enrichment / differential; no pathway strip.
        c4_col = display_cluster(C4_INTERNAL)
        prev["_c4"] = prev[c4_col]
        prev["_range"] = prev[[c for c in prev.columns if c.startswith("C")]].max(axis=1) - prev[[c for c in prev.columns if c.startswith("C")]].min(axis=1)
        prev_o = prev.sort_values(["_c4", "_range"], ascending=[False, False]).drop(columns=["_c4", "_range"])
        pw = pd.Series(["Chromosome arm"] * len(prev_o), index=prev_o.index)
        plot_standard_heatmap(prev_o, pw, "CIN", "Altered (%)", "OrRd",
                              os.path.join(OUT_DIR, "CIN_heatmap"), stats=stats_cin,
                              show_pathway=False, vmax=max(5, float(prev_o.values.max()) * 1.10))
        plot_c4_barplots(stats_cin, "CIN", os.path.join(OUT_DIR, "C4_vs_nonC4_barplots", "CIN_C4_barplots"))
    else:
        print("  No CIN arm features found")
else:
    print(f"  Missing: {p_disc}")

# ───────────────────────── SV ─────────────────────────
print("\n" + "="*76 + "\nSV\n" + "="*76)
p_sv = os.path.join(M3_DIR, "module3_discovery_binary_matrix.csv")
if os.path.exists(p_sv):
    sv = load_binary(p_sv)
    common = sv.index.intersection(clust.index)
    sv_c = sv.loc[common]
    clab = clust.loc[common, "cluster"]
    stats_sv = binary_feature_stats(sv_c, clab, "SV", feature_labeler=clean_label, min_prev=0.005)
    stats_sv.to_csv(Path(OUT_DIR, "tables", "SV_statistics.csv"), index=False)
    all_tests.append(stats_sv)
    feats = choose_features(stats_sv, sv_c.columns.tolist(), min(TOP_N["SV"], sv_c.shape[1]), CRC_DRIVER_GENES)
    print(f"  SV rows shown: {len(feats)} -> {[clean_label(f) for f in feats]}")
    if feats:
        prev = prevalence_matrix(sv_c, clab, feats)
        prev.index = [clean_label(f) for f in feats]
        prev_o, pw = order_rows_by_pathway_and_c4(prev)
        plot_standard_heatmap(prev_o, pw, "SV", "Altered (%)", "Purples",
                              os.path.join(OUT_DIR, "SV_heatmap"), stats=stats_sv,
                              show_pathway=True, vmax=max(5, float(prev.values.max()) * 1.15))
        plot_c4_barplots(stats_sv, "SV", os.path.join(OUT_DIR, "C4_vs_nonC4_barplots", "SV_C4_barplots"))
else:
    print(f"  Missing: {p_sv}")

##############################################################################
# Save combined test table + pathway legend
##############################################################################
if all_tests:
    all_df = pd.concat([x for x in all_tests if x is not None and not x.empty], ignore_index=True)
    all_df.to_csv(Path(OUT_DIR, "tables", "all_C4_vs_nonC4_fisher_tests.csv"), index=False)
    c4_sig = all_df[(all_df["c4_is_max"] == 1) & (all_df["fisher_p"] < ALPHA)].copy()
    c4_sig.to_csv(Path(OUT_DIR, "tables", "C4_enriched_p_lt_0p05_features.csv"), index=False)
    # Backward-compatible filename, now raw-p based for slide annotations.
    c4_sig.to_csv(Path(OUT_DIR, "tables", "C4_enriched_significant_features.csv"), index=False)
    print(f"\nC4-enriched significant features/events: {len(c4_sig)}")
    if len(c4_sig):
        print(c4_sig[["modality", "display_feature", "event_type", "c4_prev_pct", "nonc4_prev_pct", "odds_ratio", "fisher_p"]].head(30).to_string(index=False))

# Standalone pathway legend, not embedded in heatmaps to keep panels clean.
fig, ax = plt.subplots(figsize=(8.0, 2.4))
handles = [mpatches.Patch(color=PATHWAY_COLORS[p], label=p) for p in PATHWAY_ORDER]
ax.legend(handles=handles, loc="center", ncol=4, frameon=False, title="Pathway", title_fontsize=18, fontsize=15)
ax.axis("off")
savefig(os.path.join(OUT_DIR, "pathway_legend"))

print("\n" + "="*76)
print("MODULE 7g PPT-READY C1-C4 HEATMAPS COMPLETE")
print(f"Outputs: {OUT_DIR}")
print("="*76 + "\n")
