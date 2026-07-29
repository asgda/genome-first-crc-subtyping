#!/usr/bin/env python3
"""Characterize transcriptomic, immune and predicted drug-response phenotypes.

Precomputed GSVA, CIBERSORTx, TIDE, ESTIMATE and oncoPredict outputs are merged
with locked C1-C4 labels. Group differences use nonparametric tests and FDR
correction; predicted drug sensitivity is treated as exploratory.
"""

import os, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu
from itertools import combinations
from statsmodels.stats.multitest import fdrcorrection
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

##############################################################################
# STYLE — publication-grade, large fonts throughout
##############################################################################
plt.rcParams.update({
    # PPT-compact but readable when several panels are placed on one slide.
    "font.family": "DejaVu Sans",
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "savefig.dpi": 300,
    "axes.linewidth": 1.0,
    "figure.facecolor": "white",
})
DPI = 300  # PPT/print-ready; avoids unnecessary file bloat.

##############################################################################
# CONFIGURATION
##############################################################################
BASE = os.environ.get("CRC_BASE", str(Path(__file__).resolve().parents[1]))
DATA_DIR = f"{BASE}/crc_heterogeneity_data"

CLUSTER_FILE = os.environ.get(
    "CRC_CLUSTER_FILE",
    f"{BASE}/module05_06_loocv_results/labels/NMF_k4_LOOCV.csv",
)
# Historical input file keeps the original filename, but all display/output
# labels use the project convention: NMF only, no Brunet/OOF wording in plots.
CLUSTER_NAME = "NMF_k4_LOOCV"

OUTDIR = f"{BASE}/module9_integrative/{CLUSTER_NAME}_C1C4_ppt"
os.makedirs(OUTDIR, exist_ok=True)

HALLMARK_FILE  = f"{DATA_DIR}/HALLMARK_GSVA_FULL_CRC.csv"
KEGG_FILE      = f"{DATA_DIR}/KEGG_GSVA_FULL_CRC.csv"
REACTOME_FILE  = f"{DATA_DIR}/REACTOME_GSVA_FULL_CRC.csv"  # optional; skipped if absent
CIBER_FILE     = f"{DATA_DIR}/TRAIN_TEST_CIBERSORT_Results.csv"
TIDE_FILE      = f"{DATA_DIR}/train_test_tide_output.csv"
ESTIMATE_FILE  = f"{DATA_DIR}/estimate_scores.gct"
DRUG_FILE      = f"{DATA_DIR}/DrugPredictions.csv"
EXPR_FILE      = f"{DATA_DIR}/emtab_zscore.csv"

# Compact display caps for slide heatmaps. KEGG and Reactome are kept equal.
TOP_HALLMARK  = int(os.environ.get("CRC_M9_TOP_HALLMARK", 25))
TOP_KEGG      = int(os.environ.get("CRC_M9_TOP_KEGG", 25))
TOP_REACTOME  = int(os.environ.get("CRC_M9_TOP_REACTOME", TOP_KEGG))
TOP_DRUGS     = int(os.environ.get("CRC_M9_TOP_DRUGS", 30))
TOP_PER_CLUSTER_DRUGS = int(os.environ.get("CRC_M9_TOP_CLUSTER_DRUGS", 10))
RUN_INTEGRATIVE_SUMMARY = os.environ.get("CRC_M9_RUN_INTEGRATIVE", "0") == "1"

# Okabe-Ito inspired color-blind-safe subtype colors.
# Internal clusters remain 0-3; display labels are hard-coded as C1-C4.
CLUSTER_PALETTE = ["#009E73", "#0072B2", "#CC79A7", "#D55E00",
                   "#E69F00", "#56B4E9", "#999999", "#000000"]

CRC_DRIVERS = ["APC","KRAS","NRAS","BRAF","TP53","SMAD4","PIK3CA","EGFR",
               "ERBB2","CTNNB1","MYC","VEGFA","TGFBR2","FBXW7","SOX9","RNF43"]
EMT_POSITIVE = ["VIM","ZEB1","ZEB2","SNAI1","SNAI2","TWIST1","TWIST2","FN1","CDH2"]
EMT_NEGATIVE = ["CDH1","EPCAM"]
STEMNESS_GENES = ["SOX2","NANOG","POU5F1","KLF4","MYC"]
PROLIFERATION_GENES = ["MKI67","PCNA","TOP2A","CDK1","CCNB1","CCNB2"]

##############################################################################
# DRUG MECHANISM-OF-ACTION CATEGORIES
##############################################################################
# Substring-matched against drug names (oncoPredict/GDSC2 names often carry
# a numeric suffix, e.g. "Afatinib_1032" -- substring matching handles this).
# Not exhaustive; anything unmatched falls into "Other / uncategorised".
DRUG_CATEGORIES = {
    "EGFR / HER2 inhibitor": ["Afatinib","Erlotinib","Gefitinib","Lapatinib",
        "Sapitinib","CP724714","Tyrphostin","OSI-027"],
    "MEK / BRAF / RAS-MAPK inhibitor": ["Trametinib","Selumetinib","PD0325901",
        "Dabrafenib","Vemurafenib","Refametinib","SCH772984","Binimetinib",
        "RDEA119","CI-1040","Cobimetinib","Ulixertinib","ERK","MEK"],
    "PI3K / AKT / mTOR inhibitor": ["BEZ235","Pictilisib","Alpelisib","GDC0941",
        "MK-2206","Ipatasertib","Temsirolimus","Everolimus","Rapamycin",
        "AZD8055","GSK2126458","Omipalisib","Taselisib","PIK-93",
        "PI3K","AKT","mTOR","AZD6482"],
    "PARP / DNA-damage-response inhibitor": ["Olaparib","Talazoparib",
        "Niraparib","Rucaparib","AZD6738","AZD7762","VE-822","KU-55933",
        "Camptothecin","PARP","ATR","ATM","CHK1","CHK2","Prexasertib",
        "AZD1775","MK-1775","WEE1","NU7441"],
    "Platinum chemotherapy": ["Cisplatin","Oxaliplatin","Carboplatin"],
    "Topoisomerase inhibitor": ["Irinotecan","SN-38","Topotecan","Etoposide",
        "Doxorubicin","Epirubicin","Mitoxantrone","Teniposide"],
    "Antimetabolite": ["5-Fluorouracil","Fluorouracil","Gemcitabine",
        "Methotrexate","Pemetrexed","Cytarabine","Capecitabine","Floxuridine"],
    "Taxane / microtubule": ["Paclitaxel","Docetaxel","Vinorelbine",
        "Vinblastine","Vincristine","Epothilone","MMAE","Tubulin"],
    "CDK / cell-cycle inhibitor": ["Palbociclib","Ribociclib","Abemaciclib",
        "CGP082996","Roscovitine","Dinaciclib","Flavopiridol","CDK"],
    "BCL2 / apoptosis inhibitor": ["Venetoclax","Navitoclax","ABT-737","ABT-263",
        "S63845","A-1210477","MCL1","BCL"],
    "Proteasome / HDAC inhibitor": ["Bortezomib","Vorinostat","Panobinostat",
        "Carfilzomib","Belinostat","Entinostat","Romidepsin","HDAC","MG-132"],
    "WNT pathway inhibitor": ["WNT-C59","IWP-2","XAV939","LGK974","Porcupine"],
    "TGF-beta / SMAD inhibitor": ["Galunisertib","SB-431542","TGF","ALK5"],
    "IGF1R / insulin-pathway inhibitor": ["Linsitinib","NVP-AEW541","BMS-754807","IGF1R"],
    "VEGFR / angiogenesis inhibitor": ["Axitinib","Pazopanib","Sorafenib","Sunitinib",
        "Cediranib","Vandetanib","Nintedanib","Tivozanib","Regorafenib","VEGFR"],
    "FGFR inhibitor": ["AZD4547","BGJ398","PD173074","Dovitinib","Erdafitinib","FGFR"],
    "MET / ALK / ROS inhibitor": ["Crizotinib","Ceritinib","Alectinib","Capmatinib",
        "Tivantinib","PHA-665752","Foretinib","MET","ALK","ROS1"],
    "SRC / ABL inhibitor": ["Dasatinib","Bosutinib","Nilotinib","Imatinib",
        "Saracatinib","AZD0530","Ponatinib","SRC","ABL"],
    "JAK / STAT inhibitor": ["Ruxolitinib","Tofacitinib","AZD1480","JAK","STAT"],
    "Aurora / PLK / mitotic inhibitor": ["Alisertib","Tozasertib","Barasertib",
        "VX-680","BI-2536","Volasertib","GSK461364","MLN8054","Aurora","PLK"],
    "Epigenetic / BET inhibitor": ["JQ1","I-BET","OTX015","CPI-0610","GSK343",
        "GSK126","Tazemetostat","EPZ","UNC","BET","BRD"],
    "HSP90 inhibitor": ["Tanespimycin","Ganetespib","Luminespib","NVP-AUY922","17-AAG","HSP90"],
    "PKC inhibitor": ["Enzastaurin","Sotrastaurin","RO-31-8220","PKC"],
    "FLT3 / KIT / RET inhibitor": ["Midostaurin","Quizartinib","Cabozantinib",
        "Gilteritinib","FLT3","KIT","RET"],
}

def categorize_drug(drug_name):
    """Map GDSC/oncoPredict drug names to compact MoA categories.
    Broad fallback rules reduce the uninformative 'Other' bucket."""
    name = str(drug_name)
    name_l = re.sub(r"[_\-]+\d+$", "", name).lower()
    for cat, keys in DRUG_CATEGORIES.items():
        for k in keys:
            if k.lower() in name_l:
                return cat
    # Conservative fallbacks: these keep labels informative without inventing
    # a specific mechanism when only a generic drug-name pattern is available.
    if "mab" in name_l:
        return "Antibody / biologic"
    if "inib" in name_l:
        return "Other kinase inhibitor"
    if any(k in name_l for k in ["tox", "platin", "mycin", "rubicin"]):
        return "Cytotoxic / chemotherapy"
    return "Other"

# Short forms for side-bar display (long names overlap in narrow side-bars)
CATEGORY_SHORT = {
    "EGFR / HER2 inhibitor": "EGFR/HER2",
    "MEK / BRAF / RAS-MAPK inhibitor": "MAPK",
    "PI3K / AKT / mTOR inhibitor": "PI3K/AKT",
    "PARP / DNA-damage-response inhibitor": "DDR/PARP",
    "Platinum chemotherapy": "Platinum",
    "Topoisomerase inhibitor": "Topo",
    "Antimetabolite": "Antimetab.",
    "Taxane / microtubule": "Taxane",
    "CDK / cell-cycle inhibitor": "CDK/cycle",
    "BCL2 / apoptosis inhibitor": "BCL2/apop.",
    "Proteasome / HDAC inhibitor": "Prot/HDAC",
    "WNT pathway inhibitor": "WNT",
    "TGF-beta / SMAD inhibitor": "TGFb",
    "IGF1R / insulin-pathway inhibitor": "IGF1R",
    "VEGFR / angiogenesis inhibitor": "VEGFR",
    "FGFR inhibitor": "FGFR",
    "MET / ALK / ROS inhibitor": "MET/ALK",
    "SRC / ABL inhibitor": "SRC/ABL",
    "JAK / STAT inhibitor": "JAK/STAT",
    "Aurora / PLK / mitotic inhibitor": "Aurora/PLK",
    "Epigenetic / BET inhibitor": "Epi/BET",
    "HSP90 inhibitor": "HSP90",
    "PKC inhibitor": "PKC",
    "FLT3 / KIT / RET inhibitor": "FLT3/KIT",
    "Antibody / biologic": "Biologic",
    "Other kinase inhibitor": "Other kinase",
    "Cytotoxic / chemotherapy": "Cytotoxic",
    "Other": "Other",
}

def short_category(cat):
    return CATEGORY_SHORT.get(cat, cat[:12])

##############################################################################
# SHARED HELPERS
##############################################################################
def normalize_sample(x):
    """Canonical sample ID extraction: U#### or UM####.
    Alternation order matters -- UM\\d+ must be tried before U\\d+."""
    m = re.search(r"(UM\d+|U\d+)", str(x))
    return m.group(1) if m else np.nan



def display_cluster(c):
    """Hard-coded display convention: internal 0-3 -> C1-C4."""
    return f"C{int(c) + 1}"

def display_cluster_list(cols):
    return [display_cluster(c) for c in cols]

def normalize_cluster_values(series):
    """0-based internal labels regardless of source 0- or 1-indexing (matches
    Modules 16/17/19/20/21/22's convention)."""
    s = pd.to_numeric(series, errors="raise").astype(int)
    vals = sorted(s.dropna().unique().tolist())
    if vals and min(vals) == 1 and max(vals) <= 8 and 0 not in vals:
        s = s - 1
    return s

def load_clusters(path=CLUSTER_FILE):
    df = pd.read_csv(path)
    df["sample_id"] = df["sample_id"].astype(str).apply(normalize_sample)
    df = df.dropna(subset=["sample_id"])
    df["cluster"] = normalize_cluster_values(df["cluster"])
    # Restrict to exactly [sample_id, cluster]: the canonical label file also
    # carries cluster_display/cluster_0based columns which, left in, silently
    # leak into downstream merges and get miscounted as feature columns in
    # statistical loops (the exact bug already found and fixed in Module 11's
    # load_clusters() -- see CRC_PROJECT_CONTEXT.md Section 4).
    df = df[["sample_id", "cluster"]]
    _expected_counts = {0: 426, 1: 274, 2: 268, 3: 94}
    _observed_counts = {int(k): int(v) for k, v in df["cluster"].value_counts().sort_index().items()}
    if _observed_counts != _expected_counts:
        raise ValueError(
            f"Cluster counts after 0/1-index normalization do not match the "
            f"locked solution (observed {_observed_counts}, expected "
            f"{_expected_counts}). Check CLUSTER_FILE={path}."
        )
    return df

def significance_stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "ns"
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 0.05: return "*"
    return "ns"

def p_label(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "p=NA"
    return f"p={p:.1e}" if p < 0.001 else f"p={p:.3f}"

def compact_feature_label(x, max_len=44):
    """Short row labels for PPT heatmaps; prevents long KEGG/Reactome names
    from squeezing the heatmap body."""
    lab = str(x)
    for pref in ["HALLMARK_", "KEGG_", "REACTOME_", "GOBP_", "GO_"]:
        if lab.startswith(pref):
            lab = lab[len(pref):]
    lab = lab.replace("_", " ").strip()
    # common readable replacements
    lab = lab.replace("P53", "p53").replace("NOD LIKE", "NOD-like")
    lab = lab.replace("TOLL LIKE", "TLR")
    if len(lab) > max_len:
        lab = lab[:max_len-1].rstrip() + "…"
    return lab

def feature_global_p(stats_df, feat_col, feat):
    if stats_df is None or len(stats_df) == 0 or feat_col not in stats_df.columns:
        return np.nan
    row = stats_df.loc[stats_df[feat_col].astype(str) == str(feat)]
    if row.empty:
        return np.nan
    if "pvalue" in row.columns:
        return float(row["pvalue"].iloc[0])
    if "FDR" in row.columns:
        return float(row["FDR"].iloc[0])
    return np.nan

def feature_statistics(merged, features):
    """Kruskal-Wallis (global) per feature, BH-corrected, with effect size
    and the lowest/highest-mean cluster identification."""
    results = []
    clusters = sorted(merged["cluster"].unique())
    for feat in features:
        groups = []
        ok = True
        for c in clusters:
            vals = merged.loc[merged["cluster"]==c, feat].dropna()
            if len(vals) < 3:
                ok = False; break
            groups.append(vals)
        if not ok: continue
        try:
            _, p = kruskal(*groups)
        except Exception:
            continue
        means = merged.groupby("cluster")[feat].mean()
        results.append([feat, p, means.max()-means.min(),
                        means.idxmin(), means.idxmax()])
    res = pd.DataFrame(results, columns=["feature","pvalue","effect_size",
                                          "lowest_cluster","highest_cluster"])
    if len(res):
        res["FDR"] = fdrcorrection(res["pvalue"])[1]
    else:
        res["FDR"] = []
    return res.sort_values(["FDR","effect_size"], ascending=[True,False])

def pairwise_tests(merged, features):
    clusters = sorted(merged["cluster"].unique())
    out = []
    for feat in features:
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                c1, c2 = clusters[i], clusters[j]
                x = merged.loc[merged["cluster"]==c1, feat].dropna()
                y = merged.loc[merged["cluster"]==c2, feat].dropna()
                if len(x) < 3 or len(y) < 3: continue
                try:
                    _, p = mannwhitneyu(x, y, alternative="two-sided")
                except Exception:
                    continue
                out.append([feat, c1, c2, p])
    if not out: return pd.DataFrame()
    df = pd.DataFrame(out, columns=["feature","cluster1","cluster2","pvalue"])
    df["FDR"] = fdrcorrection(df["pvalue"])[1]
    return df

def cluster_zscore_matrix(merged, features):
    """rows=features, cols=clusters. Row-wise z-score (across clusters)
    of cluster means."""
    mat = merged.groupby("cluster")[features].mean().T
    z = mat.sub(mat.mean(axis=1), axis=0).div(
        mat.std(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return mat, z

def annotated_heatmap(zmat, stats_df, feat_col, title, cbar_label, outpath,
                       side_labels=None, side_colors=None, side_title=None,
                       cmap="RdBu_r", figsize=None):
    """Compact cluster x feature z-score heatmap for PPT.

    Fixes added in v2:
      - Long KEGG/Reactome labels are shortened so the heatmap body is not squeezed.
      - Optional side annotation is a thin strip on the RIGHT, with a separate legend.
        This prevents MoA/pathway-category text from overlapping row labels.
    """
    stats_map = dict(zip(stats_df[feat_col].astype(str), stats_df["FDR"])) if len(stats_df) else {}
    row_labels = []
    for f in zmat.index:
        star = significance_stars(stats_map.get(str(f), 1.0))
        row_labels.append(f"{compact_feature_label(f)} {'' if star=='ns' else star}")

    n_rows, n_cols = zmat.shape
    # Width is driven by heatmap columns, not by long row labels; left margin handled below.
    figsize = figsize or (max(6.2, 1.35*n_cols + 3.1 + (0.45 if side_labels is not None else 0)),
                          max(4.6, 0.31*n_rows + 1.65))

    if side_labels is not None:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 3, width_ratios=[n_cols, 0.18, 0.34], wspace=0.06)
        ax = fig.add_subplot(gs[0, 0])
        ax_side = fig.add_subplot(gs[0, 1])
        cax = fig.add_subplot(gs[0, 2])
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_side = None
        cax = None

    vmax = max(np.abs(zmat.values).max(), 1e-6)
    im = ax.imshow(zmat.values, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=16)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(display_cluster_list(zmat.columns), fontsize=18, fontweight="bold")
    ax.set_title(title, fontsize=22, fontweight="bold", pad=6)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5,n_cols,1), minor=True)
    ax.set_yticks(np.arange(-0.5,n_rows,1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax if cax is None else None, cax=cax, fraction=0.032, pad=0.02)
    cbar.set_label(cbar_label, fontsize=16, fontweight="bold")
    cbar.ax.tick_params(labelsize=15)

    if ax_side is not None:
        cats = list(side_labels)
        color_map = side_colors or {}
        for i, c in enumerate(cats):
            ax_side.add_patch(plt.Rectangle((0, i-0.5), 1, 1,
                                            color=color_map.get(c, "#CCCCCC"), ec="none"))
        ax_side.set_ylim(n_rows-0.5, -0.5)
        ax_side.set_xlim(0, 1)
        ax_side.set_xticks([]); ax_side.set_yticks([])
        for spine in ax_side.spines.values(): spine.set_visible(False)
        if side_title:
            ax_side.set_title(side_title, fontsize=15, fontweight="bold", pad=6)
        # Standalone legend like module 7: no category text inside the side strip.
        unique_cats = list(dict.fromkeys(cats))
        if unique_cats:
            leg_fig_h = max(2.6, 0.24*len(unique_cats) + 0.9)
            fig_leg, ax_leg = plt.subplots(figsize=(4.4, leg_fig_h))
            handles = [plt.Rectangle((0,0),1,1,color=color_map.get(cat, "#CCCCCC"))
                       for cat in unique_cats]
            ax_leg.legend(handles, unique_cats, loc="center left", frameon=False,
                          title=side_title or "Category", fontsize=13, title_fontsize=15)
            ax_leg.axis("off")
            plt.tight_layout()
            for ext in ["png","pdf"]:
                fig_leg.savefig(f"{outpath}_legend.{ext}", bbox_inches="tight", dpi=DPI)
            plt.close(fig_leg)

    # Manual margins avoid KEGG label distortion and preserve heatmap area.
    fig.subplots_adjust(left=0.34 if n_rows > 12 else 0.30,
                        right=0.94, bottom=0.09, top=0.92)
    for ext in ["png","pdf"]:
        fig.savefig(f"{outpath}.{ext}", bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"  Saved {outpath}.png/pdf")

def dumbbell_plot(merged, stats_df, feat_col, title, xlabel, outpath, n_top=15):
    """Point-range / dumbbell plot: one row per feature, one point per
    cluster on a shared numeric axis (cluster means). Easier to read
    directionality than a heatmap for a handful of features."""
    top = stats_df.sort_values(["FDR","effect_size"], ascending=[True,False]).head(n_top)
    feats = top[feat_col].tolist()[::-1]
    if not feats: return
    clusters = sorted(merged["cluster"].unique())
    means = merged.groupby("cluster")[feats].mean()
    fig, ax = plt.subplots(figsize=(8.5, 0.52*len(feats)+2.0))
    for i, f in enumerate(feats):
        vals = means.loc[clusters, f]
        ax.plot([vals.min(), vals.max()], [i,i], color="grey", lw=1.5, zorder=1)
        for c in clusters:
            ax.scatter(vals[c], i, color=CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)],
                      s=105, edgecolor="black", linewidth=0.8, zorder=2,
                      label=display_cluster(c) if i==0 else None)
    ax.set_yticks(range(len(feats)))
    fdr_map = dict(zip(stats_df[feat_col], stats_df["FDR"]))
    ax.set_yticklabels([f"{f}  {significance_stars(fdr_map.get(f,1.0))}" for f in feats], fontsize=15)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_title(title, fontsize=22, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0+1.2/len(feats)),
             ncol=len(clusters), framealpha=0.95, fontsize=15)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in ["png","pdf"]:
        plt.savefig(f"{outpath}.{ext}", bbox_inches="tight")
    plt.close()
    print(f"  Saved {outpath}.png/pdf")

def diverging_lollipop(zmat, stats_df, feat_col, cluster_id, title, outpath,
                       n_each=TOP_PER_CLUSTER_DRUGS, category_map=None,
                       xlabel="Cluster z-score"):
    """Per-cluster diverging lollipop: most-sensitive (z<0, blue) vs
    most-resistant (z>0, red) features, with FDR stars and optional
    category annotation. Designed for the drug-sensitivity 'showcase'."""
    col = zmat[cluster_id].sort_values()
    sens = col.head(n_each)
    res  = col.tail(n_each)
    combo = pd.concat([sens, res])
    combo = combo[~combo.index.duplicated()]
    fdr_map = dict(zip(stats_df[feat_col], stats_df["FDR"]))
    fig, ax = plt.subplots(figsize=(10.5, 0.50*len(combo)+2.0))
    y = np.arange(len(combo))[::-1]
    for yi, (feat, val) in zip(y, combo.items()):
        color = "#2166AC" if val < 0 else "#B2182B"
        ax.plot([0, val], [yi,yi], color=color, lw=2.5, zorder=1)
        ax.scatter(val, yi, color=color, s=85, zorder=2, edgecolor="black", lw=0.6)
        star = significance_stars(fdr_map.get(feat,1.0))
        label = feat if category_map is None else f"{feat}  [{category_map.get(feat,'Other')}]"
        ax.text(val + (0.05 if val>=0 else -0.05), yi, f"{label} {star}",
               ha="left" if val>=0 else "right", va="center", fontsize=14)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_title(title, fontsize=22, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    xr = max(abs(combo.min()), abs(combo.max()), 1e-6)
    ax.set_xlim(-xr*2.6, xr*2.6)
    plt.tight_layout()
    for ext in ["png","pdf"]:
        plt.savefig(f"{outpath}.{ext}", bbox_inches="tight")
    plt.close()
    print(f"  Saved {outpath}.png/pdf")

##############################################################################
# PART A — GSVA PATHWAY ACTIVITY (Hallmark + KEGG)
##############################################################################
def load_gsva(path):
    print(f"\nLoading GSVA: {os.path.basename(path)}")
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "sample_id"})
    df["sample_id"] = df["sample_id"].astype(str).apply(normalize_sample)
    df = df.dropna(subset=["sample_id"])
    print(f"  Shape: {df.shape}")
    return df

def run_gsva_analysis(cluster_df, outdir):
    print("\n" + "="*70 + "\nPART A — GSVA PATHWAY ACTIVITY\n" + "="*70)
    gsva_dir = os.path.join(outdir, "A_GSVA"); os.makedirs(gsva_dir, exist_ok=True)
    results = {}
    for name, path, top_n in [("hallmark", HALLMARK_FILE, TOP_HALLMARK),
                               ("kegg", KEGG_FILE, TOP_KEGG),
                               ("reactome", REACTOME_FILE, TOP_REACTOME)]:
        if not os.path.exists(path):
            print(f"  {path} not found -- skipping {name}"); continue
        gsva = load_gsva(path)
        merged = cluster_df.merge(gsva, on="sample_id", how="inner")
        print(f"  {name}: merged {merged.shape}")
        features = [c for c in merged.columns if c not in ("sample_id","cluster")]
        stats = feature_statistics(merged, features)
        stats.to_csv(f"{gsva_dir}/{name}_statistics.csv", index=False)

        sig = stats.loc[stats.FDR < 0.05]
        selected = (sig if len(sig) else stats).head(top_n)["feature"].tolist()
        _, zmat = cluster_zscore_matrix(merged, selected)
        annotated_heatmap(zmat, stats, "feature",
            name.capitalize(),
            "z-score", f"{gsva_dir}/{name}_heatmap")

        dumbbell_plot(merged, stats, "feature",
            name.capitalize(),
            "Mean score", f"{gsva_dir}/{name}_dumbbell", n_top=15)
        results[name] = (merged, stats)
    return results

##############################################################################
# PART B — TUMOUR MICROENVIRONMENT (CIBERSORT, TIDE, ESTIMATE)
##############################################################################
def load_cibersort():
    print("\nLoading CIBERSORT...")
    df = pd.read_csv(CIBER_FILE)
    df = df.rename(columns={"Mixture":"sample_id"})
    df["sample_id"] = df["sample_id"].astype(str).apply(normalize_sample)
    df = df.dropna(subset=["sample_id"])
    drop_cols = [c for c in ["P-value","Correlation","RMSE",
                              "Absolute score (sig.score)"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    print(f"  Shape: {df.shape}")
    return df

def load_tide():
    print("\nLoading TIDE...")
    tide = pd.read_csv(TIDE_FILE)
    tide.columns = tide.columns.astype(str).str.replace('"','',regex=False).str.strip()
    tide = tide.rename(columns={"Patient":"sample_id"})
    tide["sample_id"] = tide["sample_id"].astype(str).apply(normalize_sample)
    tide = tide.dropna(subset=["sample_id"])
    keep = [c for c in ["sample_id","TIDE","IFNG","MSI Expr Sig","Merck18",
                        "CD274","CD8","Dysfunction","Exclusion","MDSC","CAF",
                        "TAM M2"] if c in tide.columns]
    tide = tide[keep].copy()
    for c in keep:
        if c != "sample_id":
            tide[c] = pd.to_numeric(tide[c], errors="coerce")
    print(f"  Shape: {tide.shape}")
    return tide

def load_estimate():
    """FIXED: drops the GCT 'Description' column before transposing
    (standard GCT v1.2 layout is [NAME, Description, sample1, ...])."""
    print("\nLoading ESTIMATE...")
    est = pd.read_csv(ESTIMATE_FILE, sep="\t", skiprows=2)
    first_col = est.columns[0]
    est = est.rename(columns={first_col:"metric"})
    if "Description" in est.columns:
        est = est.drop(columns=["Description"])  # BUGFIX
    est = est.set_index("metric").T
    est.index.name = "sample_id"
    est = est.reset_index()
    est["sample_id"] = est["sample_id"].astype(str).str.replace(".","-",regex=False)\
                          .apply(normalize_sample)
    est = est.dropna(subset=["sample_id"])
    rename_map = {}
    for c in est.columns:
        cl = str(c).lower()
        if "immune" in cl: rename_map[c]="ImmuneScore"
        elif "stromal" in cl: rename_map[c]="StromalScore"
        elif "estimate" in cl: rename_map[c]="ESTIMATEScore"
        elif "purity" in cl: rename_map[c]="TumorPurity"
    est = est.rename(columns=rename_map)
    keep = [c for c in ["sample_id","ImmuneScore","StromalScore",
                        "ESTIMATEScore","TumorPurity"] if c in est.columns]
    est = est[keep]
    for c in keep:
        if c != "sample_id":
            est[c] = pd.to_numeric(est[c], errors="coerce")
    print(f"  Shape: {est.shape}")
    return est

def cibersort_composition_bar(merged, cell_cols, outpath):
    """Stacked bar of mean cell-type fraction per cluster -- the standard
    'immune landscape' composition plot (Thorsson et al. 2018, Immunity)."""
    means = merged.groupby("cluster")[cell_cols].mean()
    means = means.loc[:, means.mean().sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(8.5,6.5))
    bottom = np.zeros(len(means))
    colors = plt.cm.tab20.colors
    for i, c in enumerate(means.columns):
        ax.bar([display_cluster(cl) for cl in means.index], means[c].values, bottom=bottom,
              label=c, color=colors[i % len(colors)], edgecolor="white", linewidth=0.6)
        bottom += means[c].values
    ax.set_ylabel("Mean fraction", fontsize=20)
    ax.set_title("CIBERSORT", fontsize=22, fontweight="bold")
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5,-0.18), ncol=4, fontsize=13, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in ["png","pdf"]:
        plt.savefig(f"{outpath}.{ext}", bbox_inches="tight")
    plt.close()
    print(f"  Saved {outpath}.png/pdf")

def estimate_scatter(merged, outpath):
    """Immune-vs-Stromal scatter colored by cluster (standard ESTIMATE
    2-axis display; Yoshihara et al. 2013, Nat Commun)."""
    if not {"ImmuneScore","StromalScore"}.issubset(merged.columns): return
    fig, ax = plt.subplots(figsize=(7.5,6.5))
    for c in sorted(merged["cluster"].unique()):
        sub = merged[merged["cluster"]==c]
        ax.scatter(sub["StromalScore"], sub["ImmuneScore"], s=34, alpha=0.70,
                  color=CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)], label=display_cluster(c))
    ax.set_xlabel("Stromal", fontsize=20)
    ax.set_ylabel("Immune", fontsize=20)
    ax.set_title("ESTIMATE", fontsize=22, fontweight="bold")
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.legend(fontsize=15, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in ["png","pdf"]:
        plt.savefig(f"{outpath}.{ext}", bbox_inches="tight")
    plt.close()
    print(f"  Saved {outpath}.png/pdf")

def violin_panel(merged, features, stats_df, feat_col, outdir):
    """Violin plots with global Kruskal p-value and compact pairwise stars.

    Pairwise annotations show stars only for p<0.05; non-significant pairs are
    omitted to avoid crowding. Global p-value is always shown, even if ns.
    """
    import seaborn as sns
    os.makedirs(outdir, exist_ok=True)
    order = sorted(merged["cluster"].unique())
    xlabels = [display_cluster(c) for c in order]

    for feat in features:
        fig, ax = plt.subplots(figsize=(7.2, 6.8))
        sns.violinplot(data=merged, x="cluster", y=feat, inner="box", order=order,
                       palette=[CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] for i in order],
                       ax=ax)

        gp = feature_global_p(stats_df, feat_col, feat)
        ax.set_title(f"{compact_feature_label(feat, 26)}\n{p_label(gp)}", fontsize=22, fontweight="bold")
        ax.set_xlabel("Subtype", fontsize=20)
        ax.set_ylabel(compact_feature_label(feat, 24), fontsize=20)
        ax.set_xticklabels(xlabels, fontsize=18, fontweight="bold")
        ax.tick_params(axis="y", labelsize=18)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        # Pairwise raw Mann-Whitney p-values; show only significant stars.
        y_min, y_max = ax.get_ylim()
        span = y_max - y_min if y_max > y_min else 1.0
        current_y = y_max + 0.04 * span
        step = 0.075 * span
        plotted = 0
        for i, j in combinations(range(len(order)), 2):
            c1, c2 = order[i], order[j]
            x = merged.loc[merged["cluster"] == c1, feat].dropna()
            y = merged.loc[merged["cluster"] == c2, feat].dropna()
            if len(x) < 3 or len(y) < 3:
                continue
            try:
                _, p = mannwhitneyu(x, y, alternative="two-sided")
            except Exception:
                continue
            star = significance_stars(p)
            if star == "ns":
                continue
            # Limit to avoid unreadable tall annotation stacks.
            if plotted >= 6:
                break
            ax.plot([i, i, j, j], [current_y, current_y+0.015*span,
                                  current_y+0.015*span, current_y],
                    color="black", lw=1.2, clip_on=False)
            ax.text((i+j)/2, current_y+0.018*span, star,
                    ha="center", va="bottom", fontsize=17, fontweight="bold")
            current_y += step
            plotted += 1
        ax.set_ylim(y_min, current_y + 0.04*span if plotted else y_max + 0.05*span)

        fig.tight_layout()
        safe = re.sub(r"[^A-Za-z0-9_.-]","_", feat)
        for ext in ["png","pdf"]:
            fig.savefig(f"{outdir}/{safe}_violin.{ext}", bbox_inches="tight", dpi=DPI)
        plt.close(fig)

def run_immune_analysis(cluster_df, outdir):
    print("\n" + "="*70 + "\nPART B — TUMOUR MICROENVIRONMENT\n" + "="*70)
    immune_dir = os.path.join(outdir, "B_Immune"); os.makedirs(immune_dir, exist_ok=True)

    # CIBERSORT
    if os.path.exists(CIBER_FILE):
        cib = load_cibersort()
        merged = cluster_df.merge(cib, on="sample_id", how="inner")
        print(f"  CIBERSORT merged: {merged.shape}")
        cell_cols = [c for c in merged.columns if c not in ("sample_id","cluster")]
        stats = feature_statistics(merged, cell_cols)
        stats.to_csv(f"{immune_dir}/cibersort_statistics.csv", index=False)
        sig = stats.loc[stats.FDR<0.05]
        selected = (sig if len(sig) else stats).head(25)["feature"].tolist()
        _, zmat = cluster_zscore_matrix(merged, selected)
        annotated_heatmap(zmat, stats, "feature",
            "CIBERSORT",
            "z-score", f"{immune_dir}/cibersort_heatmap")
        cibersort_composition_bar(merged, cell_cols, f"{immune_dir}/cibersort_composition")
    else:
        print(f"  {CIBER_FILE} not found -- skipping CIBERSORT")

    # TIDE
    if os.path.exists(TIDE_FILE):
        tide = load_tide()
        merged = cluster_df.merge(tide, on="sample_id", how="inner")
        print(f"  TIDE merged: {merged.shape}")
        feats = [c for c in merged.columns if c not in ("sample_id","cluster")]
        stats = feature_statistics(merged, feats)
        stats.to_csv(f"{immune_dir}/tide_statistics.csv", index=False)
        pw = pairwise_tests(merged, feats)
        if len(pw): pw.to_csv(f"{immune_dir}/tide_pairwise.csv", index=False)
        sig = stats.loc[stats.FDR<0.05]
        selected = (sig if len(sig) else stats)["feature"].tolist()
        _, zmat = cluster_zscore_matrix(merged, selected)
        annotated_heatmap(zmat, stats, "feature",
            "TIDE", "Cluster z-score",
            f"{immune_dir}/tide_heatmap")
        violin_dir = f"{immune_dir}/tide_violins"; os.makedirs(violin_dir, exist_ok=True)
        priority = ["TIDE", "MSI Expr Sig", "CD8", "IFNG", "CD274", "Merck18",
                    "Dysfunction", "Exclusion", "MDSC", "CAF", "TAM M2"]
        top_feats = [f for f in priority if f in feats]
        top_feats += [f for f in stats.head(10)["feature"].tolist() if f not in top_feats]
        violin_panel(merged, top_feats[:12], stats, "feature", violin_dir)
    else:
        print(f"  {TIDE_FILE} not found -- skipping TIDE")

    # ESTIMATE
    if os.path.exists(ESTIMATE_FILE):
        est = load_estimate()
        merged = cluster_df.merge(est, on="sample_id", how="inner")
        print(f"  ESTIMATE merged: {merged.shape}")
        feats = [c for c in merged.columns if c not in ("sample_id","cluster")]
        stats = feature_statistics(merged, feats)
        stats.to_csv(f"{immune_dir}/estimate_statistics.csv", index=False)
        _, zmat = cluster_zscore_matrix(merged, feats)
        annotated_heatmap(zmat, stats, "feature",
            "ESTIMATE", "Cluster z-score",
            f"{immune_dir}/estimate_heatmap")
        violin_dir = f"{immune_dir}/estimate_violins"; os.makedirs(violin_dir, exist_ok=True)
        violin_panel(merged, feats, stats, "feature", violin_dir)
        estimate_scatter(merged, f"{immune_dir}/estimate_immune_vs_stromal")
    else:
        print(f"  {ESTIMATE_FILE} not found -- skipping ESTIMATE")

    return immune_dir

##############################################################################
# PART C — DRUG SENSITIVITY  (BUGFIXED)
##############################################################################
def load_drugs():
    """FIXED: oncoPredict/GDSC2 outputs predicted ln(IC50) -- a signed
    quantity. The previous log10(x+1) transform produced NaN for any
    value <= -1 (~20-25% of entries in a typical matrix), which were
    then silently zeroed by z-score fillna(0), flattening cluster
    differences. We now AUTO-DETECT scale: if the matrix contains a
    non-trivial fraction of negative values, treat it as already-log
    (ln IC50) and use it AS-IS (z-scored per drug downstream, no further
    transform). Only if the matrix is uniformly non-negative (i.e. raw
    IC50/AUC on a linear scale) do we apply log10(x+1)."""
    print("\nLoading drug predictions...")
    drug = pd.read_csv(DRUG_FILE)
    drug = drug.rename(columns={drug.columns[0]:"sample_id"})
    drug["sample_id"] = drug["sample_id"].astype(str).apply(normalize_sample)
    drug = drug.dropna(subset=["sample_id"])
    drug_cols = [c for c in drug.columns if c != "sample_id"]
    for c in drug_cols:
        drug[c] = pd.to_numeric(drug[c], errors="coerce")

    frac_negative = (drug[drug_cols] < 0).mean().mean()
    if frac_negative > 0.01:
        print(f"  {100*frac_negative:.1f}% of values are negative -> "
              f"treating as already-log-scale (ln IC50); NO additional "
              f"log transform applied (BUGFIX vs 09c).")
    else:
        print("  All values non-negative -> applying log10(x+1) "
              "(raw IC50/AUC scale).")
        drug[drug_cols] = np.log10(drug[drug_cols] + 1)

    print(f"  Shape: {drug.shape}")
    return drug, drug_cols

def run_drug_analysis(cluster_df, outdir):
    print("\n" + "="*70 + "\nPART C — DRUG SENSITIVITY\n" + "="*70)
    drug_dir = os.path.join(outdir, "C_Drug"); os.makedirs(drug_dir, exist_ok=True)
    if not os.path.exists(DRUG_FILE):
        print(f"  {DRUG_FILE} not found -- skipping"); return None

    drug, drug_cols = load_drugs()
    merged = cluster_df.merge(drug, on="sample_id", how="inner")
    print(f"  Merged: {merged.shape}")

    stats = feature_statistics(merged, drug_cols)
    stats = stats.rename(columns={"feature":"drug"})
    stats.to_csv(f"{drug_dir}/drug_statistics.csv", index=False)
    n_sig = (stats.FDR < 0.05).sum()
    print(f"  Drugs with FDR<0.05: {n_sig}/{len(stats)}")

    sig = stats.loc[stats.FDR < 0.05]
    selected = (sig if len(sig) else stats).head(TOP_DRUGS)["drug"].tolist()
    _, zmat = cluster_zscore_matrix(merged, selected)

    cat_map = {d: categorize_drug(d) for d in selected}
    cat_colors = dict(zip(sorted(set(cat_map.values())), plt.cm.tab20.colors))
    annotated_heatmap(zmat, stats, "drug",
        "Drug",
        "z-score", f"{drug_dir}/drug_heatmap",
        side_labels=[short_category(cat_map[d]) for d in zmat.index],
        side_colors={short_category(k):v for k,v in cat_colors.items()},
        side_title="MoA category")

    # Per-cluster diverging lollipops, using ALL drugs for ranking (not just
    # the FDR-significant top 40), so each cluster gets a genuine top/bottom.
    _, zmat_all = cluster_zscore_matrix(merged, drug_cols)
    cat_map_all = {d: categorize_drug(d) for d in drug_cols}
    for c in sorted(merged["cluster"].unique()):
        diverging_lollipop(zmat_all, stats, "drug", c,
            f"{display_cluster(c)} drug",
            f"{drug_dir}/{display_cluster(c)}_drug_lollipop",
            category_map=cat_map_all, xlabel="z-score")

    # Cluster x category mean sensitivity summary
    cat_series = pd.Series({d: categorize_drug(d) for d in drug_cols})
    cat_summary = []
    for cat in sorted(cat_series.unique()):
        ds = cat_series[cat_series==cat].index.tolist()
        means = merged.groupby("cluster")[ds].mean().mean(axis=1)
        for c, v in means.items():
            cat_summary.append({"category":cat, "cluster":c, "mean_lnIC50":v, "n_drugs":len(ds)})
    cat_df = pd.DataFrame(cat_summary)
    cat_df.to_csv(f"{drug_dir}/category_summary.csv", index=False)

    cat_pivot = cat_df.pivot(index="category", columns="cluster", values="mean_lnIC50")
    z = cat_pivot.sub(cat_pivot.mean(axis=1),axis=0).div(
        cat_pivot.std(axis=1).replace(0,np.nan),axis=0).fillna(0)
    fig, ax = plt.subplots(figsize=(1.3*z.shape[1]+5, 0.5*z.shape[0]+2))
    vmax = max(np.abs(z.values).max(), 1e-6)
    im = ax.imshow(z.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(z.index))); ax.set_yticklabels(z.index, fontsize=16)
    ax.set_xticks(range(len(z.columns))); ax.set_xticklabels([display_cluster(c) for c in z.columns],
                                                              fontsize=18, fontweight="bold")
    ax.set_title("Drug class", fontsize=22, fontweight="bold")
    for spine in ax.spines.values(): spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("z-score", fontsize=16, fontweight="bold")
    plt.tight_layout()
    for ext in ["png","pdf"]:
        plt.savefig(f"{drug_dir}/category_heatmap.{ext}", bbox_inches="tight")
    plt.close()
    print(f"  Saved {drug_dir}/category_heatmap.png/pdf")

    return merged, stats, zmat_all

##############################################################################
# PART D — TRANSCRIPTOMIC LANDSCAPE
##############################################################################
def load_expression():
    print("\nLoading expression matrix...")
    expr = pd.read_csv(EXPR_FILE, index_col=0, low_memory=False)
    expr.index = expr.index.astype(str)
    expr = expr.loc[expr.index.str.match(r"^[A-Za-z][A-Za-z0-9._-]*$", na=False)]
    expr = expr.T.apply(pd.to_numeric, errors="coerce")
    bad = [c for c in expr.columns if expr[c].isna().sum() > 0]
    expr = expr.drop(columns=bad)
    expr = expr.loc[:, expr.std(axis=0) > 0.25]
    expr.index = expr.index.astype(str).map(normalize_sample)
    expr.index.name = "sample_id"
    expr = expr.loc[~expr.index.isna()]
    expr = expr.loc[~expr.index.duplicated()]
    print(f"  Final shape: {expr.shape}")
    return expr

def deg_cluster_vs_rest(merged, cluster_id, genes):
    in_c = merged["cluster"]==cluster_id
    rows = []
    for g in genes:
        a = merged.loc[in_c, g].dropna()
        b = merged.loc[~in_c, g].dropna()
        if len(a)<3 or len(b)<3: continue
        try:
            _, p = mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            continue
        rows.append([g, a.mean(), b.mean(), a.mean()-b.mean(), p])
    deg = pd.DataFrame(rows, columns=["gene","cluster_mean","rest_mean","logFC","pvalue"])
    if len(deg):
        deg["FDR"] = fdrcorrection(deg["pvalue"])[1]
    else:
        deg["FDR"] = []
    return deg.sort_values("FDR")

def volcano_plot(deg, cluster_id, outpath):
    tmp = deg.copy()
    if len(tmp) == 0:
        return
    tmp["neglog10FDR"] = -np.log10(tmp["FDR"].clip(lower=1e-300))
    fig, ax = plt.subplots(figsize=(7.5,6.5))
    ax.scatter(tmp["logFC"], tmp["neglog10FDR"], s=14, alpha=0.4, color="grey")
    sig = tmp[(tmp.FDR<0.05) & (tmp.logFC.abs()>1)]
    if len(sig):
        ax.scatter(sig["logFC"], sig["neglog10FDR"], s=24, alpha=0.9, color="#E41A1C")
        labels = pd.concat([sig.sort_values("logFC",ascending=False).head(10),
                            sig.sort_values("logFC").head(10)])
        for _, r in labels.iterrows():
            ax.annotate(r["gene"], (r["logFC"], r["neglog10FDR"]), fontsize=9)
    for x in (1,-1): ax.axvline(x, ls="--", color="grey")
    ax.axhline(-np.log10(0.05), ls="--", color="grey")
    ax.set_xlabel("Mean Δ", fontsize=20)
    ax.set_ylabel("-log10(FDR)", fontsize=20)
    ax.set_title(f"{display_cluster(cluster_id)} vs Rest", fontsize=22, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in ["png","pdf"]:
        plt.savefig(f"{outpath}.{ext}", bbox_inches="tight")
    plt.close()

def calc_signature_scores(merged):
    for name, pos, neg in [("EMT_SCORE", EMT_POSITIVE, EMT_NEGATIVE),
                            ("STEMNESS_SCORE", STEMNESS_GENES, []),
                            ("PROLIFERATION_SCORE", PROLIFERATION_GENES, [])]:
        pos_g = [g for g in pos if g in merged.columns]
        neg_g = [g for g in neg if g in merged.columns]
        score = merged[pos_g].mean(axis=1) if pos_g else 0.0
        if neg_g:
            score = score - merged[neg_g].mean(axis=1)
        merged[name] = score
    return merged

def pca_subtype_landscape(merged, gene_cols, outpath):
    """PCA of samples in expression space, colored by genomic subtype --
    the 'subtype landscape' display style used in CMS classifier papers."""
    X = merged[gene_cols].values
    n_comp = min(2, X.shape[1])
    pcs = PCA(n_components=n_comp).fit_transform(X)
    fig, ax = plt.subplots(figsize=(7.5,6.5))
    for c in sorted(merged["cluster"].unique()):
        mask = (merged["cluster"]==c).values
        y_vals = pcs[mask,1] if n_comp>1 else np.zeros(mask.sum())
        ax.scatter(pcs[mask,0], y_vals, s=20, alpha=0.6,
                  color=CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)], label=display_cluster(c))
    ax.set_xlabel("PC1", fontsize=20); ax.set_ylabel("PC2", fontsize=20)
    ax.set_title("PCA", fontsize=22, fontweight="bold")
    ax.legend(fontsize=15, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in ["png","pdf"]:
        plt.savefig(f"{outpath}.{ext}", bbox_inches="tight")
    plt.close()
    print(f"  Saved {outpath}.png/pdf")

def run_rna_analysis(cluster_df, outdir):
    print("\n" + "="*70 + "\nPART D — TRANSCRIPTOMIC LANDSCAPE\n" + "="*70)
    rna_dir = os.path.join(outdir, "D_RNA"); os.makedirs(rna_dir, exist_ok=True)
    if not os.path.exists(EXPR_FILE):
        print(f"  {EXPR_FILE} not found -- skipping"); return None

    expr = load_expression()
    merged = cluster_df.merge(expr, left_on="sample_id", right_index=True, how="inner")
    print(f"  Merged: {merged.shape}")
    gene_cols = [c for c in merged.columns if c not in ("sample_id","cluster")]

    # DEG + volcano per cluster
    deg_dir = f"{rna_dir}/DEG"; os.makedirs(deg_dir, exist_ok=True)
    all_markers = []
    for c in sorted(merged["cluster"].unique()):
        deg = deg_cluster_vs_rest(merged, c, gene_cols)
        deg.to_csv(f"{deg_dir}/cluster_{c}_all_DEGs.csv", index=False)
        sig = deg[(deg.FDR<0.05)&(deg.logFC.abs()>0.5)]
        sig.to_csv(f"{deg_dir}/cluster_{c}_significant_DEGs.csv", index=False)
        volcano_plot(deg, c, f"{deg_dir}/cluster_{c}_volcano")
        up = sig[sig.logFC>0].sort_values("logFC",ascending=False).head(20)
        all_markers.extend(up["gene"].tolist())
    markers = list(dict.fromkeys(all_markers))
    print(f"  Marker genes identified: {len(markers)}")

    # marker heatmap
    if markers:
        marker_in_data = [g for g in markers if g in merged.columns]
        if marker_in_data:
            stub = pd.DataFrame({"feature":marker_in_data, "FDR":0.0})
            _, zmat = cluster_zscore_matrix(merged, marker_in_data)
            annotated_heatmap(zmat, stub, "feature",
                "Markers",
                "z-score", f"{rna_dir}/marker_heatmap")

    # driver gene heatmap
    drivers_present = [g for g in CRC_DRIVERS if g in merged.columns]
    if drivers_present:
        stub = pd.DataFrame({"feature":drivers_present, "FDR":0.0})
        _, zmat = cluster_zscore_matrix(merged, drivers_present)
        annotated_heatmap(zmat, stub, "feature",
            "Drivers",
            "z-score", f"{rna_dir}/driver_heatmap")

    # signature scores
    merged = calc_signature_scores(merged)
    sig_feats = ["EMT_SCORE","STEMNESS_SCORE","PROLIFERATION_SCORE"]
    stats = feature_statistics(merged, sig_feats)
    stats.to_csv(f"{rna_dir}/signature_statistics.csv", index=False)
    _, zmat = cluster_zscore_matrix(merged, sig_feats)
    annotated_heatmap(zmat, stats, "feature",
        "RNA score", "z-score", f"{rna_dir}/signature_heatmap",
        figsize=(6.3, 3.4))
    violin_panel(merged, sig_feats, stats, "feature", rna_dir)
    merged[["sample_id","cluster"]+sig_feats].to_csv(
        f"{rna_dir}/signature_scores_per_sample.csv", index=False)

    # PCA landscape
    pca_subtype_landscape(merged, gene_cols, f"{rna_dir}/pca_subtype_landscape")

    return merged, stats

##############################################################################
# PART E — INTEGRATIVE SUMMARY
##############################################################################
def run_integrative_summary(cluster_df, outdir, gsva_res, immune_dir,
                            drug_res, rna_res):
    print("\n" + "="*70 + "\nPART E — INTEGRATIVE SUMMARY\n" + "="*70)
    int_dir = os.path.join(outdir, "E_Integrative"); os.makedirs(int_dir, exist_ok=True)

    blocks, side_labels = [], []

    if "hallmark" in gsva_res:
        merged, stats = gsva_res["hallmark"]
        top = stats.head(8)["feature"].tolist()
        _, z = cluster_zscore_matrix(merged, top)
        blocks.append(z); side_labels += ["GSVA Hallmark"]*len(z)

    for fname, label, n in [("cibersort_statistics.csv","CIBERSORT",6),
                            ("estimate_statistics.csv","ESTIMATE",4),
                            ("tide_statistics.csv","TIDE",4)]:
        fpath = os.path.join(immune_dir, fname)
        if not os.path.exists(fpath): continue
        st = pd.read_csv(fpath)
        top_feats = st.head(n)["feature"].tolist()
        if "cibersort" in fname and os.path.exists(CIBER_FILE):
            m = cluster_df.merge(load_cibersort(), on="sample_id", how="inner")
        elif "estimate" in fname and os.path.exists(ESTIMATE_FILE):
            m = cluster_df.merge(load_estimate(), on="sample_id", how="inner")
        elif "tide" in fname and os.path.exists(TIDE_FILE):
            m = cluster_df.merge(load_tide(), on="sample_id", how="inner")
        else:
            continue
        top_feats = [f for f in top_feats if f in m.columns]
        if not top_feats: continue
        _, z = cluster_zscore_matrix(m, top_feats)
        blocks.append(z); side_labels += [label]*len(z)

    if drug_res is not None:
        merged, stats, zmat_all = drug_res
        top_drugs = stats.head(8)["drug"].tolist()
        z = zmat_all.loc[[d for d in top_drugs if d in zmat_all.index]]
        blocks.append(z); side_labels += ["Drug sensitivity"]*len(z)

    if rna_res is not None:
        merged, stats = rna_res
        feats = [f for f in ["EMT_SCORE","STEMNESS_SCORE","PROLIFERATION_SCORE"]
                if f in merged.columns]
        if feats:
            _, z = cluster_zscore_matrix(merged, feats)
            blocks.append(z); side_labels += ["RNA signature"]*len(z)

    if not blocks:
        print("  No data available for integrative summary -- skipping")
        return

    combined = pd.concat(blocks)
    uniq_labels = list(dict.fromkeys(side_labels))
    side_colors = dict(zip(uniq_labels, plt.cm.Set2.colors))
    stub = pd.DataFrame({"feature":combined.index, "FDR":0.0})
    annotated_heatmap(combined, stub, "feature",
        "Integrative",
        "Cluster z-score", f"{int_dir}/integrative_heatmap",
        side_labels=side_labels, side_colors=side_colors, side_title="Modality",
        figsize=(2.0*combined.shape[1]+6, 0.4*len(combined)+2))

    # Radar plot: one representative row per modality block
    seen, radar_rows = set(), []
    for lbl, idx in zip(side_labels, combined.index):
        if lbl not in seen:
            radar_rows.append(idx); seen.add(lbl)
    radar_df = combined.loc[radar_rows]
    if len(radar_df) >= 3:
        from math import pi
        categories = list(radar_df.index)
        N = len(categories)
        angles = [n/float(N)*2*pi for n in range(N)] + [0]
        fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
        for c in radar_df.columns:
            vals = radar_df[c].tolist() + [radar_df[c].iloc[0]]
            ax.plot(angles, vals, label=display_cluster(c),
                   color=CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)], lw=2)
            ax.fill(angles, vals, alpha=0.08, color=CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)])
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, fontsize=11)
        ax.set_title("Summary", fontsize=22, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3,1.1), fontsize=12)
        plt.tight_layout()
        for ext in ["png","pdf"]:
            plt.savefig(f"{int_dir}/radar_summary.{ext}", bbox_inches="tight")
        plt.close()
        print(f"  Saved {int_dir}/radar_summary.png/pdf")

##############################################################################
# MAIN
##############################################################################
if __name__ == "__main__":
    print("="*70)
    print(f"MODULE 9 — INTEGRATIVE CHARACTERIZATION ({CLUSTER_NAME})")
    print("="*70)

    cluster_df = load_clusters()
    print(f"Cluster sizes: {cluster_df['cluster'].value_counts().sort_index().to_dict()}")

    gsva_res   = run_gsva_analysis(cluster_df, OUTDIR)
    immune_dir = run_immune_analysis(cluster_df, OUTDIR)
    drug_res   = run_drug_analysis(cluster_df, OUTDIR)
    rna_res    = run_rna_analysis(cluster_df, OUTDIR)
    if RUN_INTEGRATIVE_SUMMARY:
        run_integrative_summary(cluster_df, OUTDIR, gsva_res, immune_dir, drug_res, rna_res)
    else:
        print("Skipping integrated summary panel by default (set CRC_M9_RUN_INTEGRATIVE=1 to enable).")

    print("\n" + "="*70)
    print("MODULE 9 COMPLETE")
    print(f"Outputs: {OUTDIR}")
    print("="*70)
