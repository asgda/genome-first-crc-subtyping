#!/usr/bin/env python3
"""Characterize TMB, unique SV-junction burden, ecDNA, hypoxia and purity.

Variables from the source supplementary table are merged with locked C1-C4
labels. Continuous comparisons use Kruskal-Wallis and Dunn tests; categorical
comparisons use chi-square or Fisher exact tests.
"""

import os, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import (kruskal, mannwhitneyu, chi2_contingency, fisher_exact)
from statsmodels.stats.multitest import fdrcorrection
from scikit_posthocs import posthoc_dunn   # pip install scikit-posthocs
warnings.filterwarnings("ignore")

# ── Okabe & Ito (2008) colour-blind-safe palette ─────────────────────────────
CB8 = ["#E69F00","#56B4E9","#009E73","#F0E442",
       "#0072B2","#D55E00","#CC79A7","#999999"]
# C4 (high-risk, internal index 3) uses vermillion #D55E00 throughout
CLUSTER_COLORS = {0:"#E69F00", 1:"#56B4E9", 2:"#009E73", 3:"#D55E00"}

# PPT-ready global style
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":       17,
    "axes.titlesize":  20,
    "axes.labelsize":  18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "savefig.dpi":     300,
    "axes.linewidth":  0.9,
    "figure.facecolor":"white",
})

# ── PATHS ────────────────────────────────────────────────────────────────────
BASE = os.environ.get("CRC_BASE", str(Path(__file__).resolve().parents[1]))
SUPP_TABLE = os.environ.get(
    "CRC_SUPP_TABLE",
    f"{BASE}/crc_heterogeneity_data/Supplementary_Table_01.xlsx")
CLUSTER_FILE = os.environ.get(
    "CRC_CLUSTER_FILE",
    f"{BASE}/module05_06_loocv_results/labels/NMF_k4_LOOCV.csv")
SV_BURDEN_FILE = os.environ.get(
    "CRC_M3_BURDEN",
    f"{BASE}/module3_results/module3_sv_burden_matrix.csv")
OUTDIR = os.environ.get(
    "CRC_M13_OUT", f"{BASE}/module13_genomic_characterisation")
os.makedirs(OUTDIR, exist_ok=True)
FIGDIR = os.path.join(OUTDIR, "figures"); os.makedirs(FIGDIR, exist_ok=True)

# TMB denominator — whole-genome callable territory
# Alexandrov et al. 2013, Nature established 3,000 Mb as the WGS denominator
# for somatic mutation catalogues. All 1,063 samples were called against
# GRCh38 using the same pipeline; a uniform denominator is valid.
TMB_DENOMINATOR_MB = 3_000.0
TMB_HIGH_THRESHOLD = 10.0   # FDA pembrolizumab threshold (Marabelle 2020)

ALPHA = 0.05

# ── HELPERS ─────────────────────────────────────────────────────────────────
def normalize_sample(x):
    m = re.search(r"(UM\d+|U\d+)", str(x))
    return m.group(1) if m else None

def disp_c(c):
    """0-indexed cluster → 1-indexed display label."""
    return f"C{int(c)+1}"

def normalize_cluster_values(series):
    """0-based internal labels regardless of source 0- or 1-indexing (matches
    Modules 16/17/19/20/21/22's convention)."""
    s = pd.to_numeric(series, errors="raise").astype(int)
    vals = sorted(s.dropna().unique().tolist())
    if vals and min(vals) == 1 and max(vals) <= 8 and 0 not in vals:
        s = s - 1
    return s

def savefig(path, tight=True):
    if tight:
        plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(f"{path}.{ext}", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  → {path}.png/pdf")

def stars(p):
    if   p < 0.001: return "***"
    elif p < 0.01:  return "**"
    elif p < 0.05:  return "*"
    else:           return "ns"

# ── DATA LOADING AND PREPARATION ─────────────────────────────────────────────
def load_data():
    print("Loading supplementary table …")
    st = pd.read_excel(SUPP_TABLE, index_col=0)
    st.index = st.index.astype(str)
    st["_sid"] = st.index.map(normalize_sample)
    st = st.dropna(subset=["_sid"]).set_index("_sid")

    print("Loading NMF k=4 labels …")
    cl = pd.read_csv(CLUSTER_FILE)
    cl["sample_id"] = cl["sample_id"].astype(str).apply(normalize_sample)
    cl = cl.dropna(subset=["sample_id"]).set_index("sample_id")

    df = st.join(cl[["cluster"]], how="inner")
    df["cluster"] = normalize_cluster_values(df["cluster"])
    df["cluster_display"] = df["cluster"].apply(disp_c)

    # Use the pipeline-wide canonical SV burden: one reciprocal BRASS
    # MATEID pair equals one SV junction.  Preserve the published
    # source-table value only for an explicit audit, never for analysis.
    svb = pd.read_csv(SV_BURDEN_FILE)
    if not {"sample_id", "SV_TOTAL"}.issubset(svb.columns):
        raise ValueError(
            f"SV burden file must contain sample_id and SV_TOTAL: {SV_BURDEN_FILE}"
        )
    svb["_sid"] = svb["sample_id"].map(normalize_sample)
    svb = svb.dropna(subset=["_sid"]).drop_duplicates("_sid").set_index("_sid")
    df["Structural Variants (source table)"] = pd.to_numeric(
        df["Structural Variants"], errors="coerce")
    df["Structural Variants"] = pd.to_numeric(
        svb["SV_TOTAL"], errors="coerce").reindex(df.index)
    if df["Structural Variants"].isna().any():
        missing = df.index[df["Structural Variants"].isna()].tolist()
        raise ValueError(
            f"Canonical SV burden missing for {len(missing)} labelled samples: "
            f"{missing[:10]}"
        )
    source_diff = (
        df["Structural Variants"] - df["Structural Variants (source table)"]
    )
    print(
        "Canonical SV-junction burden loaded from Module 3 "
        f"({int((source_diff != 0).sum())}/{len(df)} values differ from the "
        "published source-table summary, consistent with a small call-set "
        "provenance/version difference)."
    )
    _expected_counts = {0: 426, 1: 274, 2: 268, 3: 94}
    _observed_counts = {int(k): int(v) for k, v in df["cluster"].value_counts().sort_index().items()}
    if _observed_counts != _expected_counts:
        raise ValueError(
            f"Cluster counts after 0/1-index normalization do not match the "
            f"locked solution (observed {_observed_counts}, expected "
            f"{_expected_counts}). Check CLUSTER_FILE={CLUSTER_FILE}."
        )

    # Derived variables
    df["TMB"] = df["Total Mutation Count"] / TMB_DENOMINATOR_MB
    df["TMB_high"] = (df["TMB"] > TMB_HIGH_THRESHOLD).astype(int)
    df["ecDNA_present"] = df["ecDNA Tumour"].notna().astype(int)
    df["ecDNA_form"] = df["ecDNA Tumour"].fillna("None")
    df["RFS_event"] = (df["Recurrence"] == "Yes").astype(int)
    df["RFS_days"] = pd.to_numeric(df["Recurrence free survival days"],
                                   errors="coerce")
    df["OS_days"] = pd.to_numeric(df["Overall survival days"], errors="coerce")
    df["OS_months"] = df["OS_days"] / 30.44
    df["RFS_months"] = df["RFS_days"] / 30.44

    counts = df["cluster"].value_counts().sort_index()
    print(f"  Merged n={len(df)}")
    print(f"  Cluster sizes (0-indexed): {counts.to_dict()}")
    return df

# ── STATISTICS HELPERS ──────────────────────────────────────────────────────
def kruskal_dunn(df, col, cluster_col="cluster"):
    """
    Omnibus Kruskal-Wallis + pairwise Dunn test with BH-FDR.
    Returns (kw_stat, kw_p, dunn_df).
    Dunn test: Dunn 1964, Technometrics.
    """
    groups = [df.loc[df[cluster_col]==c, col].dropna().values
              for c in sorted(df[cluster_col].unique())]
    groups = [g for g in groups if len(g) >= 3]
    if len(groups) < 2:
        return np.nan, np.nan, pd.DataFrame()
    try:
        stat, p = kruskal(*groups)
    except Exception:
        return np.nan, np.nan, pd.DataFrame()
    # Dunn post-hoc
    sub = df[[cluster_col, col]].dropna()
    try:
        dunn = posthoc_dunn(sub, val_col=col, group_col=cluster_col,
                            p_adjust="fdr_bh")
    except Exception:
        dunn = pd.DataFrame()
    return float(stat), float(p), dunn

def fisher_c4_vs_rest(df, binary_col, c4_idx=3):
    """Fisher exact test: C4 vs all other clusters combined."""
    c4  = df[df["cluster"]==c4_idx][binary_col].dropna()
    nc4 = df[df["cluster"]!=c4_idx][binary_col].dropna()
    a, b = int(c4.sum()), int(len(c4)-c4.sum())
    c, d = int(nc4.sum()), int(len(nc4)-nc4.sum())
    _, p = fisher_exact([[a, b],[c, d]], alternative="two-sided")
    or_ = (a/max(b,1)) / (c/max(d,1)) if c*b > 0 else np.nan
    return {"OR_C4_vs_rest": or_, "fisher_p": float(p), "stars": stars(p)}

# ── MODULE A: TMB ────────────────────────────────────────────────────────────
def analysis_A_tmb(df, outdir):
    """
    TMB (total mutation count / 3,000 Mb) by subtype.
    Box plots + FDA threshold line + TMB-H enrichment bar.
    """
    print("\n" + "="*70 + "\nA. TMB by subtype\n" + "="*70)
    os.makedirs(outdir, exist_ok=True)

    # ── Statistics
    kw_stat, kw_p, dunn = kruskal_dunn(df, "TMB")
    fish = fisher_c4_vs_rest(df, "TMB_high")
    print(f"  Kruskal-Wallis: H={kw_stat:.2f}, p={kw_p:.4g}")
    print(f"  C4 vs rest TMB-H enrichment: OR={fish['OR_C4_vs_rest']:.2f}, p={fish['fisher_p']:.4g} {fish['stars']}")

    # Per-cluster TMB-H fraction
    tmb_h_frac = df.groupby("cluster")["TMB_high"].mean()
    print("  TMB-H fraction per cluster:")
    for c in sorted(df["cluster"].unique()):
        n_h = int(df[df["cluster"]==c]["TMB_high"].sum())
        n   = int((df["cluster"]==c).sum())
        print(f"    {disp_c(c)}: {n_h}/{n} = {100*n_h/n:.1f}%")

    order = sorted(df["cluster"].unique())
    labels = [disp_c(c) for c in order]
    colors = [CLUSTER_COLORS[c] for c in order]

    # ── Figure A1: box plot TMB by cluster (log10 scale)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    bdata = [df.loc[df["cluster"]==c, "TMB"].dropna().values for c in order]
    bp = ax.boxplot(bdata, patch_artist=True, widths=0.55, showfliers=True,
                    flierprops=dict(marker=".", markersize=3, alpha=0.4),
                    medianprops=dict(color="black", lw=2.2))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.80)
    ax.axhline(TMB_HIGH_THRESHOLD, color="#D55E00", ls="--", lw=1.8,
               label=f"TMB-H threshold ({TMB_HIGH_THRESHOLD} mut/Mb, FDA)")
    ax.set_xticks(range(1, len(order)+1)); ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel("TMB (mut/Mb, log₁₀ scale)")
    ax.set_title("TMB", fontweight="bold")
    ax.legend(fontsize=14, loc="upper left")
    ax.text(0.98, 0.02, f"Kruskal-Wallis p={kw_p:.3g}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=14)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    savefig(f"{outdir}/A1_TMB_boxplot")

    # ── Figure A2: stacked bar — TMB-H vs TMB-L fraction per cluster
    fig, ax = plt.subplots(figsize=(7, 5.5))
    frac_h = [df[df["cluster"]==c]["TMB_high"].mean()*100 for c in order]
    frac_l = [100 - h for h in frac_h]
    x = np.arange(len(order))
    b1 = ax.bar(x, frac_l, color=CB8[7], edgecolor="white", label="TMB-L")
    b2 = ax.bar(x, frac_h, bottom=frac_l, color=CB8[5], edgecolor="white",
                label=f"TMB-H (>{TMB_HIGH_THRESHOLD} mut/Mb)")
    for xi, (fh, c) in enumerate(zip(frac_h, order)):
        ax.text(xi, 100+1.5, f"{fh:.1f}%", ha="center", fontsize=14,
                color=CLUSTER_COLORS[c], fontweight="bold")
    # Annotate C4 bar
    ax.annotate(f"C4 Fisher p={fish['fisher_p']:.3g} {fish['stars']}",
                xy=(order.index(3), 100), xytext=(order.index(3)+0.3, 92),
                fontsize=12, color="#D55E00",
                arrowprops=dict(arrowstyle="-", color="#D55E00", lw=1))
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 115); ax.set_ylabel("% of subtype")
    ax.set_title("TMB-H fraction", fontweight="bold")
    ax.legend(fontsize=14, loc="upper left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    savefig(f"{outdir}/A2_TMB_fraction")

    # Save statistics
    stats = pd.DataFrame([{
        "cluster": disp_c(c),
        "n": int((df["cluster"]==c).sum()),
        "TMB_median": round(df[df["cluster"]==c]["TMB"].median(), 2),
        "TMB_mean": round(df[df["cluster"]==c]["TMB"].mean(), 2),
        "TMB_H_n": int(df[df["cluster"]==c]["TMB_high"].sum()),
        "TMB_H_pct": round(100*df[df["cluster"]==c]["TMB_high"].mean(), 1),
    } for c in order])
    stats.to_csv(f"{outdir}/A_TMB_statistics.csv", index=False)
    if not dunn.empty:
        dunn.to_csv(f"{outdir}/A_TMB_dunn.csv")
    print(f"  Saved to {outdir}")
    return {"kw_p": kw_p, "fish": fish, "tmb_h_frac": tmb_h_frac}

# ── MODULE B: SV BURDEN ──────────────────────────────────────────────────────
def analysis_B_sv(df, outdir):
    """
    SV count and Copy Number Segment count by subtype.
    SV burden is re-derived by Module 3 with reciprocal BRASS MATEID records
    collapsed to one junction; CN segment count is from the supplementary table.
    Copy Number Segments is used as an FGA proxy:
    — more segments = more focal CN events = higher genome alteration
    — Reference: Taylor et al. 2018, Cancer Cell (SCNA landscape)
    """
    print("\n" + "="*70 + "\nB. SV burden by subtype\n" + "="*70)
    os.makedirs(outdir, exist_ok=True)
    order = sorted(df["cluster"].unique())
    labels = [disp_c(c) for c in order]
    colors = [CLUSTER_COLORS[c] for c in order]

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
    for ax, col, title, ylabel in [
        (axes[0], "Structural Variants",   "SV Burden",     "Structural variant count"),
        (axes[1], "Copy Number Segments",  "CN Segments",   "CN segment count (FGA proxy)"),
    ]:
        kw_stat, kw_p, dunn = kruskal_dunn(df, col)
        bdata = [df.loc[df["cluster"]==c, col].dropna().values for c in order]
        bp = ax.boxplot(bdata, patch_artist=True, widths=0.55, showfliers=True,
                        flierprops=dict(marker=".", markersize=3, alpha=0.4),
                        medianprops=dict(color="black", lw=2.2))
        for patch, clr in zip(bp["boxes"], colors):
            patch.set_facecolor(clr); patch.set_alpha(0.80)
        ax.set_xticks(range(1, len(order)+1)); ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel, fontsize=22)
        ax.set_title(title, fontweight="bold", fontsize=25)
        ax.tick_params(axis="both", labelsize=20)
        ax.text(0.98, 0.97, f"KW p={kw_p:.3g}", transform=ax.transAxes,
                ha="right", va="top", fontsize=18)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        print(f"  {col}: KW H={kw_stat:.2f}, p={kw_p:.4g}")
        print("  Medians:", {disp_c(c): round(df[df["cluster"]==c][col].median(),1)
                              for c in order})
        if not dunn.empty:
            dunn.rename(columns={c: disp_c(c) for c in dunn.columns},
                        index={c: disp_c(c) for c in dunn.index}
                        ).to_csv(f"{outdir}/B_{col}_dunn.csv")
    savefig(f"{outdir}/B_SV_CNsegments")
    print(f"  Saved to {outdir}")

# ── MODULE C: ecDNA ──────────────────────────────────────────────────────────
def analysis_C_ecdna(df, outdir):
    """
    ecDNA structural forms by subtype.
    NaN = no ecDNA detected; Circular, Linear, BFB, Heavily rearranged = ecDNA+.
    — Kim et al. 2020, Nature: ecDNA Circular carries oncogene amplicons
    — BFB (breakage-fusion-bridge): cyclical chromosomal instability
    — Heavily rearranged: likely complex ecDNA/chromothripsis convergence
    — Cortes-Ciriano 2020, Nat Genet: ecDNA+ enriched in chromothripsis

    IMPORTANT CAVEAT: ecDNA+ in this dataset is 23.5% (250/1063). The
    supplementary table does not distinguish whether NaN = "not detected" or
    "not assessed". Treat NaN as "ecDNA-negative" per dataset convention,
    but acknowledge this in the paper methods.
    """
    print("\n" + "="*70 + "\nC. ecDNA by subtype\n" + "="*70)
    os.makedirs(outdir, exist_ok=True)
    order = sorted(df["cluster"].unique())
    labels = [disp_c(c) for c in order]
    colors = [CLUSTER_COLORS[c] for c in order]

    # Chi-squared: ecDNA present/absent by cluster
    ct = pd.crosstab(df["cluster"], df["ecDNA_present"])
    chi2, p_chi, dof, _ = chi2_contingency(ct)
    print(f"  Chi-squared ecDNA present: chi2={chi2:.2f}, dof={dof}, p={p_chi:.4g}")
    fish = fisher_c4_vs_rest(df, "ecDNA_present")
    print(f"  C4 vs rest Fisher: OR={fish['OR_C4_vs_rest']:.2f}, p={fish['fisher_p']:.4g} {fish['stars']}")

    # Chi-squared: ecDNA form (all 4 forms + None)
    ct_form = pd.crosstab(df["cluster"], df["ecDNA_form"])
    chi2_f, p_chi_f, dof_f, _ = chi2_contingency(ct_form)
    print(f"  Chi-squared ecDNA form: chi2={chi2_f:.2f}, dof={dof_f}, p={p_chi_f:.4g}")

    # ── Figure C1: stacked bar — ecDNA form composition per cluster
    forms_order = ["None", "Circular", "Linear", "BFB", "Heavily rearranged"]
    form_colors = {
        "None":              CB8[7],
        "Circular":          CB8[5],  # vermillion (high risk)
        "Linear":            CB8[1],
        "BFB":               CB8[4],
        "Heavily rearranged":CB8[6],
    }
    ct_norm = (ct_form.reindex(columns=[f for f in forms_order if f in ct_form.columns])
               .div(ct_form.sum(axis=1), axis=0) * 100)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    bottom = np.zeros(len(order))
    for form in [f for f in forms_order if f in ct_norm.columns]:
        vals = [ct_norm.loc[c, form] if c in ct_norm.index else 0 for c in order]
        ax.bar(range(len(order)), vals, bottom=bottom,
               color=form_colors[form], edgecolor="white",
               linewidth=0.5, label=form)
        bottom += np.array(vals)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labels)
    ax.set_ylabel("% of subtype")
    ax.set_title("ecDNA", fontweight="bold")
    ax.text(0.98, 0.97, f"χ²(form) p={p_chi_f:.3g}",
            transform=ax.transAxes, ha="right", va="top", fontsize=14)
    ax.legend(fontsize=13, loc="upper left", bbox_to_anchor=(1,1))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    savefig(f"{outdir}/C1_ecDNA_composition")

    # ── Figure C2: ecDNA+ fraction per cluster with Fisher p for C4
    frac_pos = [df[df["cluster"]==c]["ecDNA_present"].mean()*100 for c in order]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    bar_colors = [CLUSTER_COLORS[c] for c in order]
    bars = ax.bar(range(len(order)), frac_pos, color=bar_colors,
                  edgecolor="black", linewidth=0.8, alpha=0.85)
    for xi, (val, c) in enumerate(zip(frac_pos, order)):
        ax.text(xi, val+0.8, f"{val:.1f}%", ha="center", fontsize=14,
                fontweight="bold")
    if 3 in order:
        c4_xi = order.index(3)
        bars[c4_xi].set_edgecolor("#CC0000"); bars[c4_xi].set_linewidth(2.5)
        ax.annotate(f"C4 Fisher p={fish['fisher_p']:.3g} {fish['stars']}",
                    xy=(c4_xi, frac_pos[c4_xi]),
                    xytext=(c4_xi+0.4, frac_pos[c4_xi]+4),
                    fontsize=12, color="#CC0000",
                    arrowprops=dict(arrowstyle="->", color="#CC0000"))
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labels)
    ax.set_ylabel("ecDNA+ (%)")
    ax.set_title("ecDNA rate", fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    savefig(f"{outdir}/C2_ecDNA_fraction")

    # Save crosstab
    ct_form.to_csv(f"{outdir}/C_ecDNA_crosstab.csv")
    print(f"  Saved to {outdir}")
    return {"chi2_form_p": p_chi_f, "fish": fish}

# ── MODULE D: HYPOXIA ────────────────────────────────────────────────────────
def analysis_D_hypoxia(df, outdir):
    """
    Buffa 52-gene transcriptional hypoxia score by subtype.
    — Buffa et al. 2010, Br J Cancer: validated 52-gene hypoxia signature
    — Koukourakis et al. 2006, J Clin Oncol: hypoxia → FOLFOX resistance CRC
    — Mechanistic hypothesis: C4 CIN → hypoxia → immune cold → poor OS
    """
    print("\n" + "="*70 + "\nD. Hypoxia by subtype\n" + "="*70)
    os.makedirs(outdir, exist_ok=True)
    order = sorted(df["cluster"].unique())
    labels = [disp_c(c) for c in order]
    colors = [CLUSTER_COLORS[c] for c in order]

    kw_stat, kw_p, dunn = kruskal_dunn(df, "Tumour Hypoxia Buffa Score")
    print(f"  KW: H={kw_stat:.2f}, p={kw_p:.4g}")
    print("  Medians:", {disp_c(c): round(df[df["cluster"]==c]["Tumour Hypoxia Buffa Score"].median(),1)
                         for c in order})

    fig, ax = plt.subplots(figsize=(8, 6.5))
    bdata = [df.loc[df["cluster"]==c, "Tumour Hypoxia Buffa Score"].dropna().values
             for c in order]
    bp = ax.boxplot(bdata, patch_artist=True, widths=0.55, showfliers=True,
                    flierprops=dict(marker=".", markersize=3, alpha=0.4),
                    medianprops=dict(color="black", lw=2.2))
    for patch, clr in zip(bp["boxes"], colors):
        patch.set_facecolor(clr); patch.set_alpha(0.80)
    ax.axhline(0, color="grey", ls="--", lw=1, alpha=0.5)
    ax.set_xticks(range(1, len(order)+1)); ax.set_xticklabels(labels)
    ax.set_ylabel("Buffa hypoxia score")
    ax.set_title("Hypoxia", fontweight="bold")
    ax.text(0.98, 0.97, f"KW p={kw_p:.3g}", transform=ax.transAxes,
            ha="right", va="top", fontsize=14)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    savefig(f"{outdir}/D_Hypoxia_boxplot")
    if not dunn.empty:
        dunn.rename(columns={c: disp_c(c) for c in dunn.columns},
                    index={c: disp_c(c) for c in dunn.index}
                    ).to_csv(f"{outdir}/D_Hypoxia_dunn.csv")
    print(f"  Saved to {outdir}")
    return {"kw_p": kw_p}

# ── MODULE E: PURITY & CLONALITY PROXY ──────────────────────────────────────
def analysis_E_purity_clonality(df, outdir):
    """
    Tumour Cell Content (pathology-assessed purity) and Median SNV VAF
    (clonality proxy, acknowledging purity/ploidy confound) by subtype.
    — McGranahan & Swanton 2017, Science: low median VAF → high ITH
    — Dentro et al. 2021, Cell: CCF formula (purity-corrected approach)
    NOTE: Median SNV VAF is a simplified proxy. True CCF requires
    per-segment ploidy correction. This is documented as a limitation.
    """
    print("\n" + "="*70 + "\nE. Purity & clonality proxy\n" + "="*70)
    os.makedirs(outdir, exist_ok=True)
    order = sorted(df["cluster"].unique())
    labels = [disp_c(c) for c in order]
    colors = [CLUSTER_COLORS[c] for c in order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, col, title, ylabel in [
        (axes[0], "Tumour Cell Content Pathology", "Purity",    "Tumour cell content (%)"),
        (axes[1], "Median SNV VAF",                "Clonality", "Median SNV VAF\n(proxy: ↓ = higher ITH)"),
    ]:
        kw_stat, kw_p, dunn = kruskal_dunn(df, col)
        bdata = [df.loc[df["cluster"]==c, col].dropna().values for c in order]
        bp = ax.boxplot(bdata, patch_artist=True, widths=0.55, showfliers=True,
                        flierprops=dict(marker=".", markersize=3, alpha=0.4),
                        medianprops=dict(color="black", lw=2.2))
        for patch, clr in zip(bp["boxes"], colors):
            patch.set_facecolor(clr); patch.set_alpha(0.80)
        ax.set_xticks(range(1, len(order)+1)); ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
        ax.text(0.98, 0.97, f"KW p={kw_p:.3g}", transform=ax.transAxes,
                ha="right", va="top", fontsize=14)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        print(f"  {col}: KW H={kw_stat:.2f}, p={kw_p:.4g}")
        if not dunn.empty:
            dunn.rename(columns={c: disp_c(c) for c in dunn.columns},
                        index={c: disp_c(c) for c in dunn.index}
                        ).to_csv(f"{outdir}/E_{col.replace(' ','_')}_dunn.csv")
    savefig(f"{outdir}/E_Purity_Clonality")
    print(f"  Saved to {outdir}")

# ── MODULE F: iCMS CONCORDANCE ───────────────────────────────────────────────
def analysis_F_icms(df, outdir):
    """
    iCMS (intrinsic CMS) concordance with NMF subtypes.
    iCMS2 = CIN/chromosomally unstable, WNT-active
    iCMS3 = CIMP/MSI enriched, KRAS-active
    Reference: Joanito et al. 2022, Nature Genetics
    This adds a transcription-layer validation beyond CMS (which mixes
    tumour-intrinsic and microenvironment signals).
    """
    print("\n" + "="*70 + "\nF. iCMS concordance\n" + "="*70)
    os.makedirs(outdir, exist_ok=True)
    order = sorted(df["cluster"].unique())
    labels = [disp_c(c) for c in order]

    ct = pd.crosstab(df["cluster"], df["iCMS Tumour"])
    chi2, p, dof, _ = chi2_contingency(ct)
    print(f"  Chi-squared iCMS x cluster: chi2={chi2:.2f}, dof={dof}, p={p:.4g}")

    ct_norm = ct.div(ct.sum(axis=1), axis=0) * 100
    icms_order = [c for c in ["iCMS2", "iCMS3", "Undefined"] if c in ct_norm.columns]
    icms_colors = {"iCMS2": CB8[0], "iCMS3": CB8[1], "Undefined": CB8[7]}

    fig, ax = plt.subplots(figsize=(9, 6.5))
    bottom = np.zeros(len(order))
    for icms in icms_order:
        vals = [ct_norm.loc[c, icms] if c in ct_norm.index else 0 for c in order]
        ax.bar(range(len(order)), vals, bottom=bottom,
               color=icms_colors[icms], edgecolor="white", linewidth=0.5,
               label=icms)
        for xi, (val, b) in enumerate(zip(vals, bottom)):
            if val > 8:
                ax.text(xi, b+val/2, f"{val:.0f}%", ha="center", va="center",
                        fontsize=13, fontweight="bold", color="white")
        bottom += np.array(vals)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labels)
    ax.set_ylabel("% of subtype")
    ax.set_title("iCMS", fontweight="bold")
    ax.text(0.98, 0.97, f"χ² p={p:.3g}", transform=ax.transAxes,
            ha="right", va="top", fontsize=14)
    ax.legend(fontsize=14, loc="upper left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    savefig(f"{outdir}/F_iCMS_composition")
    ct.to_csv(f"{outdir}/F_iCMS_crosstab.csv")
    print(f"  Saved to {outdir}")
    return {"chi2_p": p}

# ── SUMMARY HEATMAP: all continuous variables Z-scored by cluster ────────────
def summary_heatmap(df, outdir):
    """
    Single-panel summary heatmap: cluster-mean z-score for all continuous
    genomic variables from this module. Same visual language as Module 9
    integrative heatmap. Colour scale: RdBu_r (red-blue, not red-green).
    """
    order = sorted(df["cluster"].unique())
    labels_disp = [disp_c(c) for c in order]

    features = [
        ("TMB",                             "TMB (mut/Mb)"),
        ("Structural Variants",             "SV count"),
        ("Copy Number Segments",            "CN segments"),
        ("Median SNV VAF",                  "Median SNV VAF"),
        ("Tumour Cell Content Pathology",   "Tumour purity (%)"),
        ("Tumour Hypoxia Buffa Score",      "Hypoxia score"),
    ]
    rows, row_labels = [], []
    for col, lbl in features:
        if col not in df.columns: continue
        mu = df.groupby("cluster")[col].mean().reindex(order)
        if mu.std() < 1e-9: continue
        z  = (mu - mu.mean()) / mu.std()
        rows.append(z.values); row_labels.append(lbl)

    if not rows: return
    mat = np.array(rows)
    vmax = max(abs(mat).max(), 1e-6)

    fig, ax = plt.subplots(figsize=(max(6, len(order)*1.5), 0.7*len(rows)+2.5))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labels_disp, fontsize=16)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=15)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Cluster z-score", fontsize=14, fontweight="bold")
    ax.set_title("Genomic summary", fontweight="bold")
    for spine in ax.spines.values(): spine.set_visible(False)
    savefig(f"{outdir}/summary_heatmap")
    print(f"  → {outdir}/summary_heatmap.png/pdf")

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("="*70)
    print("MODULE 13 — GENOMIC CHARACTERISATION")
    print("="*70)

    df = load_data()

    sub_dirs = {
        "A": os.path.join(FIGDIR, "A_TMB"),
        "B": os.path.join(FIGDIR, "B_SV"),
        "C": os.path.join(FIGDIR, "C_ecDNA"),
        "D": os.path.join(FIGDIR, "D_Hypoxia"),
        "E": os.path.join(FIGDIR, "E_Purity"),
        "F": os.path.join(FIGDIR, "F_iCMS"),
    }
    for d in sub_dirs.values(): os.makedirs(d, exist_ok=True)

    res_A = analysis_A_tmb(df, sub_dirs["A"])
    analysis_B_sv(df, sub_dirs["B"])
    res_C = analysis_C_ecdna(df, sub_dirs["C"])
    res_D = analysis_D_hypoxia(df, sub_dirs["D"])
    analysis_E_purity_clonality(df, sub_dirs["E"])
    res_F = analysis_F_icms(df, sub_dirs["F"])
    summary_heatmap(df, FIGDIR)

    # Master summary CSV
    order = sorted(df["cluster"].unique())
    summary = pd.DataFrame([{
        "cluster":           disp_c(c),
        "n":                 int((df["cluster"]==c).sum()),
        "TMB_median":        round(df[df["cluster"]==c]["TMB"].median(), 2),
        "TMB_H_pct":         round(100*df[df["cluster"]==c]["TMB_high"].mean(), 1),
        "SV_median":         round(df[df["cluster"]==c]["Structural Variants"].median(), 1),
        "CNseg_median":      round(df[df["cluster"]==c]["Copy Number Segments"].median(), 0),
        "ecDNA_pos_pct":     round(100*df[df["cluster"]==c]["ecDNA_present"].mean(), 1),
        "hypoxia_median":    round(df[df["cluster"]==c]["Tumour Hypoxia Buffa Score"].median(), 1),
        "purity_median_pct": round(df[df["cluster"]==c]["Tumour Cell Content Pathology"].median(), 1),
        "median_snv_vaf":    round(df[df["cluster"]==c]["Median SNV VAF"].median(), 3),
    } for c in order])
    summary.to_csv(f"{OUTDIR}/M13_summary_by_cluster.csv", index=False)

    print("\n" + "="*70)
    print("MODULE 13 COMPLETE")
    print(f"Outputs: {OUTDIR}")
    print("="*70)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
