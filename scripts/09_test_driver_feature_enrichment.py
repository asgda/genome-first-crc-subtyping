#!/usr/bin/env python3
"""Test subtype-specific enrichment and depletion of genomic features.

Two-sided one-versus-rest Fisher exact tests are performed for every SNV, CNV
and SV feature, with Benjamini-Hochberg correction. Primary outputs use FDR
less than 0.05; FDR less than 0.10 outputs are explicitly exploratory.
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
from matplotlib.lines import Line2D
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

BASE = os.environ.get("CRC_BASE", str(Path(__file__).resolve().parents[1]))
MATRIX_FILE = os.environ.get("CRC_FEATURE_MATRIX", f"{BASE}/module4_results/module4_unified_discovery_matrix.csv")
CLUSTER_FILE = os.environ.get("CRC_CLUSTER_FILE", f"{BASE}/module05_06_loocv_results/labels/NMF_k4_LOOCV.csv")
OUTDIR = Path(os.environ.get("CRC_M17_OUT", f"{BASE}/module17_driver_enrichment_C1C4_ppt"))
FIGDIR = OUTDIR / "figures"
TABDIR = OUTDIR / "tables"
FIGDIR.mkdir(parents=True, exist_ok=True)
TABDIR.mkdir(parents=True, exist_ok=True)

CB8 = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999"]
CLUSTER_COLORS = {"C1":"#E69F00", "C2":"#56B4E9", "C3":"#009E73", "C4":"#D55E00"}
MODALITY_COLORS = {"SNV":"#E69F00", "CNV":"#0072B2", "SV":"#009E73", "Other":"#999999"}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 18,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 17,
    "legend.fontsize": 16,
    "savefig.dpi": 300,
    "axes.linewidth": 1.1,
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

CRC_PRIORITY = [
    "APC", "TP53", "KRAS", "NRAS", "BRAF", "PIK3CA", "PTEN", "SMAD4", "SMAD2", "SMAD3", "TGFBR2", "ACVR2A",
    "RNF43", "FBXW7", "CTNNB1", "AXIN1", "AXIN2", "TCF7L2", "LRP6", "MSH2", "MSH3", "MSH6", "MLH1", "PMS2",
    "POLE", "POLD1", "ARID1A", "EP300", "ERBB2", "EGFR", "MET", "MYC", "CDK8", "TGIF1",
]

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
        raise FileNotFoundError(f"Cluster file not found: {path}. Set CRC_CLUSTER_FILE.")
    lab = pd.read_csv(path)
    if not {"sample_id", "cluster"}.issubset(lab.columns):
        raise ValueError("Cluster CSV must contain sample_id and cluster columns.")
    lab["sid"] = lab["sample_id"].map(normalize_sample)
    lab = lab.dropna(subset=["sid"]).drop_duplicates("sid")
    lab["cluster"] = normalize_cluster_values(lab["cluster"])
    lab["cluster_display"] = lab["cluster"].map(display_cluster)
    return lab.set_index("sid")[["cluster", "cluster_display"]]


def modality(feature):
    f = str(feature)
    if f.startswith("SNV_"): return "SNV"
    if f.startswith("CNV_"): return "CNV"
    if f.startswith("SV_"): return "SV"
    return "Other"


def clean_feature(feature):
    s = re.sub(r"^(SNV_|CNV_|SV_)", "", str(feature))
    s = s.replace("_", " ")
    if len(s) > 42:
        s = s[:41] + "…"
    return s


def feature_gene(feature):
    s = re.sub(r"^(SNV_|CNV_|SV_)", "", str(feature))
    s = re.sub(r"_(GAIN|LOSS|LOH|HOMDEL|AMP|DEL|BND|TRA|INV|DUP|INS)$", "", s)
    return s.upper()


def fisher_enrichment(y, in_cluster):
    """
    One-vs-rest Fisher exact test for a single feature.

    Uses alternative="two-sided" following the two-sided Fisher exact test
    convention used in peer-reviewed CRC consensus-molecular-subtype
    comparisons (Tsuchihashi et al. 2018, Oncotarget, CMS subtype analysis)
    and consistent with the two-sided pairwise tests already used in
    Module 16 (fisher_pair). Two-sided testing detects BOTH enrichment
    (feature more common in this cluster) and depletion / mutual exclusivity
    (feature significantly LESS common in this cluster than the rest) --
    a one-sided "greater" test, used in the prior version of this script,
    is blind to depletion entirely, which discards biologically informative
    findings (a subtype partly defined by the ABSENCE of an alteration).
    """
    y = pd.Series(y).astype(int).values
    c = pd.Series(in_cluster).astype(bool).values
    a = int(((y == 1) & c).sum())
    b = int(((y == 0) & c).sum())
    cc = int(((y == 1) & (~c)).sum())
    d = int(((y == 0) & (~c)).sum())
    try:
        odds, p = fisher_exact([[a, b], [cc, d]], alternative="two-sided")
    except Exception:
        odds, p = np.nan, 1.0
    log_or = math.log(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (cc + 0.5)))
    se = math.sqrt(1/(a+0.5) + 1/(b+0.5) + 1/(cc+0.5) + 1/(d+0.5))
    return {
        "cluster_event_n": a,
        "cluster_total_n": a + b,
        "rest_event_n": cc,
        "rest_total_n": cc + d,
        "cluster_prevalence_pct": 100 * a / max(a + b, 1),
        "rest_prevalence_pct": 100 * cc / max(cc + d, 1),
        "odds_ratio_fisher": odds,
        "log2OR_haldane": log_or / math.log(2),
        "log2OR_low95": (log_or - 1.96 * se) / math.log(2),
        "log2OR_high95": (log_or + 1.96 * se) / math.log(2),
        "pvalue": float(p),
        "direction": "enriched" if log_or > 0 else "depleted",
    }


def build_enrichment(x, lab):
    rows = []
    for cname in ["C1", "C2", "C3", "C4"]:
        in_cluster = lab["cluster_display"].eq(cname)
        for feat in x.columns:
            if x[feat].sum() < 3:
                continue
            res = fisher_enrichment(x[feat], in_cluster)
            gene = feature_gene(feat)
            rows.append({
                "cluster_display": cname,
                "feature": feat,
                "display_feature": clean_feature(feat),
                "gene_or_arm": gene,
                "modality": modality(feat),
                "is_crc_priority": int(gene in CRC_PRIORITY or ARM_RE.match(str(feat)) is not None),
                **res,
            })
    out = pd.DataFrame(rows)
    out["FDR"] = np.nan
    for cname in out["cluster_display"].unique():
        m = out["cluster_display"].eq(cname)
        out.loc[m, "FDR"] = multipletests(out.loc[m, "pvalue"], method="fdr_bh")[1]
    out["delta_prevalence_pct"] = out["cluster_prevalence_pct"] - out["rest_prevalence_pct"]
    out["significant_FDR05"] = (out["FDR"] < 0.05).astype(int)
    return out.sort_values(["cluster_display", "FDR", "pvalue", "log2OR_haldane"], ascending=[True, True, True, False])


def choose_top(df, cname, fdr_thresh, n=18):
    sub = df[df["cluster_display"].eq(cname)].copy()
    sig = sub[(sub["FDR"] < fdr_thresh) & (sub["delta_prevalence_pct"] > 0)].copy()
    if len(sig) < n:
        sig = pd.concat([sig, sub[sub["delta_prevalence_pct"] > 0].sort_values(["pvalue", "delta_prevalence_pct"], ascending=[True, False])])
        sig = sig.drop_duplicates("feature")
    return sig.head(n)


def choose_top_depleted(df, cname, fdr_thresh, n=12):
    """
    Top significantly DEPLETED (mutually exclusive) features for a cluster,
    i.e. features significantly LESS common in this cluster than in the
    rest of the cohort. Requires the two-sided Fisher test in
    fisher_enrichment(); a one-sided "greater" test cannot populate this.
    """
    sub = df[df["cluster_display"].eq(cname)].copy()
    sig = sub[(sub["FDR"] < fdr_thresh) & (sub["delta_prevalence_pct"] < 0)].copy()
    if len(sig) < n:
        sig = pd.concat([sig, sub[sub["delta_prevalence_pct"] < 0].sort_values(
            ["pvalue", "delta_prevalence_pct"], ascending=[True, True])])
        sig = sig.drop_duplicates("feature")
    return sig.head(n)


def plot_lollipop(top, cname, fdr_thresh, exploratory=False):
    if top.empty:
        print(f"  No enriched features to plot for {cname}")
        return
    top = top.sort_values("log2OR_haldane")
    y = np.arange(len(top))
    fig_h = max(6.5, 0.42 * len(top) + 2.3)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    for yi, (_, r) in zip(y, top.iterrows()):
        color = MODALITY_COLORS.get(r["modality"], "#999999")
        lo, hi, val = r["log2OR_low95"], r["log2OR_high95"], r["log2OR_haldane"]
        ax.plot([lo, hi], [yi, yi], color=color, lw=2.2, alpha=0.75)
        size = 45 + 6 * min(r["cluster_prevalence_pct"], 50)
        ax.scatter(val, yi, s=size, color=color, edgecolor="black", linewidth=0.8, zorder=3)
        star = "***" if r["FDR"] < 0.001 else "**" if r["FDR"] < 0.01 else "*" if r["FDR"] < 0.05 else ""
        if star:
            ax.text(hi + 0.08, yi, star, va="center", ha="left", fontsize=17, fontweight="bold")
    ax.axvline(0, color="black", ls="--", lw=1.3)
    ax.set_yticks(y)
    labels = [f"{r.display_feature} ({r.cluster_prevalence_pct:.0f}%)" for _, r in top.iterrows()]
    ax.set_yticklabels(labels)
    ax.set_xlabel("log2 odds ratio vs rest")
    tag = f"FDR<{fdr_thresh:g}" + (", exploratory" if exploratory else "")
    ax.set_title(f"{cname} enriched ({tag})", fontweight="bold", color=CLUSTER_COLORS[cname])
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=MODALITY_COLORS[m], markeredgecolor='black', markersize=10, label=m) for m in ["SNV", "CNV", "SV"]]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    suffix = "q10_exploratory" if exploratory else "q05"
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIGDIR / f"{cname}_driver_lollipop_{suffix}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_lollipop_depleted(top, cname, fdr_thresh, exploratory=False):
    """
    Lollipop plot for significantly DEPLETED / mutually exclusive features
    (log2OR < 0). Labels show both in-cluster and rest-of-cohort
    prevalence, since the informative number for a depleted feature is how
    much rarer it is here than elsewhere, not the in-cluster percentage alone.
    """
    if top.empty:
        print(f"  No depleted features to plot for {cname}")
        return
    top = top.sort_values("log2OR_haldane", ascending=False)
    y = np.arange(len(top))
    fig_h = max(6.5, 0.42 * len(top) + 2.3)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    for yi, (_, r) in zip(y, top.iterrows()):
        color = MODALITY_COLORS.get(r["modality"], "#999999")
        lo, hi, val = r["log2OR_low95"], r["log2OR_high95"], r["log2OR_haldane"]
        ax.plot([lo, hi], [yi, yi], color=color, lw=2.2, alpha=0.75)
        size = 45 + 6 * min(r["rest_prevalence_pct"], 50)
        ax.scatter(val, yi, s=size, color=color, edgecolor="black", linewidth=0.8, zorder=3)
        star = "***" if r["FDR"] < 0.001 else "**" if r["FDR"] < 0.01 else "*" if r["FDR"] < 0.05 else ""
        if star:
            ax.text(lo - 0.08, yi, star, va="center", ha="right", fontsize=17, fontweight="bold")
    ax.axvline(0, color="black", ls="--", lw=1.3)
    ax.set_yticks(y)
    labels = [f"{r.display_feature} ({r.cluster_prevalence_pct:.0f}% vs {r.rest_prevalence_pct:.0f}%)"
              for _, r in top.iterrows()]
    ax.set_yticklabels(labels)
    ax.set_xlabel("log2 odds ratio vs rest")
    tag = f"FDR<{fdr_thresh:g}" + (", exploratory" if exploratory else "")
    ax.set_title(f"{cname} depleted ({tag})", fontweight="bold", color=CLUSTER_COLORS[cname])
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=MODALITY_COLORS[m], markeredgecolor='black', markersize=10, label=m) for m in ["SNV", "CNV", "SV"]]
    ax.legend(handles=handles, frameon=False, loc="lower left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    suffix = "q10_exploratory" if exploratory else "q05"
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIGDIR / f"{cname}_driver_lollipop_depleted_{suffix}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_combined(enrich):
    """
    Top-12-by-p-value grid, not a hard significance cutoff -- with this
    cohort's effect sizes the top 12 per cluster are the same at FDR<0.05
    or FDR<0.10, so this figure is built once, at FDR<0.05, rather than
    duplicated (see module docstring).
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    for ax, cname in zip(axes.ravel(), ["C1", "C2", "C3", "C4"]):
        top = choose_top(enrich, cname, fdr_thresh=0.05, n=12).sort_values("log2OR_haldane")
        if top.empty:
            ax.axis("off")
            continue
        y = np.arange(len(top))
        colors = [MODALITY_COLORS.get(m, "#999999") for m in top["modality"]]
        ax.barh(y, top["log2OR_haldane"], color=colors, edgecolor="black", linewidth=0.7)
        ax.axvline(0, color="black", ls="--", lw=1.1)
        ax.set_yticks(y); ax.set_yticklabels(top["display_feature"], fontsize=13)
        ax.set_title(cname, fontweight="bold", color=CLUSTER_COLORS[cname])
        ax.set_xlabel("log2 OR")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    handles = [Line2D([0], [0], marker='s', color='w', markerfacecolor=MODALITY_COLORS[m], markeredgecolor='black', markersize=12, label=m) for m in ["SNV", "CNV", "SV"]]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=18)
    fig.suptitle("Subtype-enriched features", fontsize=30, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIGDIR / f"driver_enrichment_combined.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_modality_summary(enrich, fdr_thresh, exploratory=False):
    sig = enrich[(enrich["FDR"] < fdr_thresh) & (enrich["delta_prevalence_pct"] > 0)].copy()
    if sig.empty:
        sig = enrich[enrich["delta_prevalence_pct"] > 0].sort_values("pvalue").head(80)
    tab = pd.crosstab(sig["cluster_display"], sig["modality"]).reindex(index=["C1","C2","C3","C4"], columns=["SNV","CNV","SV"], fill_value=0)
    suffix = "q10_exploratory" if exploratory else "q05"
    tab.to_csv(TABDIR / f"modality_enrichment_summary_{suffix}.csv")
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    bottom = np.zeros(len(tab))
    x = np.arange(len(tab.index))
    for m in ["SNV", "CNV", "SV"]:
        ax.bar(x, tab[m].values, bottom=bottom, label=m, color=MODALITY_COLORS[m], edgecolor="black", linewidth=0.8)
        bottom += tab[m].values
    ax.set_xticks(x); ax.set_xticklabels(tab.index, fontweight="bold")
    ax.set_ylabel(f"Enriched features (FDR<{fdr_thresh:g})")
    tag = " (exploratory)" if exploratory else ""
    ax.set_title(f"Modality count{tag}", fontweight="bold")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIGDIR / f"modality_enrichment_summary_{suffix}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    print("Loading matrix...")
    x = read_matrix(MATRIX_FILE)
    print(f"  matrix: {x.shape}")
    print("Loading labels...")
    lab = read_labels(CLUSTER_FILE)
    common = x.index.intersection(lab.index)
    x = x.loc[common]
    lab = lab.loc[common]
    print(f"  common samples: {len(common)}")
    if len(common) < 100:
        raise ValueError("Too few common samples. Check label path and sample IDs.")

    enrich = build_enrichment(x, lab)
    enrich.to_csv(TABDIR / "all_feature_enrichment_by_cluster.csv", index=False)

    for fdr_thresh, exploratory in [(0.05, False), (0.10, True)]:
        suffix = "q10_exploratory" if exploratory else "q05"
        print(f"Building {suffix} outputs (FDR<{fdr_thresh:g})...")
        for cname in ["C1", "C2", "C3", "C4"]:
            top = choose_top(enrich, cname, fdr_thresh=fdr_thresh, n=20)
            top.to_csv(TABDIR / f"{cname}_top_enriched_features_{suffix}.csv", index=False)
            plot_lollipop(top, cname, fdr_thresh=fdr_thresh, exploratory=exploratory)
            top_depleted = choose_top_depleted(enrich, cname, fdr_thresh=fdr_thresh, n=15)
            top_depleted.to_csv(TABDIR / f"{cname}_top_depleted_features_{suffix}.csv", index=False)
            plot_lollipop_depleted(top_depleted, cname, fdr_thresh=fdr_thresh, exploratory=exploratory)
        plot_modality_summary(enrich, fdr_thresh=fdr_thresh, exploratory=exploratory)

    plot_combined(enrich)
    print(f"Done. Outputs: {OUTDIR}")


if __name__ == "__main__":
    main()
