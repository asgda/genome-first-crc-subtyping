#!/usr/bin/env python3
"""Test convergent CNV-loss/LOH and SV hits in WNT, TGF-beta and PI3K genes.

The prespecified aggregate endpoint requires both event classes within the
same pathway gene set. C4 is compared with C1 and with the pooled remaining
subtypes using Fisher exact tests and FDR correction.
"""
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, chi2_contingency, mannwhitneyu
from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch

BASE = os.environ.get("CRC_BASE", str(Path(__file__).resolve().parents[1]))
OUT_DIR = os.path.join(BASE, "module25_pathway_dual_hit_C1C4")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(os.path.join(OUT_DIR, "tables"), exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Locked project cluster palette (Okabe-Ito, colour-blind safe), matching
# Modules 12/17/20/21/22/23/24 -- NOT Module 7g's older, inconsistent mapping.
CLUSTER_COLORS = {"C1": "#E69F00", "C2": "#56B4E9", "C3": "#009E73", "C4": "#D55E00"}
CLUSTER_ORDER = ["C1", "C2", "C3", "C4"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 15,
    "axes.titlesize": 19,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "axes.linewidth": 1.1,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

WNT_GENES = ["APC", "CTNNB1", "AXIN1", "AXIN2", "RNF43", "TCF7L2", "KREMEN1", "DVL2", "DVL3",
             "WNT7B", "WNT5B", "LRP6", "FBXW7", "AMER1", "BCL9", "BCL9L", "PRKD1"]
TGFB_GENES = ["SMAD2", "SMAD3", "SMAD4", "TGFBR1", "TGFBR2", "ACVR2A", "ACVR1B", "TGFB1", "STRAP"]
PI3K_GENES = ["PIK3CA", "PIK3R1", "PIK3R2", "PIK3R5", "PTEN", "AKT1", "AKT2", "AKT3", "STK11",
              "TSC1", "TSC2", "TULP3"]
PATHWAYS = {"WNT": WNT_GENES, "TGFb": TGFB_GENES, "PI3K": PI3K_GENES}


def norm(x):
    m = re.search(r"(UM\d+|U\d+)", str(x))
    return m.group(1) if m else None


def main():
    lab = pd.read_csv(f"{BASE}/module05_06_loocv_results/labels/NMF_k4_LOOCV.csv")
    lab["sid"] = lab["sample_id"].astype(str).map(norm)
    lab = lab.dropna(subset=["sid"]).drop_duplicates("sid")
    s = pd.to_numeric(lab["cluster"])
    if s.min() == 1:
        s = s - 1
    lab["c0"] = s.astype(int)
    labels = lab.set_index("sid")["c0"].map({0: "C1", 1: "C2", 2: "C3", 3: "C4"})

    cnv_dir = pd.read_csv(f"{BASE}/module2_results/module2_discovery_directional_matrix.csv", index_col=0)
    cnv_dir.index = cnv_dir.index.map(norm)
    cnv_dir = cnv_dir.loc[~cnv_dir.index.isna() & ~cnv_dir.index.duplicated()]

    sv_bin = pd.read_csv(f"{BASE}/module3_results/module3_discovery_binary_matrix.csv", index_col=0)
    sv_bin.index = sv_bin.index.map(norm)
    sv_bin = sv_bin.loc[~sv_bin.index.isna() & ~sv_bin.index.duplicated()]

    common = labels.index.intersection(cnv_dir.index).intersection(sv_bin.index)
    labels_c = labels.loc[common]
    print(f"n={len(common)}")

    def has_cnv_loss(gene, idx):
        cols = [c for c in [f"CNV_{gene}_LOSS", f"CNV_{gene}_LOH", f"CNV_{gene}_HOMDEL"] if c in cnv_dir.columns]
        if not cols:
            return pd.Series(0, index=idx)
        return (cnv_dir.loc[idx, cols].sum(axis=1) > 0).astype(int)

    def has_sv(gene, idx):
        col = f"SV_{gene}"
        if col not in sv_bin.columns:
            return pd.Series(0, index=idx)
        return sv_bin.loc[idx, col].astype(int)

    dual_hit = pd.DataFrame(index=common)
    for pw, genes in PATHWAYS.items():
        cnv_any = pd.Series(0, index=common)
        sv_any = pd.Series(0, index=common)
        for g in genes:
            cnv_any = cnv_any | has_cnv_loss(g, common)
            sv_any = sv_any | has_sv(g, common)
        dual_hit[pw] = (cnv_any & sv_any).astype(int)

    dual_hit["n_pathways_dual_hit"] = dual_hit[list(PATHWAYS.keys())].sum(axis=1)
    dual_hit["any_dual_hit"] = (dual_hit["n_pathways_dual_hit"] > 0).astype(int)
    dual_hit["cluster"] = labels_c
    dual_hit.to_csv(os.path.join(OUT_DIR, "tables", "dual_hit_per_sample.csv"))

    rows = []

    def fisher_row(mask_a, mask_b, label_a, label_b, name):
        a = int(dual_hit.loc[mask_a, "any_dual_hit"].sum())
        na = int(mask_a.sum())
        b = int(dual_hit.loc[mask_b, "any_dual_hit"].sum())
        nb = int(mask_b.sum())
        odds, p = fisher_exact([[a, na - a], [b, nb - b]], alternative="greater")
        rows.append({"test": name, "group_a": label_a, "n_a": na, "hit_a": a, "pct_a": 100 * a / na,
                     "group_b": label_b, "n_b": nb, "hit_b": b, "pct_b": 100 * b / nb,
                     "odds_ratio": odds, "p_value": p})

    c4 = dual_hit["cluster"] == "C4"
    c1 = dual_hit["cluster"] == "C1"
    rest = dual_hit["cluster"] != "C4"
    fisher_row(c4, c1, "C4", "C1", "any_dual_hit: C4 vs C1")
    fisher_row(c4, rest, "C4", "rest", "any_dual_hit: C4 vs rest")

    ct = pd.crosstab(dual_hit["cluster"], dual_hit["any_dual_hit"])
    chi2, p_omni, _, _ = chi2_contingency(ct)
    rows.append({"test": "any_dual_hit: 4-way omnibus", "group_a": "C1-C4", "n_a": len(dual_hit),
                "hit_a": np.nan, "pct_a": np.nan, "group_b": "", "n_b": np.nan, "hit_b": np.nan,
                "pct_b": np.nan, "odds_ratio": np.nan, "p_value": p_omni})

    c4n = dual_hit.loc[c4, "n_pathways_dual_hit"]
    c1n = dual_hit.loc[c1, "n_pathways_dual_hit"]
    u, p_mw = mannwhitneyu(c4n, c1n, alternative="greater")
    delta = 2 * u / (len(c4n) * len(c1n)) - 1
    rows.append({"test": "n_pathways_dual_hit ordinal: C4 vs C1 (Cliff's delta)", "group_a": "C4",
                "n_a": len(c4n), "hit_a": float(c4n.mean()), "pct_a": delta, "group_b": "C1",
                "n_b": len(c1n), "hit_b": float(c1n.mean()), "pct_b": np.nan,
                "odds_ratio": np.nan, "p_value": p_mw})

    res = pd.DataFrame(rows)
    res["FDR"] = multipletests(res["p_value"], method="fdr_bh")[1]
    res.to_csv(os.path.join(OUT_DIR, "tables", "dual_hit_statistics.csv"), index=False)
    print(res.to_string(index=False))
    print(f"\nOutputs written to {OUT_DIR}/tables/")

    make_figures(dual_hit, res)
    print(f"Figures written to {FIG_DIR}/")


##############################################################################
# FIGURE
##############################################################################
def _sig_bracket(ax, x1, x2, y, h, text):
    """Draw a horizontal significance bracket between two bar x-positions."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.6, color="black")
    ax.text((x1 + x2) / 2, y + h * 1.15, text, ha="center", va="bottom",
             fontsize=13, fontweight="bold")


def make_figures(dual_hit, res):
    pathway_names = ["WNT", "TGFb", "PI3K"]
    display_names = {"WNT": "Wnt", "TGFb": "TGF-β", "PI3K": "PI3K"}
    groups = pathway_names + ["any_dual_hit"]
    group_labels = [display_names[p] for p in pathway_names] + ["Any\npathway"]

    # ---- data for Panel A: % dual-hit per pathway (+ aggregate) per cluster ----
    pct = pd.DataFrame(index=CLUSTER_ORDER, columns=groups, dtype=float)
    for cl in CLUSTER_ORDER:
        sub = dual_hit[dual_hit["cluster"] == cl]
        for g in groups:
            pct.loc[cl, g] = 100 * sub[g].mean()

    # ---- data for Panel B: ordinal distribution of n_pathways_dual_hit ----
    ordinal = pd.crosstab(dual_hit["cluster"], dual_hit["n_pathways_dual_hit"], normalize="index") * 100
    ordinal = ordinal.reindex(CLUSTER_ORDER).fillna(0.0)
    for k in [0, 1, 2, 3]:
        if k not in ordinal.columns:
            ordinal[k] = 0.0
    ordinal = ordinal[[0, 1, 2, 3]]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.69, 6.5))  # A4-landscape width

    # ===================== PANEL A =====================
    n_grp = len(groups)
    n_cl = len(CLUSTER_ORDER)
    bar_w = 0.19
    x = np.arange(n_grp)
    any_idx = groups.index("any_dual_hit")
    for i, cl in enumerate(CLUSTER_ORDER):
        offset = (i - (n_cl - 1) / 2) * bar_w
        vals = pct.loc[cl, groups].values
        axA.bar(x + offset, vals, width=bar_w * 0.94, color=CLUSTER_COLORS[cl],
               edgecolor="black", linewidth=0.7, label=cl, zorder=3)
        # Only the "Any pathway" group is individually labelled, to keep the
        # three per-pathway groups readable from bar height alone rather than
        # crowding every bar with a number.
        v_any = pct.loc[cl, "any_dual_hit"]
        axA.text(any_idx + offset, v_any + 1.0, f"{v_any:.0f}%", ha="center", va="bottom",
                 fontsize=12, fontweight="bold")

    axA.set_xticks(x)
    axA.set_xticklabels(group_labels, fontweight="bold")
    axA.set_ylabel("Tumours with a convergent\nCNV-loss + SV double hit (%)")
    axA.set_title("A", loc="left", fontweight="bold", fontsize=22)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)
    axA.set_ylim(0, max(pct.values.max() * 1.001, 1) + 13)
    axA.grid(axis="y", color="#DDDDDD", lw=0.7, zorder=0)
    axA.set_axisbelow(True)

    # significance bracket: C4 vs C1 on the "Any pathway" group. Placed well
    # above the tallest bar and its %-label so it never overlaps the legend,
    # which sits below the panel (matching Panel B's layout) rather than
    # inside the plotting area.
    c4_x = any_idx + (CLUSTER_ORDER.index("C4") - (n_cl - 1) / 2) * bar_w
    c1_x = any_idx + (CLUSTER_ORDER.index("C1") - (n_cl - 1) / 2) * bar_w
    y_top = max(pct.loc["C4", "any_dual_hit"], pct.loc["C1", "any_dual_hit"])
    fdr_c4_c1 = res.loc[res["test"] == "any_dual_hit: C4 vs C1", "FDR"].values[0]
    _sig_bracket(axA, c1_x, c4_x, y_top + 6.0, 2.4, f"FDR={fdr_c4_c1:.3f}")

    # C4-vs-pooled-rest statistic reported as an annotation, since "rest" is
    # not a single bar and a bracket to it would be visually ambiguous.
    fdr_c4_rest = res.loc[res["test"] == "any_dual_hit: C4 vs rest", "FDR"].values[0]
    or_c4_rest = res.loc[res["test"] == "any_dual_hit: C4 vs rest", "odds_ratio"].values[0]
    axA.text(0.015, 0.985,
             f"C4 vs pooled rest (Any pathway):\nOR={or_c4_rest:.1f}, FDR={fdr_c4_rest:.1e}",
             transform=axA.transAxes, ha="left", va="top", fontsize=12,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#F7F7F7", edgecolor="#999999"))

    handles = [Patch(facecolor=CLUSTER_COLORS[cl], edgecolor="black", label=cl) for cl in CLUSTER_ORDER]
    axA.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4,
               frameon=False, title="Subtype", title_fontsize=13, columnspacing=1.3,
               handletextpad=0.5)

    # ===================== PANEL B =====================
    seq_colors = ["#F0F0F0", "#FDBE85", "#E6550D", "#7F0000"]  # 0,1,2,3 pathways, light->dark
    bottom = np.zeros(len(CLUSTER_ORDER))
    xB = np.arange(len(CLUSTER_ORDER))
    for k, col in zip([0, 1, 2, 3], seq_colors):
        vals = ordinal[k].values
        axB.bar(xB, vals, bottom=bottom, width=0.62, color=col, edgecolor="black",
               linewidth=0.7, label=f"{k}", zorder=3)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 4:
                txt_color = "white" if k >= 2 else "black"
                axB.text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                         fontsize=11, color=txt_color, fontweight="bold")
        bottom += vals

    axB.set_xticks(xB)
    axB.set_xticklabels(CLUSTER_ORDER, fontweight="bold")
    for tick, cl in zip(axB.get_xticklabels(), CLUSTER_ORDER):
        tick.set_color(CLUSTER_COLORS[cl])
    axB.set_ylabel("Tumours (%)")
    axB.set_ylim(0, 100)
    axB.set_title("B", loc="left", fontweight="bold", fontsize=22)
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)
    axB.legend(title="Pathways\ndual-hit", loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=4, frameon=False, title_fontsize=12, handletextpad=0.5, columnspacing=1.1)

    fig.suptitle("Pathway-level convergent CNV-loss + structural-variant double hits",
                 fontsize=19, fontweight="bold", y=1.03)
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])

    base = os.path.join(FIG_DIR, "dual_hit_summary")
    fig.savefig(f"{base}.jpg", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
