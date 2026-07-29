#!/usr/bin/env python3
"""Compare aggregate genomic burdens across the locked CRC subtypes.

The script reports Cliff's delta, percentile-bootstrap 95% confidence
intervals and Benjamini-Hochberg adjusted P values for C4 versus the pooled
remaining subtypes and for the prespecified C4-versus-C1 contrast. The minimal
classifier analysis present in the development script is intentionally omitted
because it is not part of the manuscript.
"""

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

##############################################################################
# CONFIG
##############################################################################
BASE = os.environ.get("CRC_BASE", str(Path(__file__).resolve().parents[1]))
M1_BURDEN = os.environ.get("CRC_M1_BURDEN", f"{BASE}/module1_results/module1_snv_burden_matrix.csv")
M2_DIR    = os.environ.get("CRC_M2_DIRECTIONAL", f"{BASE}/module2_results/module2_discovery_directional_matrix.csv")
M2_ARMCIN = os.environ.get("CRC_M2_ARMCIN", f"{BASE}/module2_results/module2_arm_cin_matrix.csv")
M3_BURDEN = os.environ.get("CRC_M3_BURDEN", f"{BASE}/module3_results/module3_sv_burden_matrix.csv")
M3_BIN    = os.environ.get("CRC_M3_BIN", f"{BASE}/module3_results/module3_discovery_binary_matrix.csv")
M3_ARCH   = os.environ.get("CRC_M3_ARCH", f"{BASE}/module3_results/module3_sv_architecture_matrix.csv")
CLUSTER_FILE = os.environ.get("CRC_CLUSTER_FILE", f"{BASE}/module05_06_loocv_results/labels/NMF_k4_LOOCV.csv")

OUTDIR = os.environ.get("CRC_M24_OUT", f"{BASE}/module24_subtype_burden_panel_C1C4_ppt")
FIGDIR = os.path.join(OUTDIR, "figures")
TABDIR = os.path.join(OUTDIR, "tables")
for d in (OUTDIR, FIGDIR, TABDIR):
    os.makedirs(d, exist_ok=True)

N_BOOTSTRAP = int(os.environ.get("CRC_M24_NBOOT", "2000"))
RANDOM_STATE = int(os.environ.get("CRC_M24_SEED", "42"))
C4_INTERNAL = 3

CLUSTER_COLORS = {"C1": "#E69F00", "C2": "#56B4E9", "C3": "#009E73", "C4": "#D55E00"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 16, "axes.titlesize": 22,
    "axes.labelsize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15,
    "legend.fontsize": 14, "savefig.dpi": 300, "axes.linewidth": 1.1,
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
    lab["sid"] = lab["sample_id"].map(normalize_sample)
    lab["cluster"] = normalize_cluster_values(lab["cluster"])
    lab = lab.dropna(subset=["sid"]).drop_duplicates("sid")
    return lab.set_index("sid")["cluster"]


def _read_idx(path):
    d = pd.read_csv(path, index_col=0)
    d.index = d.index.map(normalize_sample)
    return d.loc[~d.index.isna() & ~d.index.duplicated()]


def savefig(base):
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(f"{base}.{ext}", bbox_inches="tight")
    plt.close()
    print(f"  -> {base}.png/pdf")


##############################################################################
# PART A — BURDEN EFFECT SIZES
##############################################################################
def cliffs_delta_from_U(x, y):
    """Cliff's delta of x vs y via Mann-Whitney U (delta>0 => x tends higher).
    delta = 2*U1/(n1*n2) - 1, where U1 counts (x>y) pairs (+0.5 ties)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan
    try:
        U1, p = mannwhitneyu(x, y, alternative="two-sided")
    except ValueError:  # all identical
        return 0.0, 1.0
    delta = 2.0 * U1 / (n1 * n2) - 1.0
    return float(delta), float(p)


def bootstrap_cliffs_ci(x, y, n_boot, rng):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    y = np.asarray(y, float); y = y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    if n1 < 3 or n2 < 3:
        return np.nan, np.nan
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        xb = x[rng.integers(0, n1, n1)]
        yb = y[rng.integers(0, n2, n2)]
        d, _ = cliffs_delta_from_U(xb, yb)
        deltas[b] = d
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def build_burden_features(labels):
    """Assemble the per-sample burden metric table (samples x metrics)."""
    feats = {}

    snv = _read_idx(M1_BURDEN)
    feats["SNV_TOTAL_BURDEN"] = snv.iloc[:, 0]

    svb = _read_idx(M3_BURDEN)
    feats["SV_TOTAL"] = svb.iloc[:, 0]

    svbin = _read_idx(M3_BIN)
    feats["SV_GENE_EVENT_FRACTION"] = svbin.mean(axis=1)

    arch = _read_idx(M3_ARCH)
    degenerate = [c for c in arch.columns if arch[c].nunique() <= 1]
    if degenerate:
        print(f"  [warn] excluding constant architecture columns: {degenerate}")
    for c in ["SVCLASS_deletion", "SVCLASS_inversion",
              "SVCLASS_translocation", "SVCLASS_tandem-duplication"]:
        if c in arch.columns and arch[c].nunique() > 1:
            feats[c] = arch[c]

    arm_cin = _read_idx(M2_ARMCIN)
    feats["ARM_ALTERED_FRACTION"] = arm_cin.mean(axis=1)  # genome-wide CIN

    dire = _read_idx(M2_DIR)
    arm_re = re.compile(r"^CNV_(?:[0-9]{1,2}|X|Y)[pq]_(GAIN|LOSS)$")
    arm_loss = [c for c in dire.columns if arm_re.match(c) and c.endswith("LOSS")]
    arm_gain = [c for c in dire.columns if arm_re.match(c) and c.endswith("GAIN")]
    gene_loss = [c for c in dire.columns if c.endswith("_LOSS") and not arm_re.match(c)]
    gene_gain = [c for c in dire.columns if c.endswith("_GAIN") and not arm_re.match(c)]
    gene_loh = [c for c in dire.columns if c.endswith("_LOH")]
    feats["ARM_LOSS_FRACTION"] = dire[arm_loss].mean(axis=1)
    feats["ARM_GAIN_FRACTION"] = dire[arm_gain].mean(axis=1)
    feats["FOCAL_LOSS_FRACTION"] = dire[gene_loss].mean(axis=1)
    feats["FOCAL_GAIN_FRACTION"] = dire[gene_gain].mean(axis=1)
    if gene_loh:
        feats["FOCAL_LOH_FRACTION"] = dire[gene_loh].mean(axis=1)

    fdf = pd.DataFrame(feats)
    fdf = fdf.loc[fdf.index.intersection(labels.index)]
    fdf["cluster"] = labels.reindex(fdf.index)
    fdf = fdf.dropna(subset=["cluster"])
    print(f"  burden feature table: {fdf.shape[0]} samples x {fdf.shape[1]-1} metrics")
    return fdf


# Ordered, readable metric labels for the figure/table
METRIC_LABELS = {
    "ARM_LOSS_FRACTION": "Arm-level loss fraction",
    "FOCAL_LOH_FRACTION": "Focal LOH fraction",
    "FOCAL_LOSS_FRACTION": "Focal loss fraction",
    "ARM_ALTERED_FRACTION": "Genome-wide CIN (arm-altered fraction)",
    "SV_GENE_EVENT_FRACTION": "SV gene-event fraction",
    "SV_TOTAL": "Total structural variants",
    "SVCLASS_translocation": "SV: translocations",
    "SVCLASS_inversion": "SV: inversions",
    "SVCLASS_deletion": "SV: deletions",
    "SVCLASS_tandem-duplication": "SV: tandem duplications",
    "ARM_GAIN_FRACTION": "Arm-level gain fraction",
    "FOCAL_GAIN_FRACTION": "Focal gain fraction",
    "SNV_TOTAL_BURDEN": "Total SNV burden",
}


def run_part_a(labels):
    print("\n" + "=" * 70 + "\nPART A — BURDEN EFFECT SIZES (C4 vs rest)\n" + "=" * 70)
    fdf = build_burden_features(labels)
    rng = np.random.default_rng(RANDOM_STATE)
    metrics = [c for c in fdf.columns if c != "cluster"]
    is_c4 = (fdf["cluster"] == C4_INTERNAL).values

    rows = []
    for m in metrics:
        vals = pd.to_numeric(fdf[m], errors="coerce").values
        x = vals[is_c4]        # C4
        y = vals[~is_c4]       # rest
        delta, p = cliffs_delta_from_U(x, y)
        lo, hi = bootstrap_cliffs_ci(x, y, N_BOOTSTRAP, rng)
        rows.append({
            "metric": m,
            "label": METRIC_LABELS.get(m, m),
            "C4_median": float(np.nanmedian(x)),
            "rest_median": float(np.nanmedian(y)),
            "cliffs_delta": delta,
            "delta_CI_low": lo, "delta_CI_high": hi,
            "direction": "C4 higher" if delta > 0 else "C4 lower",
            "mannwhitney_p": p,
            "n_C4": int(np.sum(~np.isnan(x))), "n_rest": int(np.sum(~np.isnan(y))),
        })
    res = pd.DataFrame(rows)
    res["FDR"] = multipletests(res["mannwhitney_p"], method="fdr_bh")[1]
    res["significant_FDR"] = res["FDR"] < 0.05
    res = res.sort_values("cliffs_delta", ascending=False).reset_index(drop=True)

    out_csv = os.path.join(TABDIR, "burden_effectsizes_C4_vs_rest.csv")
    res.to_csv(out_csv, index=False)
    print(res[["label", "C4_median", "rest_median", "cliffs_delta",
               "delta_CI_low", "delta_CI_high", "FDR", "significant_FDR"]].to_string(index=False))

    # Forest plot of Cliff's delta with bootstrap CI
    plot_df = res.sort_values("cliffs_delta")
    fig, ax = plt.subplots(figsize=(13.5, max(7, 0.68 * len(plot_df) + 2.0)))
    y = np.arange(len(plot_df))
    for yi, (_, r) in zip(y, plot_df.iterrows()):
        d, lo, hi = r["cliffs_delta"], r["delta_CI_low"], r["delta_CI_high"]
        color = "#D55E00" if d > 0 else "#0072B2"
        star = "*" if r["significant_FDR"] else ""
        xerr = [[max(0, d - lo)], [max(0, hi - d)]] if np.isfinite(lo) else None
        ax.errorbar(d, yi, xerr=xerr, fmt="o", color=color, ecolor=color,
                    elinewidth=2.2, capsize=4, markersize=9, markeredgecolor="black")
        ax.text(1.02, yi, f"{d:+.2f}{star}", va="center", fontsize=17,
                transform=ax.get_yaxis_transform())
    ax.axvline(0, color="grey", ls="--", lw=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"].tolist(), fontsize=18)
    ax.set_xlim(-1.05, 1.05)
    ax.tick_params(axis="x", labelsize=18)
    ax.set_xlabel("Cliff's delta (C4 vs rest)  ·  bootstrap 95% CI", fontsize=22)
    ax.set_title("Burden", fontweight="bold", fontsize=26)
    ax.text(0.99, -0.11, "* FDR<0.05    right = higher in C4",
            transform=ax.transAxes, ha="right", va="top", fontsize=17)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    savefig(os.path.join(FIGDIR, "burden_effectsize_forest"))
    return res


def run_part_a2_c4_vs_c1(labels):
    """C4-vs-C1 burden effect sizes (added 2026-07-24; see module docstring).
    Identical methodology to run_part_a() -- same features, same Cliff's
    delta/bootstrap/BH-FDR -- restricted to the C4-vs-C1 subgroup instead of
    C4-vs-pooled-rest, because that is the specific contrast cited in the
    manuscript/thesis Results text."""
    print("\n" + "=" * 70 + "\nPART A2 — BURDEN EFFECT SIZES (C4 vs C1)\n" + "=" * 70)
    fdf = build_burden_features(labels)
    rng = np.random.default_rng(RANDOM_STATE)
    metrics = [c for c in fdf.columns if c != "cluster"]
    is_c4 = (fdf["cluster"] == C4_INTERNAL).values
    is_c1 = (fdf["cluster"] == 0).values

    rows = []
    for m in metrics:
        vals = pd.to_numeric(fdf[m], errors="coerce").values
        x = vals[is_c4]   # C4
        y = vals[is_c1]   # C1 only, NOT pooled rest
        delta, p = cliffs_delta_from_U(x, y)
        lo, hi = bootstrap_cliffs_ci(x, y, N_BOOTSTRAP, rng)
        rows.append({
            "metric": m, "label": METRIC_LABELS.get(m, m),
            "C4_median": float(np.nanmedian(x)), "C1_median": float(np.nanmedian(y)),
            "cliffs_delta": delta, "delta_CI_low": lo, "delta_CI_high": hi,
            "direction": "C4 higher" if delta > 0 else "C4 lower",
            "mannwhitney_p": p,
            "n_C4": int(np.sum(~np.isnan(x))), "n_C1": int(np.sum(~np.isnan(y))),
        })
    res = pd.DataFrame(rows)
    res["FDR"] = multipletests(res["mannwhitney_p"], method="fdr_bh")[1]
    res["significant_FDR"] = res["FDR"] < 0.05
    res = res.sort_values("cliffs_delta", ascending=False).reset_index(drop=True)

    out_csv = os.path.join(TABDIR, "burden_effectsizes_C4_vs_C1.csv")
    res.to_csv(out_csv, index=False)
    print(res[["label", "C4_median", "C1_median", "cliffs_delta",
               "delta_CI_low", "delta_CI_high", "FDR", "significant_FDR"]].to_string(index=False))

    plot_df = res.sort_values("cliffs_delta")
    fig, ax = plt.subplots(figsize=(13.5, max(7, 0.68 * len(plot_df) + 2.0)))
    y = np.arange(len(plot_df))
    for yi, (_, r) in zip(y, plot_df.iterrows()):
        d, lo, hi = r["cliffs_delta"], r["delta_CI_low"], r["delta_CI_high"]
        color = "#D55E00" if d > 0 else "#0072B2"
        star = "*" if r["significant_FDR"] else ""
        xerr = [[max(0, d - lo)], [max(0, hi - d)]] if np.isfinite(lo) else None
        ax.errorbar(d, yi, xerr=xerr, fmt="o", color=color, ecolor=color,
                    elinewidth=2.2, capsize=4, markersize=9, markeredgecolor="black")
        ax.text(1.02, yi, f"{d:+.2f}{star}", va="center", fontsize=17,
                transform=ax.get_yaxis_transform())
    ax.axvline(0, color="grey", ls="--", lw=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"].tolist(), fontsize=18)
    ax.set_xlim(-1.05, 1.05)
    ax.tick_params(axis="x", labelsize=18)
    ax.set_xlabel("Cliff's delta (C4 vs C1)  ·  bootstrap 95% CI", fontsize=22)
    ax.set_title("Burden (C4 vs C1)", fontweight="bold", fontsize=26)
    ax.text(0.99, -0.11, "* FDR<0.05    right = higher in C4",
            transform=ax.transAxes, ha="right", va="top", fontsize=17)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    savefig(os.path.join(FIGDIR, "burden_effectsize_forest_C4_vs_C1"))
    return res



##############################################################################
# MAIN
##############################################################################
def main():
    print("=" * 70)
    print("GENOMIC BURDEN COMPARISONS")
    print("=" * 70)
    labels = load_labels()
    run_part_a(labels)
    run_part_a2_c4_vs_c1(labels)
    print(f"Outputs: {OUTDIR}")


if __name__ == "__main__":
    main()
