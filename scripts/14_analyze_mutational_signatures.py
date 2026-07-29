#!/usr/bin/env python3
"""Analyze SBS, DBS and indel mutational-signature activities by subtype.

Raw mutation counts are summarized separately from within-class relative
contributions. Reference COSMIC signatures, aetiological groups and de novo
signatures are compared using nonparametric tests and FDR correction.
"""

import os, re, warnings, zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection


##############################################################################
# ZERO-DEPENDENCY XLSX READER
##############################################################################
# openpyxl is NOT available in vep_env (conda can't resolve it for Python
# 3.10 on this system), and pip install is blocked. Since .xlsx is just a
# ZIP of XML files, we read it directly with Python's built-in zipfile +
# xml.etree.ElementTree — no third-party dependency needed.
def _xlsx_to_dataframe(path, sheet="sheet1"):
    """Read a single sheet from an .xlsx file using only stdlib modules.
    Returns a DataFrame with no header (row 0 = first spreadsheet row)."""
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    with zipfile.ZipFile(path) as z:
        with z.open("xl/sharedStrings.xml") as f:
            ss_root = ET.parse(f).getroot()
        shared = [
            "".join((t.text or "") for t in si.iter(f"{{{ns}}}t"))
            for si in ss_root.findall(f"{{{ns}}}si")
        ]
        sheet_path = f"xl/worksheets/{sheet}.xml"
        if sheet_path not in z.namelist():
            # fall back to first worksheet
            sheet_path = next(n for n in z.namelist() if "worksheets/sheet" in n)
        with z.open(sheet_path) as f:
            ws_root = ET.parse(f).getroot()

    def col_idx(col_str):
        n = 0
        for ch in col_str.upper():
            n = n * 26 + (ord(ch) - ord("A") + 1)
        return n - 1

    parsed, max_col = {}, 0
    for row_el in ws_root.findall(f".//{{{ns}}}row"):
        r = int(row_el.attrib["r"]) - 1
        for c_el in row_el:
            ref = c_el.attrib["r"]
            col_str = re.match(r"([A-Za-z]+)", ref).group(1)
            ci = col_idx(col_str)
            max_col = max(max_col, ci)
            t = c_el.attrib.get("t", "")
            v_el = c_el.find(f"{{{ns}}}v")
            if v_el is None:
                continue
            if t == "s":
                val = shared[int(v_el.text)]
                if isinstance(val, bytes):
                    val = val.decode("utf-8", errors="replace")
            elif t == "b":
                val = bool(int(v_el.text))
            else:
                try:
                    val = float(v_el.text)
                except (ValueError, TypeError):
                    val = v_el.text
            parsed[(r, ci)] = val

    if not parsed:
        return pd.DataFrame()
    max_row = max(r for r, _ in parsed)
    data = [[parsed.get((r, c), np.nan) for c in range(max_col + 1)]
            for r in range(max_row + 1)]
    return pd.DataFrame(data)

warnings.filterwarnings("ignore")

##############################################################################
# STYLE — publication-grade, large fonts, colour-blind-safe palette
##############################################################################
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 14,
    "axes.titlesize": 18, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 13, "savefig.dpi": 300, "axes.linewidth": 0.8,
})

# Okabe & Ito (2008) colour-blind-safe qualitative palette (used for
# aetiology-category colouring in Parts B-F; NOT used directly for
# cluster identity -- see CLUSTER_COLORS below).
CLUSTER_PALETTE = ["#E69F00","#56B4E9","#009E73","#F0E442",
                   "#0072B2","#D55E00","#CC79A7","#999999"]
DIVERGING_CMAP = "RdBu_r"  # red-blue (not red-green) -- colour-blind safe

# Dedicated per-cluster identity colours, locked to the project convention:
# C1=orange, C2=sky-blue, C3=bluish-green, C4=vermillion (high-risk).
# Keyed by the DISPLAY label so it can never silently drift out of sync
# with display_cluster() below, regardless of internal 0-based ordering.
CLUSTER_COLORS = {"C1": "#E69F00", "C2": "#56B4E9", "C3": "#009E73", "C4": "#D55E00"}

##############################################################################
# CONFIGURATION
##############################################################################
BASE = os.environ.get("CRC_BASE", str(Path(__file__).resolve().parents[1]))
DATA_DIR = f"{BASE}/crc_heterogeneity_data"

CLUSTER_FILE = os.environ.get(
    "CRC_CLUSTER_FILE",
    f"{BASE}/module05_06_loocv_results/labels/NMF_k4_LOOCV.csv",
)
# All display/output labels use the project convention: NMF only, no
# Brunet/OOF wording in plots, folders, or filenames; clusters shown as C1-C4.
CLUSTER_NAME = "NMF_k4_LOOCV"

SBS_FILE = f"{DATA_DIR}/SBS_signatures.xlsx"
OUTDIR   = f"{BASE}/module11_mutational_signatures/{CLUSTER_NAME}_C1C4_ppt"
os.makedirs(OUTDIR, exist_ok=True)

QC_METRICS = ["Total Mutations","Cosine Similarity","L1 Norm","L1 Norm %",
               "L2 Norm","L2 Norm %","KL Divergence","Correlation"]

TOP_REFERENCE = 20   # top differential reference signatures per class to plot
TOP_DENOVO    = 15   # top differential de novo signatures per class to plot
TOP_PER_CLUSTER = 10 # signatures shown per side in diverging lollipop

# Part C2 (aetiology effect sizes) -- same convention as Module 24 Part A
N_BOOTSTRAP  = int(os.environ.get("CRC_M11_NBOOT", "2000"))
RANDOM_STATE = int(os.environ.get("CRC_M11_SEED", "42"))
C4_INTERNAL  = 3

##############################################################################
# SIGNATURE AETIOLOGY ANNOTATIONS  (COSMIC v3 / Alexandrov et al. 2020)
##############################################################################
# Maps individual reference signatures to a broad mutational-process
# category. Anything not listed (de novo SBS96*/DBS78*/ID83* and *-CRC*
# signatures) is categorised separately by `categorize_signature()`.
SIGNATURE_ETIOLOGY = {
    # Clock-like (age-related, 5-methylcytosine deamination)
    "SBS1":"Clock-like (age-related)", "SBS5":"Clock-like (age-related)",
    # APOBEC cytidine deaminase activity
    "SBS2":"APOBEC", "SBS13":"APOBEC",
    # Homologous recombination deficiency
    "SBS3":"HRD (homologous recombination deficiency)",
    "ID6":"HRD (homologous recombination deficiency)",
    # UV light exposure
    "SBS7b":"UV exposure", "SBS7c":"UV exposure",
    # POLE exonuclease-domain proofreading deficiency
    "SBS10a":"POLE proofreading deficiency",
    "SBS10b":"POLE proofreading deficiency",
    "SBS28":"POLE proofreading deficiency",
    # Mismatch-repair deficiency / MSI
    "SBS6":"MMR deficiency / MSI", "SBS14":"MMR deficiency / MSI",
    "SBS15":"MMR deficiency / MSI", "SBS20":"MMR deficiency / MSI",
    "SBS21":"MMR deficiency / MSI", "SBS26":"MMR deficiency / MSI",
    "SBS44":"MMR deficiency / MSI", "DBS7":"MMR deficiency / MSI",
    "ID1":"MMR deficiency / MSI", "ID2":"MMR deficiency / MSI",
    "ID7":"MMR deficiency / MSI",
    # Reactive-oxygen-species / base-excision-repair deficiency
    "SBS18":"Oxidative damage / BER deficiency",
    "SBS30":"Oxidative damage / BER deficiency",
    # Treatment / chemotherapy exposure
    "SBS17a":"Treatment-associated", "SBS17b":"Treatment-associated",
    "SBS32":"Treatment-associated", "SBS86":"Treatment-associated",
    "SBS90":"Treatment-associated", "DBS5":"Treatment-associated",
    # Colibactin (pks+ E. coli) -- characteristic of CRC
    "SBS88":"Colibactin (pks+ E. coli)", "ID18":"Colibactin (pks+ E. coli)",
    # Double-strand-break repair / NHEJ
    "ID8":"DSB repair / NHEJ", "DBS8":"DSB repair / NHEJ",
    # Occupational / environmental exposure
    "SBS42":"Occupational exposure (haloalkanes)",
}
# Everything else among reference signatures: "Unknown aetiology"
ETIOLOGY_ORDER = [
    "Clock-like (age-related)", "APOBEC", "MMR deficiency / MSI",
    "POLE proofreading deficiency", "HRD (homologous recombination deficiency)",
    "Oxidative damage / BER deficiency", "Colibactin (pks+ E. coli)",
    "DSB repair / NHEJ", "Treatment-associated",
    "Occupational exposure (haloalkanes)", "Unknown aetiology",
    "CRC-specific (study-derived)", "De novo (unassigned)",
]
# 13-colour colour-blind-safe qualitative palette: Okabe & Ito (2008) 8
# colours extended with 5 additional CVD-distinguishable colours (Paul Tol
# muted-scheme additions) so that no two of the 13 aetiology categories
# share a colour -- the Okabe-Ito palette alone (8 colours) was found to
# wrap around and collide "MMR deficiency / MSI" with "Unknown aetiology".
ETIOLOGY_PALETTE = CLUSTER_PALETTE + ["#882255","#44AA99","#117733","#AA4499","#332288"]
ETIOLOGY_COLORS = dict(zip(ETIOLOGY_ORDER, ETIOLOGY_PALETTE[:len(ETIOLOGY_ORDER)]))

def signature_class(name):
    """SBS / DBS / ID, based on name prefix."""
    if name.startswith("SBS"): return "SBS"
    if name.startswith("DBS"): return "DBS"
    if name.startswith("ID"):  return "ID"
    return "Other"

def is_denovo(name):
    """De novo SigProfiler signatures: SBS96A.., DBS78A.., ID83A.. """
    return bool(re.match(r"^(SBS96|DBS78|ID83)[A-Z]+$", str(name)))

def is_crc_specific(name):
    return "-CRC" in str(name)

def categorize_signature(name):
    """Aetiology category for a REFERENCE signature (not de novo / CRC)."""
    name = str(name)
    if name in SIGNATURE_ETIOLOGY:
        return SIGNATURE_ETIOLOGY[name]
    if is_crc_specific(name):
        return "CRC-specific (study-derived)"
    if is_denovo(name):
        return "De novo (unassigned)"
    return "Unknown aetiology"

##############################################################################
# SHARED HELPERS  (same conventions as Module 9, self-contained here)
##############################################################################
def normalize_sample(x):
    m = re.search(r"(UM\d+|U\d+)", str(x))
    return m.group(1) if m else np.nan

def normalize_cluster_values(series):
    """Normalize cluster labels to a 0-based internal representation
    regardless of whether the source CSV is 0- or 1-indexed. Same logic
    as Modules 16/17/19/20/21 for consistency across the pipeline."""
    s = pd.to_numeric(series, errors="raise").astype(int)
    vals = sorted(s.dropna().unique().tolist())
    if vals and min(vals) == 1 and max(vals) <= 8 and 0 not in vals:
        s = s - 1
    return s

def display_cluster(c):
    """Hard-coded display convention: internal 0-3 -> C1-C4."""
    return f"C{int(c) + 1}"

def display_cluster_list(cols):
    return [display_cluster(c) for c in cols]

def load_clusters(path=CLUSTER_FILE):
    df = pd.read_csv(path)
    if not {"sample_id", "cluster"}.issubset(df.columns):
        raise ValueError("Cluster CSV must contain sample_id and cluster columns.")
    df["sample_id"] = df["sample_id"].astype(str).apply(normalize_sample)
    df = df.dropna(subset=["sample_id"])
    df["cluster"] = normalize_cluster_values(df["cluster"])
    # Restrict to exactly the two columns needed. NMF_k4_LOOCV.csv also
    # carries cluster_display/cluster_0based; left unrestricted these would
    # ride through every downstream cluster_df.merge(...) and get
    # miscounted as feature columns in Parts A/C/D.
    return df[["sample_id", "cluster"]]

def significance_stars(p):
    if pd.isna(p): return ""
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 0.05: return "*"
    return "ns"

def feature_statistics(merged, features):
    results = []
    clusters = sorted(merged["cluster"].unique())
    for feat in features:
        groups, ok = [], True
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

def cliffs_delta_from_U(x, y):
    """Cliff's delta of x vs y via Mann-Whitney U (delta>0 => x tends higher).
    delta = 2*U1/(n1*n2) - 1, where U1 counts (x>y) pairs (+0.5 ties).
    Identical definition to Module 24 Part A (Cliff 1993, Psychol. Bull.),
    reused here for cross-script statistical consistency."""
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

def cluster_zscore_matrix(merged, features):
    mat = merged.groupby("cluster")[features].mean().T
    z = mat.sub(mat.mean(axis=1), axis=0).div(
        mat.std(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return mat, z

def annotated_heatmap(zmat, stats_df, feat_col, title, cbar_label, outpath,
                       side_labels=None, side_colors=None, side_title=None,
                       cmap=DIVERGING_CMAP, figsize=None):
    stats_map = dict(zip(stats_df[feat_col], stats_df["FDR"]))
    row_labels = [f"{f}  {significance_stars(stats_map.get(f, np.nan))}".rstrip()
                  for f in zmat.index]
    n_rows, n_cols = zmat.shape
    figsize = figsize or (1.3*n_cols + (6 if side_labels is not None else 4),
                          max(6, 0.34*n_rows + 1.5))
    if side_labels is not None:
        fig, (ax_side, ax) = plt.subplots(
            1, 2, figsize=figsize,
            gridspec_kw={"width_ratios":[0.55, n_cols]})
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_side = None

    vmax = max(np.abs(zmat.values).max(), 1e-6)
    im = ax.imshow(zmat.values, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(n_rows)); ax.set_yticklabels(row_labels, fontsize=12)
    ax.set_xticks(range(n_cols)); ax.set_xticklabels(display_cluster_list(zmat.columns),
                                                       fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=17, fontweight="bold", pad=10)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5,n_cols,1), minor=True)
    ax.set_yticks(np.arange(-0.5,n_rows,1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label, fontsize=13, fontweight="bold")
    cbar.ax.tick_params(labelsize=12)

    if ax_side is not None:
        cats = list(side_labels)
        color_map = side_colors or {}
        prev, start = None, 0
        for i, c in enumerate(cats + [None]):
            if c != prev:
                if prev is not None:
                    n = i - start
                    ax_side.axhspan(start, i, color=color_map.get(prev, "#CCCCCC"))
                    if n >= 2:
                        ax_side.text(0.5, start+n/2, str(prev), rotation=90,
                                     ha="center", va="center", fontsize=9,
                                     fontweight="bold", color="white")
                    elif n == 1:
                        ax_side.text(-0.15, start+n/2, str(prev), rotation=0,
                                     ha="right", va="center", fontsize=8,
                                     color="#333333", clip_on=False)
                start, prev = i, c
        ax_side.set_ylim(ax.get_ylim())
        ax_side.set_xlim(0,1); ax_side.axis("off")
        # NOTE: side_title intentionally not rendered as ax_side.set_title()
        # -- it visually overlaps the main heatmap title. The category
        # names are already shown as row annotations within the side-bar.

    plt.tight_layout()
    for ext in ["png","pdf"]:
        plt.savefig(f"{outpath}.{ext}", bbox_inches="tight")
    plt.close()
    print(f"  Saved {outpath}.png/pdf")

def dumbbell_plot(merged, stats_df, feat_col, title, xlabel, outpath, n_top=15):
    top = stats_df.sort_values(["FDR","effect_size"], ascending=[True,False]).head(n_top)
    feats = top[feat_col].tolist()[::-1]
    if not feats: return
    clusters = sorted(merged["cluster"].unique())
    means = merged.groupby("cluster")[feats].mean()
    fig, ax = plt.subplots(figsize=(9, 0.5*len(feats)+2.5))
    for i, f in enumerate(feats):
        vals = means.loc[clusters, f]
        ax.plot([vals.min(), vals.max()], [i,i], color="grey", lw=1.5, zorder=1)
        for c in clusters:
            ax.scatter(vals[c], i, color=CLUSTER_COLORS[display_cluster(c)],
                      s=90, edgecolor="black", linewidth=0.7, zorder=2,
                      label=display_cluster(c) if i==0 else None)
    ax.set_yticks(range(len(feats)))
    fdr_map = dict(zip(stats_df[feat_col], stats_df["FDR"]))
    ax.set_yticklabels([f"{f}  {significance_stars(fdr_map.get(f,np.nan))}" for f in feats], fontsize=12)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_title(title, fontsize=17, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0+1.6/len(feats)),
             ncol=len(clusters), framealpha=0.95, fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in ["png","pdf"]:
        plt.savefig(f"{outpath}.{ext}", bbox_inches="tight")
    plt.close()
    print(f"  Saved {outpath}.png/pdf")

def diverging_lollipop(zmat, stats_df, feat_col, cluster_id, title, outpath,
                       n_each=TOP_PER_CLUSTER, category_map=None,
                       xlabel="Cluster z-score"):
    col = zmat[cluster_id].sort_values()
    combo = pd.concat([col.head(n_each), col.tail(n_each)])
    combo = combo[~combo.index.duplicated()]
    fdr_map = dict(zip(stats_df[feat_col], stats_df["FDR"]))
    fig, ax = plt.subplots(figsize=(11, 0.42*len(combo)+2))
    y = np.arange(len(combo))[::-1]
    for yi, (feat, val) in zip(y, combo.items()):
        color = "#0072B2" if val < 0 else "#D55E00"  # Okabe-Ito blue/vermillion
        ax.plot([0, val], [yi,yi], color=color, lw=2.5, zorder=1)
        ax.scatter(val, yi, color=color, s=70, zorder=2, edgecolor="black", lw=0.6)
        star = significance_stars(fdr_map.get(feat, np.nan))
        label = feat if category_map is None else f"{feat}  [{category_map.get(feat,'Other')}]"
        ax.text(val + (0.05 if val>=0 else -0.05), yi, f"{label} {star}",
               ha="left" if val>=0 else "right", va="center", fontsize=11)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    xr = max(abs(combo.min()), abs(combo.max()), 1e-6)
    ax.set_xlim(-xr*2.6, xr*2.6)
    plt.tight_layout()
    for ext in ["png","pdf"]:
        plt.savefig(f"{outpath}.{ext}", bbox_inches="tight")
    plt.close()
    print(f"  Saved {outpath}.png/pdf")

def composition_bar(merged, cols, color_map, title, outpath, label_map=None):
    """Stacked bar of mean relative contributions per cluster (CIBERSORT-
    composition style), colour-blind-safe categorical colours."""
    means = merged.groupby("cluster")[cols].mean()
    means = means.loc[:, means.mean().sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(9,7))
    bottom = np.zeros(len(means))
    for c in means.columns:
        lbl = label_map.get(c, c) if label_map else c
        ax.bar(display_cluster_list(means.index), means[c].values, bottom=bottom,
              label=lbl, color=color_map.get(c, "#999999"),
              edgecolor="white", linewidth=0.6)
        bottom += means[c].values
    ax.set_ylabel("Mean relative contribution", fontsize=14)
    ax.set_title(title, fontsize=17, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5,-0.15),
             ncol=min(4,len(means.columns)), fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in ["png","pdf"]:
        plt.savefig(f"{outpath}.{ext}", bbox_inches="tight")
    plt.close()
    print(f"  Saved {outpath}.png/pdf")

def violin_panel(merged, features, stats_df, feat_col, outdir):
    import seaborn as sns
    order = sorted(merged["cluster"].unique())
    xlabels = display_cluster_list(order)
    palette = [CLUSTER_COLORS[lbl] for lbl in xlabels]
    for feat in features:
        plt.figure(figsize=(7,6))
        sns.violinplot(data=merged, x="cluster", y=feat, inner="box",
                       order=order, palette=palette)
        fdr = stats_df.loc[stats_df[feat_col]==feat,"FDR"]
        fdr_v = fdr.values[0] if len(fdr) else np.nan
        plt.title(f"{feat}\nKruskal-Wallis FDR={fdr_v:.3g}", fontsize=15, fontweight="bold")
        plt.xlabel("Cluster", fontsize=14)
        plt.ylabel(feat, fontsize=14)
        plt.gca().set_xticklabels(xlabels)
        plt.tight_layout()
        safe = re.sub(r"[^A-Za-z0-9_.-]","_", feat)
        for ext in ["png","pdf"]:
            plt.savefig(f"{outdir}/{safe}_violin.{ext}", bbox_inches="tight")
        plt.close()

##############################################################################
# LOADER
##############################################################################
def load_signature_blocks(path=SBS_FILE):
    """Returns dict {"SBS":df, "DBS":df, "ID":df}, each indexed by
    normalized sample_id, with QC columns prefixed by block name and
    signature columns left as-is (already globally unique)."""
    print(f"\nLoading mutational signatures: {os.path.basename(path)}")
    raw = _xlsx_to_dataframe(path)
    # Coerce any bytes values to str across all cells.
    # applymap was deprecated in pandas ≥2.1 (renamed to map); use a
    # lambda on each column to stay compatible across pandas versions.
    def _decode(v):
        return v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v
    for col in raw.columns:
        if raw[col].dtype == object:
            raw[col] = raw[col].apply(_decode)
    # Force all column-name candidates to str to prevent bytes leaking through
    colnames = [str(v) if not (isinstance(v, float) and np.isnan(v)) else np.nan
                for v in raw.iloc[2].tolist()]
    data = raw.iloc[3:].reset_index(drop=True)
    sample_ids = data.iloc[:,0].astype(str).apply(normalize_sample)

    starts = [i for i, v in enumerate(colnames)
              if isinstance(v, str) and v == "Total Mutations"]
    if len(starts) != 3:
        raise ValueError(f"Expected 3 'Total Mutations' block markers, found "
                         f"{len(starts)} -- input layout may have changed.")
    block_names = ["SBS","DBS","ID"]
    ends = starts[1:] + [len(colnames)]

    blocks = {}
    for name, s, e in zip(block_names, starts, ends):
        sub = data.iloc[:, s:e].copy()
        cols = []
        for c in colnames[s:e]:
            if isinstance(c, float) and np.isnan(c):
                cols.append(f"{name}_EMPTY_{len(cols)}")  # unnamed trailing column
            elif str(c) in QC_METRICS:
                cols.append(f"{name}_{c}")
            else:
                cols.append(str(c))
        sub.columns = cols
        sub = sub.apply(pd.to_numeric, errors="coerce")
        sub.index = sample_ids
        sub = sub.loc[~sub.index.isna()]
        sub = sub.loc[~sub.index.duplicated()]
        sub.index.name = "sample_id"
        blocks[name] = sub
        print(f"  {name}: {sub.shape[0]} samples x {sub.shape[1]} columns "
              f"({sub.shape[1]-8} signatures)")
    return blocks

def add_relative_contributions(block_df, block_name):
    """Adds REL__<signature> columns = count / Total_Mutations (NaN-safe)."""
    total_col = f"{block_name}_Total Mutations"
    sig_cols = [c for c in block_df.columns
                if c not in [f"{block_name}_{m}" for m in QC_METRICS]
                and not c.startswith(f"{block_name}_EMPTY_")]
    out = block_df.copy()
    total = out[total_col].replace(0, np.nan)
    for c in sig_cols:
        out[f"REL__{c}"] = out[c] / total
    return out, sig_cols

##############################################################################
# PART A — MUTATION BURDEN
##############################################################################
def run_burden_analysis(cluster_df, blocks, outdir):
    print("\n" + "="*70 + "\nPART A — MUTATION BURDEN BY SIGNATURE CLASS\n" + "="*70)
    a_dir = os.path.join(outdir, "A_Burden"); os.makedirs(a_dir, exist_ok=True)

    burden_cols = []
    merged = cluster_df.copy()
    for name, df in blocks.items():
        col = f"{name}_Total Mutations"
        merged = merged.merge(df[[col]], on="sample_id", how="inner")
        burden_cols.append(col)

    print(f"  Merged: {merged.shape}")
    stats = feature_statistics(merged, burden_cols)
    stats.to_csv(f"{a_dir}/burden_statistics.csv", index=False)
    pw = pairwise_tests(merged, burden_cols)
    if len(pw): pw.to_csv(f"{a_dir}/burden_pairwise.csv", index=False)
    violin_panel(merged, burden_cols, stats, "feature", a_dir)

    # combined burden heatmap (log10 scale, since counts span orders of magnitude)
    log_merged = merged.copy()
    log_cols = []
    for c in burden_cols:
        lc = f"log10_{c}"
        log_merged[lc] = np.log10(log_merged[c].clip(lower=1))
        log_cols.append(lc)
    log_stats = feature_statistics(log_merged, log_cols)
    _, zmat = cluster_zscore_matrix(log_merged, log_cols)
    annotated_heatmap(zmat, log_stats, "feature",
        f"Mutation burden by signature class ({CLUSTER_NAME})",
        "Cluster z-score (log10 total mutations)", f"{a_dir}/burden_heatmap")

    return merged, stats

##############################################################################
# PART B — REFERENCE SIGNATURE ACTIVITY (relative contributions)
##############################################################################
def run_reference_signature_analysis(cluster_df, blocks, outdir):
    print("\n" + "="*70 + "\nPART B — REFERENCE SIGNATURE ACTIVITY\n" + "="*70)
    b_dir = os.path.join(outdir, "B_ReferenceSignatures"); os.makedirs(b_dir, exist_ok=True)
    results = {}
    for name, df in blocks.items():
        rel, sig_cols = add_relative_contributions(df, name)
        ref_cols = [f"REL__{c}" for c in sig_cols
                    if not is_denovo(c) and not is_crc_specific(c)]
        merged = cluster_df.merge(rel[ref_cols], on="sample_id", how="inner")
        print(f"\n  {name}: merged {merged.shape}, "
              f"{len(ref_cols)} reference signatures")
        stats = feature_statistics(merged, ref_cols)
        stats.to_csv(f"{b_dir}/{name}_reference_statistics.csv", index=False)
        n_sig = (stats.FDR < 0.05).sum()
        print(f"    Significant (FDR<0.05): {n_sig}/{len(stats)}")

        sig = stats.loc[stats.FDR < 0.05]
        selected = (sig if len(sig) else stats).head(TOP_REFERENCE)["feature"].tolist()
        if not selected:
            print(f"    No reference signatures with data -- skipping {name}")
            continue
        _, zmat = cluster_zscore_matrix(merged, selected)
        # strip REL__ prefix for display
        zmat.index = [i.replace("REL__","") for i in zmat.index]
        disp_stats = stats.copy(); disp_stats["feature"] = disp_stats["feature"].str.replace("REL__","")
        cat_map = {i: categorize_signature(i) for i in zmat.index}
        cat_colors = {c: ETIOLOGY_COLORS.get(c,"#999999") for c in cat_map.values()}
        annotated_heatmap(zmat, disp_stats, "feature",
            f"{name} reference signature activity ({CLUSTER_NAME})",
            "Cluster z-score (relative contribution)",
            f"{b_dir}/{name}_reference_heatmap",
            side_labels=[cat_map[i] for i in zmat.index],
            side_colors=cat_colors, side_title="Aetiology")

        dumbbell_plot(merged.rename(columns={c:c.replace("REL__","") for c in merged.columns}),
            disp_stats, "feature",
            f"Top differential {name} reference signatures",
            "Mean relative contribution", f"{b_dir}/{name}_reference_dumbbell", n_top=15)

        results[name] = (merged, stats, ref_cols)
    return results

##############################################################################
# PART C — AETIOLOGY-GROUPED COMPOSITION
##############################################################################
def run_etiology_composition(cluster_df, blocks, outdir):
    print("\n" + "="*70 + "\nPART C — AETIOLOGY-GROUPED COMPOSITION\n" + "="*70)
    c_dir = os.path.join(outdir, "C_Etiology"); os.makedirs(c_dir, exist_ok=True)

    # build one merged table of all reference-signature relative
    # contributions across all three blocks, summed by aetiology category
    cat_tables = []
    for name, df in blocks.items():
        rel, sig_cols = add_relative_contributions(df, name)
        ref_cols = [c for c in sig_cols if not is_denovo(c) and not is_crc_specific(c)]
        cat_for_col = {f"REL__{c}": categorize_signature(c) for c in ref_cols}
        sub = rel[[f"REL__{c}" for c in ref_cols]].copy()
        sub.columns = [c.replace("REL__","") for c in sub.columns]
        cat_for_col = {k.replace("REL__",""): v for k,v in cat_for_col.items()}
        grouped = sub.T.groupby([cat_for_col[i] for i in sub.columns]).sum().T
        cat_tables.append(grouped)

    # sum category contributions across SBS/DBS/ID blocks (outer-join on sample)
    combined = cat_tables[0]
    for t in cat_tables[1:]:
        combined = combined.add(t, fill_value=0)
    combined = combined.div(combined.sum(axis=1).replace(0, np.nan), axis=0)
    combined.index.name = "sample_id"
    combined = combined.reset_index()

    merged = cluster_df.merge(combined, on="sample_id", how="inner")
    print(f"  Merged: {merged.shape}, categories: {[c for c in merged.columns if c not in ('sample_id','cluster')]}")

    cat_cols = [c for c in merged.columns if c not in ("sample_id","cluster")]
    stats = feature_statistics(merged, cat_cols)
    stats.to_csv(f"{c_dir}/etiology_statistics.csv", index=False)

    color_map = {c: ETIOLOGY_COLORS.get(c,"#999999") for c in cat_cols}
    composition_bar(merged, cat_cols, color_map,
        "Mutational-process (aetiology) composition by subtype\n"
        "(combined SBS + DBS + ID relative contributions)",
        f"{c_dir}/etiology_composition")

    _, zmat = cluster_zscore_matrix(merged, cat_cols)
    annotated_heatmap(zmat, stats, "feature",
        f"Aetiology category enrichment ({CLUSTER_NAME})",
        "Cluster z-score (relative contribution)", f"{c_dir}/etiology_heatmap")

    return merged, stats

##############################################################################
# PART C2 — AETIOLOGY EFFECT SIZES, C4 vs REST (Cliff's delta + bootstrap CI)
##############################################################################
def run_etiology_effectsize_C4(merged, outdir):
    """C4-vs-rest Cliff's delta effect sizes per aetiology category, with
    nonparametric bootstrap 95% CIs and BH-FDR correction across categories.

    Rebuilds the deleted '12g_event_weighted_mutational_aetiology' analysis
    (audit 2026-07-09) here inside Module 11 rather than as a standalone
    script, reusing the per-sample aetiology-fraction table already built by
    run_etiology_composition() (DRY -- avoids re-parsing SBS_signatures.xlsx).
    Statistically mirrors Module 24 Part A (genomic-burden effect sizes) so
    the two manuscript-strengthening analyses use one consistent method."""
    print("\n" + "="*70 + "\nPART C2 — AETIOLOGY EFFECT SIZES (C4 vs rest)\n" + "="*70)
    c_dir = os.path.join(outdir, "C_Etiology"); os.makedirs(c_dir, exist_ok=True)

    cat_cols = [c for c in merged.columns if c not in ("sample_id", "cluster")]
    rng = np.random.default_rng(RANDOM_STATE)
    is_c4 = (merged["cluster"] == C4_INTERNAL).values

    rows = []
    for cat in cat_cols:
        vals = pd.to_numeric(merged[cat], errors="coerce").values
        x = vals[is_c4]        # C4
        y = vals[~is_c4]       # rest
        delta, p = cliffs_delta_from_U(x, y)
        lo, hi = bootstrap_cliffs_ci(x, y, N_BOOTSTRAP, rng)
        rows.append({
            "category": cat,
            "C4_median": float(np.nanmedian(x)) if len(x) else np.nan,
            "rest_median": float(np.nanmedian(y)) if len(y) else np.nan,
            "cliffs_delta": delta,
            "delta_CI_low": lo, "delta_CI_high": hi,
            "direction": "C4 higher" if (pd.notna(delta) and delta > 0) else "C4 lower",
            "mannwhitney_p": p,
            "n_C4": int(np.sum(~np.isnan(x))), "n_rest": int(np.sum(~np.isnan(y))),
        })
    res = pd.DataFrame(rows)
    res["FDR"] = fdrcorrection(res["mannwhitney_p"])[1]
    res["significant_FDR"] = res["FDR"] < 0.05
    res = res.sort_values("cliffs_delta", ascending=False).reset_index(drop=True)

    out_csv = os.path.join(c_dir, "etiology_effectsize_C4_vs_rest.csv")
    res.to_csv(out_csv, index=False)
    print(res[["category", "C4_median", "rest_median", "cliffs_delta",
               "delta_CI_low", "delta_CI_high", "FDR", "significant_FDR"]].to_string(index=False))

    # Forest plot of Cliff's delta with bootstrap CI (Module 24 Part A style,
    # for visual/methodological consistency across the two analyses)
    plot_df = res.sort_values("cliffs_delta")
    fig, ax = plt.subplots(figsize=(10.5, max(5, 0.5 * len(plot_df) + 1.5)))
    y = np.arange(len(plot_df))
    for yi, (_, r) in zip(y, plot_df.iterrows()):
        d, lo, hi = r["cliffs_delta"], r["delta_CI_low"], r["delta_CI_high"]
        color = "#D55E00" if d > 0 else "#0072B2"
        star = "*" if r["significant_FDR"] else ""
        xerr = [[max(0, d - lo)], [max(0, hi - d)]] if np.isfinite(lo) else None
        ax.errorbar(d, yi, xerr=xerr, fmt="o", color=color, ecolor=color,
                    elinewidth=2.2, capsize=4, markersize=9, markeredgecolor="black")
        ax.text(1.02, yi, f"{d:+.2f}{star}", va="center", fontsize=13,
                transform=ax.get_yaxis_transform())
    ax.axvline(0, color="grey", ls="--", lw=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["category"].tolist())
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Cliff's delta (C4 vs rest)  ·  bootstrap 95% CI")
    ax.set_title(f"Aetiology category effect sizes ({CLUSTER_NAME})", fontweight="bold")
    ax.text(0.99, -0.11, "* FDR<0.05    right = higher in C4",
            transform=ax.transAxes, ha="right", va="top", fontsize=13)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(f"{c_dir}/etiology_effectsize_forest.{ext}", bbox_inches="tight")
    plt.close()
    print(f"  Saved {c_dir}/etiology_effectsize_forest.png/pdf")
    return res

##############################################################################
# PART D — PER-CLUSTER SIGNATURE PROFILE (diverging lollipop)
##############################################################################
def run_per_cluster_profiles(cluster_df, ref_results, outdir):
    print("\n" + "="*70 + "\nPART D — PER-CLUSTER SIGNATURE PROFILES\n" + "="*70)
    d_dir = os.path.join(outdir, "D_PerClusterProfiles"); os.makedirs(d_dir, exist_ok=True)

    # combine all reference signature relative contributions across blocks
    all_merged = cluster_df.copy()
    all_cols, all_stats = [], []
    for name, (merged, stats, ref_cols) in ref_results.items():
        m2 = merged.rename(columns={c: c.replace("REL__","") for c in merged.columns})
        cols = [c.replace("REL__","") for c in ref_cols]
        all_merged = all_merged.merge(m2[["sample_id"]+cols], on="sample_id", how="inner")
        all_cols += cols
        s2 = stats.copy(); s2["feature"] = s2["feature"].str.replace("REL__","")
        all_stats.append(s2)
    all_stats = pd.concat(all_stats, ignore_index=True)

    _, zmat = cluster_zscore_matrix(all_merged, all_cols)
    cat_map = {c: categorize_signature(c) for c in all_cols}
    for c in sorted(all_merged["cluster"].unique()):
        diverging_lollipop(zmat, all_stats, "feature", c,
            f"{display_cluster(c)}: mutational signature profile\n"
            "(blue = depleted, vermillion = enriched, relative to other subtypes)",
            f"{d_dir}/{display_cluster(c)}_signature_lollipop",
            category_map=cat_map, xlabel="Cluster z-score (relative contribution)")

    return all_merged, all_stats, zmat

##############################################################################
# PART E — DE NOVO SIGNATURE ACTIVITY
##############################################################################
def run_denovo_analysis(cluster_df, blocks, outdir):
    print("\n" + "="*70 + "\nPART E — DE NOVO SIGNATURE ACTIVITY\n" + "="*70)
    e_dir = os.path.join(outdir, "E_DeNovo"); os.makedirs(e_dir, exist_ok=True)
    for name, df in blocks.items():
        rel, sig_cols = add_relative_contributions(df, name)
        denovo_cols = [f"REL__{c}" for c in sig_cols if is_denovo(c)]
        if not denovo_cols:
            continue
        merged = cluster_df.merge(rel[denovo_cols], on="sample_id", how="inner")
        stats = feature_statistics(merged, denovo_cols)
        stats.to_csv(f"{e_dir}/{name}_denovo_statistics.csv", index=False)
        sig = stats.loc[stats.FDR < 0.05]
        selected = (sig if len(sig) else stats).head(TOP_DENOVO)["feature"].tolist()
        if not selected: continue
        _, zmat = cluster_zscore_matrix(merged, selected)
        zmat.index = [i.replace("REL__","") for i in zmat.index]
        disp_stats = stats.copy(); disp_stats["feature"] = disp_stats["feature"].str.replace("REL__","")
        annotated_heatmap(zmat, disp_stats, "feature",
            f"{name} de novo signature activity ({CLUSTER_NAME})\n"
            "(dataset-specific; no published aetiology)",
            "Cluster z-score (relative contribution)", f"{e_dir}/{name}_denovo_heatmap")

##############################################################################
# PART F — INTEGRATIVE SUMMARY
##############################################################################
def run_integrative_summary(cluster_df, burden_res, ref_results, etio_res, outdir):
    print("\n" + "="*70 + "\nPART F — INTEGRATIVE SUMMARY\n" + "="*70)
    f_dir = os.path.join(outdir, "F_Integrative"); os.makedirs(f_dir, exist_ok=True)

    blocks_z, side_labels = [], []

    # burden (log10)
    merged_b, stats_b = burden_res
    burden_cols = stats_b["feature"].tolist()
    log_cols = []
    for c in burden_cols:
        lc = f"log10_{c}"
        merged_b[lc] = np.log10(merged_b[c].clip(lower=1))
        log_cols.append(lc)
    log_stats = feature_statistics(merged_b, log_cols)
    _, z = cluster_zscore_matrix(merged_b, log_cols)
    z.index = [i.replace("log10_","log10 ") for i in z.index]
    blocks_z.append(z); side_labels += ["Burden"]*len(z)

    # top reference signatures per block
    for name, (merged, stats, ref_cols) in ref_results.items():
        top = stats.head(4)["feature"].tolist()
        if not top: continue
        _, z = cluster_zscore_matrix(merged, top)
        z.index = [i.replace("REL__","") for i in z.index]
        blocks_z.append(z); side_labels += [f"{name} signature"]*len(z)

    # etiology categories
    merged_e, stats_e = etio_res
    top_e = stats_e.head(6)["feature"].tolist()
    if top_e:
        _, z = cluster_zscore_matrix(merged_e, top_e)
        blocks_z.append(z); side_labels += ["Aetiology"]*len(z)

    if not blocks_z:
        print("  Nothing to integrate -- skipping"); return

    combined = pd.concat(blocks_z)
    uniq = list(dict.fromkeys(side_labels))
    side_colors = dict(zip(uniq, CLUSTER_PALETTE))
    stub = pd.DataFrame({"feature":combined.index, "FDR":np.nan})
    annotated_heatmap(combined, stub, "feature",
        f"Integrative mutational-signature summary ({CLUSTER_NAME})",
        "Cluster z-score", f"{f_dir}/integrative_heatmap",
        side_labels=side_labels, side_colors=side_colors, side_title="Block",
        figsize=(2.0*combined.shape[1]+6, 0.4*len(combined)+2))

    # radar: one representative row per block
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
                   color=CLUSTER_COLORS[display_cluster(c)], lw=2)
            ax.fill(angles, vals, alpha=0.08, color=CLUSTER_COLORS[display_cluster(c)])
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, fontsize=10)
        ax.set_title("Representative signature profile by subtype",
                     fontsize=16, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3,1.1), fontsize=12)
        plt.tight_layout()
        for ext in ["png","pdf"]:
            plt.savefig(f"{f_dir}/radar_summary.{ext}", bbox_inches="tight")
        plt.close()
        print(f"  Saved {f_dir}/radar_summary.png/pdf")

##############################################################################
# MAIN
##############################################################################
if __name__ == "__main__":
    print("="*70)
    print(f"MODULE 11 — MUTATIONAL SIGNATURE CHARACTERIZATION ({CLUSTER_NAME})")
    print("="*70)

    cluster_df = load_clusters()
    sizes = cluster_df["cluster"].value_counts().sort_index()
    sizes.index = display_cluster_list(sizes.index)
    print(f"Cluster sizes: {sizes.to_dict()}")

    blocks = load_signature_blocks()

    burden_res = run_burden_analysis(cluster_df, blocks, OUTDIR)
    ref_results = run_reference_signature_analysis(cluster_df, blocks, OUTDIR)
    etio_res    = run_etiology_composition(cluster_df, blocks, OUTDIR)
    run_etiology_effectsize_C4(etio_res[0], OUTDIR)
    run_per_cluster_profiles(cluster_df, ref_results, OUTDIR)
    run_denovo_analysis(cluster_df, blocks, OUTDIR)
    run_integrative_summary(cluster_df, burden_res, ref_results, etio_res, OUTDIR)

    print("\n" + "="*70)
    print("MODULE 11 COMPLETE")
    print(f"Outputs: {OUTDIR}")
    print("="*70)
