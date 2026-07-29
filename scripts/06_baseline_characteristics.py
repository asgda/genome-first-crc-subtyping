#!/usr/bin/env python3
"""Generate baseline clinical and genomic characteristics by CRC subtype.

Continuous variables use Kruskal-Wallis tests and categorical variables use
chi-square or Fisher exact tests as appropriate. Structural-variant burden is
the unique BRASS junction count produced by script 03.
"""

import os
import re
import zipfile
import warnings
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.stats import kruskal, chi2_contingency, fisher_exact

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

##############################################################################
# CONFIGURATION
##############################################################################
BASE = os.environ.get("CRC_BASE", str(Path(__file__).resolve().parents[1]))
SUPP_TABLE = os.environ.get(
    "CRC_SUPP_TABLE", f"{BASE}/crc_heterogeneity_data/Supplementary_Table_01.xlsx")
CLUSTER_FILE = os.environ.get(
    "CRC_CLUSTER_FILE", f"{BASE}/module05_06_loocv_results/labels/NMF_k4_LOOCV.csv")
SV_BURDEN_FILE = os.environ.get(
    "CRC_M3_BURDEN", f"{BASE}/module3_results/module3_sv_burden_matrix.csv")
OUTDIR = os.environ.get("CRC_M23_OUT", f"{BASE}/module23_table1_baseline_C1C4")
FIGDIR = os.path.join(OUTDIR, "figures")
TABDIR = os.path.join(OUTDIR, "tables")
for d in (OUTDIR, FIGDIR, TABDIR):
    os.makedirs(d, exist_ok=True)

TMB_DENOMINATOR_MB = 3_000.0
MISSING_TOKENS = {"", "nan", "na", "n/a", "not_applicable", "undefined", "none", "unknown"}

CLUSTER_COLORS = {"C1": "#E69F00", "C2": "#56B4E9", "C3": "#009E73", "C4": "#D55E00"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
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
    """0-based internal labels regardless of source 0- or 1-indexing
    (matches Modules 16/17/19/20/21/22)."""
    s = pd.to_numeric(series, errors="raise").astype(int)
    vals = sorted(s.dropna().unique().tolist())
    if vals and min(vals) == 1 and max(vals) <= 8 and 0 not in vals:
        s = s - 1
    return s


def _xlsx_to_dataframe(path, sheet="sheet1"):
    """Read a single sheet from an .xlsx using only stdlib (no openpyxl).
    Copied verbatim from Module 11 (Section 4 environment gotcha)."""
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    with zipfile.ZipFile(path) as z:
        with z.open("xl/sharedStrings.xml") as f:
            ss_root = ET.parse(f).getroot()
        shared = ["".join((t.text or "") for t in si.iter(f"{{{ns}}}t"))
                  for si in ss_root.findall(f"{{{ns}}}si")]
        sheet_path = f"xl/worksheets/{sheet}.xml"
        if sheet_path not in z.namelist():
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
            ci = col_idx(re.match(r"([A-Za-z]+)", ref).group(1))
            max_col = max(max_col, ci)
            t = c_el.attrib.get("t", "")
            v_el = c_el.find(f"{{{ns}}}v")
            if v_el is None:
                continue
            if t == "s":
                val = shared[int(v_el.text)]
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


def _is_missing(x):
    return str(x).strip().lower() in MISSING_TOKENS


##############################################################################
# LOAD
##############################################################################
def load_data():
    raw = _xlsx_to_dataframe(SUPP_TABLE)
    raw.columns = raw.iloc[0]
    st = raw.iloc[1:].reset_index(drop=True)
    st["sid"] = st["Sample ID"].map(normalize_sample)
    st = st.dropna(subset=["sid"]).drop_duplicates("sid")

    lab = pd.read_csv(CLUSTER_FILE)
    if not {"sample_id", "cluster"}.issubset(lab.columns):
        raise ValueError("Cluster file must contain sample_id and cluster columns.")
    lab["sid"] = lab["sample_id"].map(normalize_sample)
    lab["cluster"] = normalize_cluster_values(lab["cluster"])
    lab = lab.dropna(subset=["sid"]).drop_duplicates("sid")[["sid", "cluster"]]

    df = st.merge(lab, on="sid", how="inner")
    df["cluster_display"] = df["cluster"].map(display_cluster)

    svb = pd.read_csv(SV_BURDEN_FILE)
    if not {"sample_id", "SV_TOTAL"}.issubset(svb.columns):
        raise ValueError(
            f"SV burden file must contain sample_id and SV_TOTAL: {SV_BURDEN_FILE}"
        )
    svb["sid"] = svb["sample_id"].map(normalize_sample)
    svb = svb.dropna(subset=["sid"]).drop_duplicates("sid")[["sid", "SV_TOTAL"]]
    df = df.merge(svb, on="sid", how="left", validate="one_to_one")

    _expected = {0: 426, 1: 274, 2: 268, 3: 94}
    _obs = {int(k): int(v) for k, v in df["cluster"].value_counts().sort_index().items()}
    if _obs != _expected:
        raise ValueError(
            f"Subtype sizes after merge do not match the locked solution "
            f"(observed {_obs}, expected {_expected}). Check CLUSTER_FILE={CLUSTER_FILE} "
            f"and that every labelled sample is present in the supplementary table.")

    # Derived continuous fields (defined identically to Module 13).
    df["TMB"] = pd.to_numeric(df["Total Mutation Count"], errors="coerce") / TMB_DENOMINATOR_MB
    df["Age"] = pd.to_numeric(df["Age at diagnosis"], errors="coerce")
    df["SV_count"] = pd.to_numeric(df["SV_TOTAL"], errors="coerce")
    if df["SV_count"].isna().any():
        missing = df.loc[df["SV_count"].isna(), "sid"].tolist()
        raise ValueError(
            f"Canonical SV burden missing for {len(missing)} labelled samples: "
            f"{missing[:10]}"
        )
    df["CN_segments"] = pd.to_numeric(df["Copy Number Segments"], errors="coerce")
    df["Purity"] = pd.to_numeric(df["Tumour Cell Content Pathology"], errors="coerce")
    df["Hypoxia"] = pd.to_numeric(df["Tumour Hypoxia Buffa Score"], errors="coerce")
    print(f"Loaded {len(df)} labelled samples; sizes {_obs}")
    return df


##############################################################################
# TABLE 1 ROW BUILDERS
##############################################################################
ORDER = ["C1", "C2", "C3", "C4"]


def _fmt_med_iqr(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return "-"
    return f"{s.median():.1f} [{s.quantile(0.25):.1f}-{s.quantile(0.75):.1f}]"


def continuous_row(df, col, label):
    groups = [pd.to_numeric(df.loc[df.cluster_display == c, col], errors="coerce").dropna().values
              for c in ORDER]
    cells = {c: _fmt_med_iqr(df.loc[df.cluster_display == c, col]) for c in ORDER}
    cells["Overall"] = _fmt_med_iqr(df[col])
    usable = [g for g in groups if len(g) >= 1]
    if len(usable) >= 2 and all(len(g) > 0 for g in groups):
        stat, p = kruskal(*groups)
        test = "Kruskal-Wallis"
    else:
        stat, p, test = np.nan, np.nan, "n/a (empty group)"
    n_nonmiss = int(pd.to_numeric(df[col], errors="coerce").notna().sum())
    return {"characteristic": f"{label}, median [IQR]", **cells,
            "p_value": p, "test": test, "stat": stat, "n_nonmissing": n_nonmiss}


def categorical_rows(df, col, label, level_order=None):
    """Header row (with omnibus p) + one indented row per observed level.
    Explicit-missing tokens are shown as a 'Missing/Undefined' level but
    excluded from the omnibus test (complete-case)."""
    vals = df[col].astype(str)
    is_miss = vals.map(_is_missing)
    obs_levels = sorted([v for v in vals[~is_miss].unique()])
    if level_order:
        obs_levels = [lv for lv in level_order if lv in obs_levels] + \
                     [lv for lv in obs_levels if lv not in level_order]

    # complete-case contingency (levels x subtype) for the omnibus test
    sub = df.loc[~is_miss]
    ct = pd.crosstab(sub[col].astype(str), sub["cluster_display"]).reindex(
        index=obs_levels, columns=ORDER, fill_value=0)
    p, test, stat = np.nan, "n/a", np.nan
    if ct.shape[0] >= 2 and ct.values.sum() > 0 and (ct.sum(axis=0) > 0).all():
        chi2, p_chi, dof, expected = chi2_contingency(ct.values)
        min_exp = expected.min()
        if ct.shape == (2, 2) and min_exp < 5:
            _, p = fisher_exact(ct.values)
            test, stat = "Fisher exact", np.nan
        elif min_exp < 5:
            p, stat, test = p_chi, chi2, "chi2 (sparse; interpret with caution)"
        else:
            p, stat, test = p_chi, chi2, "chi-square"

    rows = [{"characteristic": f"{label}, n (%)",
             **{c: "" for c in ORDER}, "Overall": "",
             "p_value": p, "test": test, "stat": stat,
             "n_nonmissing": int((~is_miss).sum())}]
    n_by_cluster = {c: int((df.cluster_display == c).sum()) for c in ORDER}
    n_overall = len(df)
    display_levels = obs_levels + (["Missing/Undefined"] if is_miss.any() else [])
    for lv in display_levels:
        if lv == "Missing/Undefined":
            mask = is_miss
        else:
            mask = (vals == lv) & (~is_miss)
        cells = {}
        for c in ORDER:
            n = int((mask & (df.cluster_display == c)).sum())
            cells[c] = f"{n} ({100 * n / n_by_cluster[c]:.1f})" if n_by_cluster[c] else "0 (0.0)"
        n_all = int(mask.sum())
        cells["Overall"] = f"{n_all} ({100 * n_all / n_overall:.1f})"
        rows.append({"characteristic": f"    {lv}", **cells,
                     "p_value": np.nan, "test": "", "stat": np.nan, "n_nonmissing": np.nan})
    return rows


##############################################################################
# RENDER FIGURE
##############################################################################
def render_table_figure(tbl, out_base):
    show = tbl[["characteristic", "Overall"] + ORDER + ["p_value"]].copy()
    show["p_value"] = show["p_value"].map(
        lambda p: "" if pd.isna(p) else ("<0.001" if p < 0.001 else f"{p:.3f}"))
    n_rows = len(show)
    fig, ax = plt.subplots(figsize=(13, max(4, 0.34 * n_rows + 1.2)))
    ax.axis("off")
    header = ["Characteristic", f"Overall\n(n={len(tbl)})"] + \
             [f"{c}" for c in ORDER] + ["p"]
    t = ax.table(cellText=show.values, colLabels=header,
                 cellLoc="left", loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.25)
    for j, c in enumerate(ORDER):
        cell = t[0, 2 + j]
        cell.set_facecolor(CLUSTER_COLORS[c])
        cell.set_text_props(color="white", fontweight="bold")
    for j in range(len(header)):
        t[0, j].set_text_props(fontweight="bold")
    # bold section-header rows (those carrying a p-value / not indented)
    for i in range(len(show)):
        first = str(show.iloc[i]["characteristic"])
        if not first.startswith("    "):
            for j in range(len(header)):
                t[i + 1, j].set_text_props(fontweight="bold")
    ax.set_title("Table 1  —  Baseline characteristics by WGS subtype",
                 fontweight="bold", fontsize=14, pad=12)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(f"{out_base}.{ext}", bbox_inches="tight")
    plt.close()
    print(f"  -> {out_base}.png/pdf")


##############################################################################
# MAIN
##############################################################################
def main():
    print("=" * 70)
    print("MODULE 23 — TABLE 1 BASELINE CHARACTERISTICS (C1-C4)")
    print("=" * 70)
    df = load_data()

    rows = []
    # --- Demographics / clinical ---
    rows.append(continuous_row(df, "Age", "Age at diagnosis (years)"))
    rows += categorical_rows(df, "Sex", "Sex", ["Female", "Male"])
    rows += categorical_rows(df, "Tumour Stage", "AJCC stage",
                             ["Stage I", "Stage II", "Stage III", "Stage IV"])
    rows += categorical_rows(df, "Tumour Site", "Primary site",
                             ["Right Colon", "Left Colon", "Rectum"])
    rows += categorical_rows(df, "Tumour Grade", "Tumour grade")
    rows += categorical_rows(df, "Pre-Treated", "Neoadjuvant treatment",
                             ["Untreated", "Treated"])
    # --- Molecular classifiers ---
    rows += categorical_rows(df, "MSI Status", "MSI status", ["MSS", "MSI"])
    rows += categorical_rows(df, "CMS Tumour", "CMS",
                             ["CMS1", "CMS2", "CMS3", "CMS4"])
    rows += categorical_rows(df, "iCMS Tumour", "iCMS", ["iCMS2", "iCMS3"])
    rows += categorical_rows(df, "CRPS Tumour", "CRPS",
                             ["CRPS1", "CRPS2", "CRPS3", "CRPS4", "CRPS5"])
    # --- Coarse genomic burden ---
    rows.append(continuous_row(df, "TMB", "TMB (mut/Mb)"))
    rows.append(continuous_row(df, "SV_count", "Structural variants (n)"))
    rows.append(continuous_row(df, "CN_segments", "Copy-number segments (n)"))
    rows.append(continuous_row(df, "Purity", "Tumour purity (%)"))
    rows.append(continuous_row(df, "Hypoxia", "Hypoxia (Buffa score)"))
    # --- Outcome ---
    rows += categorical_rows(df, "Recurrence", "Recurrence", ["No", "Yes"])
    rows += categorical_rows(df, "Vital Status", "Vital status", ["Alive", "Dead"])

    tbl = pd.DataFrame(rows)
    # ensure column order
    tbl = tbl[["characteristic", "Overall"] + ORDER +
              ["p_value", "test", "stat", "n_nonmissing"]]

    main_csv = os.path.join(TABDIR, "table1_baseline_characteristics.csv")
    tbl[["characteristic", "Overall"] + ORDER + ["p_value"]].to_csv(main_csv, index=False)
    stat_csv = os.path.join(TABDIR, "table1_row_statistics.csv")
    tbl[["characteristic", "p_value", "test", "stat", "n_nonmissing"]].dropna(
        subset=["test"]).query("test != ''").to_csv(stat_csv, index=False)
    render_table_figure(tbl, os.path.join(FIGDIR, "table1_baseline"))

    print(f"\nSaved:\n  {main_csv}\n  {stat_csv}")
    print("\n" + "=" * 70)
    print("MODULE 23 COMPLETE")
    print(f"Outputs: {OUTDIR}")
    print("=" * 70)
    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.max_colwidth", 40):
        print(tbl[["characteristic", "Overall"] + ORDER + ["p_value", "test"]].to_string(index=False))
    return tbl


if __name__ == "__main__":
    main()
