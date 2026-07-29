#!/usr/bin/env python3
"""Compare genomic C1-C4 labels with CMS, CRPS and iCMS classifications.

Defined comparator labels are used for ARI, NMI and Cramer V; undefined calls
are summarized separately as callability categories. Same-sample nested Cox
models assess complementary prognostic information.
"""

import os
import re
import warnings
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from scipy.stats import chi2_contingency
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

warnings.filterwarnings("ignore")

BASE = os.environ.get("CRC_BASE", str(Path(__file__).resolve().parents[1]))
SUPP_TABLE = os.environ.get("CRC_SUPP_TABLE", f"{BASE}/crc_heterogeneity_data/Supplementary_Table_01.xlsx")
CLUSTER_FILE = os.environ.get("CRC_CLUSTER_FILE", f"{BASE}/module05_06_loocv_results/labels/NMF_k4_LOOCV.csv")
OUTDIR = Path(os.environ.get("CRC_M19_OUT", f"{BASE}/module19_crps_cms_icms_comparison_C1C4_ppt"))
FIGDIR = OUTDIR / "figures"
TABDIR = OUTDIR / "tables"
FIGDIR.mkdir(parents=True, exist_ok=True)
TABDIR.mkdir(parents=True, exist_ok=True)

CB8 = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999"]
CLUSTER_COLORS = {"C1":"#E69F00", "C2":"#56B4E9", "C3":"#009E73", "C4":"#D55E00"}
CMS_COLORS = {"CMS1":"#E69F00", "CMS2":"#56B4E9", "CMS3":"#009E73", "CMS4":"#D55E00", "Undefined":"#BDBDBD", "NA":"#BDBDBD"}
CRPS_COLORS = {"CRPS1":"#E69F00", "CRPS2":"#56B4E9", "CRPS3":"#009E73", "CRPS4":"#D55E00", "CRPS5":"#CC79A7", "Undefined":"#BDBDBD", "NA":"#BDBDBD"}
ICMS_COLORS = {"iCMS2":"#0072B2", "iCMS3":"#D55E00", "iCMS2-like":"#56B4E9", "iCMS3-like":"#E69F00", "Undefined":"#BDBDBD", "NA":"#BDBDBD"}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 18,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 15,
    "savefig.dpi": 300,
    "axes.linewidth": 1.1,
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def require_lifelines():
    try:
        from lifelines import CoxPHFitter, KaplanMeierFitter
        from lifelines.statistics import multivariate_logrank_test
        return CoxPHFitter, KaplanMeierFitter, multivariate_logrank_test
    except ImportError as e:
        raise ImportError("This script needs lifelines. Install in vep_env: pip install lifelines") from e


def read_xlsx_first_sheet(path):
    """
    Lightweight .xlsx reader for environments without openpyxl.

    The module only needs the first worksheet of Supplementary_Table_01.xlsx,
    so this parser intentionally keeps the scope narrow: shared strings,
    inline strings, and raw cell values from the first worksheet are enough
    for the downstream column lookup and numeric coercion.
    """
    ns_main = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    ns_rel = "{http://schemas.openxmlformats.org/package/2006/relationships}"

    def cell_text(cell, shared_strings):
        cell_type = cell.attrib.get("t")
        if cell_type == "s":
            v = cell.find(f"{ns_main}v")
            return shared_strings[int(v.text)] if v is not None and v.text is not None else ""
        if cell_type == "inlineStr":
            parts = [t.text or "" for t in cell.findall(f".//{ns_main}t")]
            return "".join(parts)
        if cell_type == "b":
            v = cell.find(f"{ns_main}v")
            return "1" if v is not None and v.text == "1" else "0"
        v = cell.find(f"{ns_main}v")
        return "" if v is None or v.text is None else v.text

    def col_to_idx(cell_ref):
        letters = re.match(r"([A-Z]+)", cell_ref).group(1)
        idx = 0
        for ch in letters:
            idx = idx * 26 + (ord(ch) - 64)
        return idx - 1

    with zipfile.ZipFile(path) as zf:
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall(f"{ns_rel}Relationship")
        }
        sheets = wb.find(f"{ns_main}sheets")
        first_sheet = sheets.find(f"{ns_main}sheet")
        rel_id = first_sheet.attrib[f"{{http://schemas.openxmlformats.org/officeDocument/2006/relationships}}id"]
        sheet_path = "xl/" + rel_map[rel_id].lstrip("/")

        shared_strings = []
        if "xl/sharedStrings.xml" in zf.namelist():
            sst = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in sst.findall(f"{ns_main}si"):
                shared_strings.append("".join(t.text or "" for t in si.findall(f".//{ns_main}t")))

        sheet = ET.fromstring(zf.read(sheet_path))
        rows = []
        for row in sheet.findall(f".//{ns_main}row"):
            values = {}
            for cell in row.findall(f"{ns_main}c"):
                ref = cell.attrib.get("r", "")
                idx = col_to_idx(ref)
                values[idx] = cell_text(cell, shared_strings)
            if values:
                max_idx = max(values)
                rows.append([values.get(i, "") for i in range(max_idx + 1)])
            else:
                rows.append([])

    if not rows:
        return pd.DataFrame()
    header = [str(x).strip() for x in rows[0]]
    data = rows[1:]
    width = max(len(header), max((len(r) for r in data), default=0))
    header = header + [f"Unnamed: {i}" for i in range(len(header), width)]
    padded = [r + [""] * (width - len(r)) for r in data]
    return pd.DataFrame(padded, columns=header)


def normalize_sample(x):
    m = re.search(r"(UM\d+|U\d+)", str(x))
    return m.group(1) if m else None


def clean_cat(x):
    if pd.isna(x): return "NA"
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "not_available", "not applicable"}:
        return "NA"
    return s


def normalize_cluster_values(series):
    s = pd.to_numeric(series, errors="raise").astype(int)
    vals = sorted(s.dropna().unique().tolist())
    if vals and min(vals) == 1 and max(vals) <= 8 and 0 not in vals:
        s = s - 1
    return s


def display_cluster(c):
    return f"C{int(c)+1}"


def find_col(df, candidates, required=True):
    lookup = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower().strip() in lookup:
            return lookup[cand.lower().strip()]
    for cand in candidates:
        key = cand.lower().strip()
        for low, original in lookup.items():
            if key in low:
                return original
    if required:
        raise KeyError(f"Could not find any of columns: {candidates}")
    return None


def normalize_cms(x):
    s = clean_cat(x).upper().replace(" ", "")
    if s in {"CMS1", "CMS2", "CMS3", "CMS4"}: return s
    if s in {"NA", "UNDEFINED"}: return "Undefined"
    m = re.search(r"CMS\s*([1-4])", str(x), re.I)
    return f"CMS{m.group(1)}" if m else clean_cat(x)


def normalize_crps(x):
    s = clean_cat(x).upper().replace(" ", "")
    if re.match(r"^CRPS[1-5]$", s): return s
    if s in {"NA", "UNDEFINED"}: return "Undefined"
    m = re.search(r"CRPS\s*([1-5])", str(x), re.I)
    return f"CRPS{m.group(1)}" if m else clean_cat(x)


def normalize_icms(x):
    s = clean_cat(x)
    su = s.upper().replace(" ", "")
    if su in {"ICMS2", "ICMS3"}: return "iCMS" + su[-1]
    if "ICMS2" in su: return "iCMS2-like"
    if "ICMS3" in su: return "iCMS3-like"
    if su in {"NA", "UNDEFINED"}: return "Undefined"
    return s


def normalize_stage(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    stage_map = {"Stage I":1, "Stage II":2, "Stage III":3, "Stage IV":4, "I":1, "II":2, "III":3, "IV":4}
    if s in stage_map: return stage_map[s]
    m = re.search(r"Stage\s*(I{1,3}|IV)", s, re.I)
    return stage_map.get("Stage " + m.group(1).upper(), np.nan) if m else pd.to_numeric(s, errors="coerce")


def normalize_msi(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower().replace("_", "-").replace(" ", "-")
    if s in {"msi", "msi-h", "msi-high", "unstable", "instable"}: return 1.0
    if s in {"mss", "msi-l", "msi-low", "stable"}: return 0.0
    return np.nan


def normalize_sex(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s in {"male", "m"}: return 1.0
    if s in {"female", "f"}: return 0.0
    return np.nan


def normalize_pretreated(x):
    if pd.isna(x): return 0.0
    s = str(x).strip().lower()
    return 1.0 if ("treated" in s and "not" not in s and "untreated" not in s) else 0.0


def read_labels():
    if not Path(CLUSTER_FILE).exists():
        raise FileNotFoundError(f"Cluster label file not found: {CLUSTER_FILE}. Set CRC_CLUSTER_FILE.")
    lab = pd.read_csv(CLUSTER_FILE)
    lab["sid"] = lab["sample_id"].map(normalize_sample)
    lab = lab.dropna(subset=["sid"]).drop_duplicates("sid")
    lab["cluster"] = normalize_cluster_values(lab["cluster"])
    lab["WGS"] = lab["cluster"].map(display_cluster)
    return lab[["sid", "cluster", "WGS"]]


def read_supplementary():
    """
    Single source of truth for all clinical/molecular covariates used in
    this module, including CMS and CRPS.

    FIX: the previous version pulled CMS/CRPS from a separate file
    (clinical_data.tsv, via read_clinical()) while pulling iCMS, RFS, stage,
    age, sex, MSI from Supplementary_Table_01.xlsx -- two files, two merges,
    two chances for sample-ID mismatch to silently shrink the cohort.
    Direct inspection of Supplementary_Table_01.xlsx confirmed it already
    contains clean "CMS Tumour" and "CRPS Tumour" columns alongside
    everything else (CMS: CMS1-4 + Undefined, n=1063; CRPS: CRPS1-5 +
    Undefined, n=1063), so there is no need for a second file at all.
    """
    if not Path(SUPP_TABLE).exists():
        raise FileNotFoundError(f"Supplementary table not found: {SUPP_TABLE}. Set CRC_SUPP_TABLE.")
    try:
        st = pd.read_excel(SUPP_TABLE)
    except ImportError:
        st = read_xlsx_first_sheet(SUPP_TABLE)
    sample_col = find_col(st, ["DNA Tumor Sample Barcode", "DNA Tumour Sample Barcode", "sample_id", "Sample ID"], required=False) or st.columns[0]
    st["sid"] = st[sample_col].map(normalize_sample)
    icms_col = find_col(st, ["iCMS", "iCMS Tumour", "iCMS Tumor", "Intrinsic CMS", "iCMS classification"], required=False)
    cms_col = find_col(st, ["CMS Tumour", "CMS_TUMOR", "CMS Tumor", "CMS"], required=False)
    crps_col = find_col(st, ["CRPS Tumour", "CRPS_TUMOR", "CRPS Tumor", "CRPS"], required=False)
    rfs_col = find_col(st, ["Recurrence free survival days", "RFS_days", "RFS days"], required=False)
    rec_col = find_col(st, ["Recurrence", "RFS_event"], required=False)
    stage_col = find_col(st, ["Tumour Stage", "Tumor Stage", "Tumor_Stage"], required=False)
    age_col = find_col(st, ["Age at diagnosis", "Age"], required=False)
    sex_col = find_col(st, ["Sex", "Gender"], required=False)
    msi_col = find_col(st, ["MSI Status", "MSI_STATUS", "MSI"], required=False)
    pret_col = find_col(st, ["Pre-Treated", "Pre Treated", "Pretreated", "Pre-treatment"], required=False)
    out = st[["sid"]].copy()
    out["iCMS"] = st[icms_col].map(normalize_icms) if icms_col else "NA"
    out["CMS"] = st[cms_col].map(normalize_cms) if cms_col else "NA"
    out["CRPS"] = st[crps_col].map(normalize_crps) if crps_col else "NA"
    if rfs_col:
        out["RFS_days"] = pd.to_numeric(st[rfs_col], errors="coerce")
        out["RFS_months"] = out["RFS_days"] / 30.44
    if rec_col:
        rec = st[rec_col].astype(str).str.lower().str.strip()
        out["RFS_event"] = np.where(rec.isin(["yes", "1", "true", "recurrence", "recurred"]), 1, 0)
    # ADDED (OS arm): overall survival from this same single source, so the OS
    # models inherit the identical sample-ID handling as everything above and
    # no second merge can silently shrink the cohort.
    os_col = find_col(st, ["Overall survival days", "OS_days", "OS days"], required=False)
    vit_col = find_col(st, ["Vital Status", "Vital_Status", "OS_STATUS"], required=False)
    if os_col:
        out["OS_days"] = pd.to_numeric(st[os_col], errors="coerce")
        out["OS_months"] = out["OS_days"] / 30.44
    if vit_col:
        vit = st[vit_col].astype(str).str.lower().str.strip()
        out["OS_event"] = np.where(vit.isin(["dead", "deceased", "1", "true"]), 1.0,
                                   np.where(vit.isin(["alive", "0", "false"]), 0.0, np.nan))
    out["stage_num"] = st[stage_col].map(normalize_stage).astype(float) if stage_col else np.nan
    out["age"] = pd.to_numeric(st[age_col], errors="coerce") if age_col else np.nan
    out["sex_m"] = st[sex_col].map(normalize_sex) if sex_col else np.nan
    out["msi_bin"] = st[msi_col].map(normalize_msi) if msi_col else np.nan
    out["pretreated"] = st[pret_col].map(normalize_pretreated) if pret_col else 0.0
    return out.dropna(subset=["sid"]).drop_duplicates("sid")


def load_all():
    lab = read_labels()
    supp = read_supplementary()
    df = lab.merge(supp, on="sid", how="left")
    for c in ["CMS", "CRPS", "iCMS"]:
        df[c] = df[c].fillna("NA").replace("NA", "Undefined")
    df["C4_binary"] = (df["WGS"] == "C4").astype(float)
    return df


def cramers_v(ct):
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.values.sum()
    r, k = ct.shape
    return np.sqrt((chi2/n) / max(min(k-1, r-1), 1))


def concordance_metrics(df):
    rows = []
    for target in ["CMS", "CRPS", "iCMS"]:
        sub = df[["WGS", target]].dropna()
        sub = sub[(sub[target] != "Undefined") & (sub[target] != "NA")]
        if len(sub) < 30 or sub[target].nunique() < 2:
            rows.append({"comparison": f"WGS_vs_{target}", "n": len(sub), "status": "skipped"})
            continue
        ct = pd.crosstab(sub["WGS"], sub[target])
        chi2, p, dof, _ = chi2_contingency(ct)
        rows.append({
            "comparison": f"WGS_vs_{target}",
            "n": len(sub),
            "n_WGS_classes": sub["WGS"].nunique(),
            f"n_{target}_classes": sub[target].nunique(),
            "ARI": adjusted_rand_score(sub["WGS"], sub[target]),
            "NMI": normalized_mutual_info_score(sub["WGS"], sub[target]),
            "chi2": chi2,
            "dof": dof,
            "pvalue": p,
            "cramers_v": cramers_v(ct),
            "status": "ok",
        })
        ct.to_csv(TABDIR / f"crosstab_WGS_vs_{target}.csv")
        (ct.div(ct.sum(axis=1), axis=0)*100).to_csv(TABDIR / f"crosstab_WGS_vs_{target}_rowpercent.csv")
    out = pd.DataFrame(rows)
    out.to_csv(TABDIR / "concordance_metrics.csv", index=False)
    return out


def assignment_coverage(df):
    """Summarise comparator callability and WGS assignments among undefined calls.

    Undefined labels are excluded from concordance statistics because they are
    not biological classes in CMS, CRPS or iCMS.  They are nevertheless an
    informative callability group: the WGS workflow assigns every matched
    tumour to C1-C4.  Reporting this in a separate table preserves both facts
    without treating "Undefined" as an additional transcriptomic subtype.
    """
    rows = []
    full_tables = {}
    wgs_order = ["C1", "C2", "C3", "C4"]
    for target in ["CMS", "CRPS", "iCMS"]:
        sub = df[["WGS", target]].dropna(subset=["WGS"]).copy()
        undefined = sub[target].isin(["Undefined", "NA"]) | sub[target].isna()
        n_total = int(len(sub))
        n_undefined = int(undefined.sum())
        counts = sub.loc[undefined, "WGS"].value_counts().reindex(wgs_order, fill_value=0)
        row = {
            "classification": target,
            "n_total_matched": n_total,
            "n_defined": n_total - n_undefined,
            "defined_percent": 100 * (n_total - n_undefined) / n_total if n_total else np.nan,
            "n_undefined": n_undefined,
            "undefined_percent": 100 * n_undefined / n_total if n_total else np.nan,
            "n_undefined_assigned_WGS": int(counts.sum()),
        }
        for wgs in wgs_order:
            row[f"undefined_assigned_{wgs}_n"] = int(counts[wgs])
            row[f"undefined_assigned_{wgs}_percent"] = (
                100 * int(counts[wgs]) / n_undefined if n_undefined else 0.0
            )
        rows.append(row)

        full_ct = pd.crosstab(sub["WGS"], sub[target]).reindex(wgs_order, fill_value=0)
        full_tables[target] = full_ct
        full_ct.to_csv(TABDIR / f"crosstab_WGS_vs_{target}_including_undefined.csv")

    out = pd.DataFrame(rows)
    out.to_csv(TABDIR / "classification_callability_and_WGS_assignment.csv", index=False)
    return out, full_tables


def _bar_segments(order, totals, total_n):
    segs, cur = {}, 0.0
    for lbl in order:
        frac = totals.get(lbl, 0) / max(total_n, 1)
        segs[lbl] = [cur, cur + frac]
        cur += frac
    return segs


def _ribbon(ax, x0, y0a, y0b, x1, y1a, y1b, color, alpha=0.38):
    xm = (x0 + x1) / 2
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a),
             (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, zorder=1))


def alluvial_crps_wgs_cms(df):
    sub = df[(df["CRPS"] != "Undefined") & (df["CMS"] != "Undefined")].copy()
    if len(sub) < 50:
        print("Skipping alluvial: too few samples with CRPS and CMS")
        return
    crps_order = [x for x in ["CRPS1", "CRPS2", "CRPS3", "CRPS4", "CRPS5"] if x in set(sub["CRPS"])]
    wgs_order = ["C1", "C2", "C3", "C4"]
    cms_order = [x for x in ["CMS1", "CMS2", "CMS3", "CMS4"] if x in set(sub["CMS"])]
    N = len(sub)
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.08); ax.axis("off")
    x_left, x_mid, x_right = 0.10, 0.50, 0.90
    bw = 0.055
    totals_l = sub["CRPS"].value_counts().reindex(crps_order, fill_value=0)
    totals_m = sub["WGS"].value_counts().reindex(wgs_order, fill_value=0)
    totals_r = sub["CMS"].value_counts().reindex(cms_order, fill_value=0)
    seg_l = _bar_segments(crps_order, totals_l, N)
    seg_m = _bar_segments(wgs_order, totals_m, N)
    seg_r = _bar_segments(cms_order, totals_r, N)
    cons_l = {k:v[0] for k,v in seg_l.items()}
    cons_m_l = {k:v[0] for k,v in seg_m.items()}
    for crps in crps_order:
        flow = sub[sub["CRPS"] == crps].groupby("WGS").size().reindex(wgs_order, fill_value=0)
        for wgs, n in flow.items():
            if n == 0: continue
            frac = n / N
            y0a, y0b = cons_l[crps], cons_l[crps] + frac
            y1a, y1b = cons_m_l[wgs], cons_m_l[wgs] + frac
            cons_l[crps], cons_m_l[wgs] = y0b, y1b
            _ribbon(ax, x_left + bw/2, y0a, y0b, x_mid - bw/2, y1a, y1b, CRPS_COLORS.get(crps, "#999999"))
    cons_m_r = {k:v[0] for k,v in seg_m.items()}
    cons_r = {k:v[0] for k,v in seg_r.items()}
    for wgs in wgs_order:
        flow = sub[sub["WGS"] == wgs].groupby("CMS").size().reindex(cms_order, fill_value=0)
        for cms, n in flow.items():
            if n == 0: continue
            frac = n / N
            y0a, y0b = cons_m_r[wgs], cons_m_r[wgs] + frac
            y1a, y1b = cons_r[cms], cons_r[cms] + frac
            cons_m_r[wgs], cons_r[cms] = y0b, y1b
            _ribbon(ax, x_mid + bw/2, y0a, y0b, x_right - bw/2, y1a, y1b, CMS_COLORS.get(cms, "#999999"))
    def draw_bar(segs, x, colors):
        for lbl, (a,b) in segs.items():
            ax.fill_betweenx([a,b], x-bw/2, x+bw/2, color=colors.get(lbl, "#999999"), zorder=3)
            pct = 100*(b-a)
            ax.text(x, (a+b)/2, f"{lbl}\n{pct:.0f}%", ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if pct > 7 else "black", zorder=5)
    draw_bar(seg_l, x_left, CRPS_COLORS)
    draw_bar(seg_m, x_mid, CLUSTER_COLORS)
    draw_bar(seg_r, x_right, CMS_COLORS)
    ax.text(x_left, 1.03, "CRPS", ha="center", fontsize=24, fontweight="bold")
    ax.text(x_mid, 1.03, "WGS subtype\n(NMF)", ha="center", fontsize=22, fontweight="bold")
    ax.text(x_right, 1.03, "CMS", ha="center", fontsize=24, fontweight="bold")
    for ext in ["png", "pdf"]:
        fig.savefig(FIGDIR / f"CRPS_WGS_CMS_alluvial.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def icms_stacked(df):
    sub = df[(df["iCMS"] != "Undefined") & (df["iCMS"] != "NA")].copy()
    if len(sub) < 50:
        return
    wgs_order = ["C1", "C2", "C3", "C4"]
    cats = [c for c in ["iCMS2", "iCMS3", "iCMS2-like", "iCMS3-like"] if c in set(sub["iCMS"])]
    ct = pd.crosstab(sub["WGS"], sub["iCMS"]).reindex(index=wgs_order, columns=cats, fill_value=0)
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    x = np.arange(len(wgs_order)); bottom = np.zeros(len(wgs_order))
    for cat in cats:
        vals = pct[cat].values
        ax.bar(x, vals, bottom=bottom, color=ICMS_COLORS.get(cat, "#999999"), edgecolor="white", label=cat)
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(wgs_order, fontweight="bold")
    ax.set_ylabel("Subtype fraction (%)")
    ax.set_title("iCMS", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIGDIR / f"iCMS_WGS_stacked_bar.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_km_group(df, group_col, label):
    _, KaplanMeierFitter, multivariate_logrank_test = require_lifelines()
    sub = df.dropna(subset=["RFS_months", "RFS_event", group_col])
    sub = sub[(sub["stage_num"].isin([1,2,3])) & (sub[group_col] != "Undefined")]
    if len(sub) < 50 or sub[group_col].nunique() < 2:
        return
    lr = multivariate_logrank_test(sub["RFS_months"], sub[group_col], sub["RFS_event"])
    fig, ax = plt.subplots(figsize=(10, 7.5))
    kmf = KaplanMeierFitter()
    cats = sorted(sub[group_col].unique())
    palette = CLUSTER_COLORS if group_col == "WGS" else CMS_COLORS if group_col == "CMS" else CRPS_COLORS if group_col == "CRPS" else ICMS_COLORS
    for cat in cats:
        m = sub[group_col].eq(cat)
        if m.sum() < 5: continue
        kmf.fit(sub.loc[m, "RFS_months"], sub.loc[m, "RFS_event"], label=f"{cat} n={m.sum()}")
        kmf.plot_survival_function(ax=ax, color=palette.get(cat, "#999999"), ci_show=False, lw=2.4)
    ax.set_xlabel("RFS (months)"); ax.set_ylabel("Probability")
    ax.set_title(f"{label} (p={lr.p_value:.3g})", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, fontsize=12)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIGDIR / f"KM_{group_col}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def endpoint_cols(endpoint):
    """ADDED: map an endpoint name to its (time, event) column pair."""
    return ("RFS_months", "RFS_event") if endpoint == "RFS" else ("OS_months", "OS_event")


def restrict_sample(df, categorical_vars, endpoint="RFS"):
    """
    Row-filtering step, kept separate from model fitting so the SAME
    restricted sample can be reused for both the reduced ("Clinical") and
    full ("Clinical + X") model in a given comparison -- see model_comparison().

    ADDED (`endpoint`): RFS keeps the original Stage I-III restriction, since
    recurrence is undefined for Stage IV. OS uses ALL stages, matching the
    manuscript's primary design. The default argument reproduces the original
    RFS behaviour exactly, so nothing downstream of the RFS path changes.
    """
    tcol, ecol = endpoint_cols(endpoint)
    if endpoint == "RFS":
        sub = df[(df["stage_num"].isin([1, 2, 3]))].copy()
    else:
        sub = df.copy()
    needed = [tcol, ecol, "age", "sex_m", "stage_num", "msi_bin", "pretreated"] + categorical_vars
    sub = sub[needed].dropna()
    for v in categorical_vars:
        sub = sub[sub[v] != "Undefined"]
    return sub


def fit_model(sub, name, categorical_vars, endpoint="RFS", stage_handling="covariate"):
    """
    Fits a Cox model on an ALREADY row-restricted sample `sub` (see
    restrict_sample). categorical_vars controls which dummy-coded
    categorical covariates are included in this specific model.

    ADDED (`endpoint`, `stage_handling`): `stage_handling="strata"` fits the
    model stratified by stage, which is what the manuscript's primary OS
    analysis does and which also sidesteps stage's known proportional-hazards
    violation. Both defaults reproduce the original RFS behaviour.
    """
    CoxPHFitter, _, _ = require_lifelines()
    tcol, ecol = endpoint_cols(endpoint)
    if len(sub) < 80 or int(sub[ecol].sum()) < 20:
        return {"model": name, "n": len(sub), "events": int(sub[ecol].sum()) if len(sub) else 0, "status": "skipped"}, None
    X = sub[[tcol, ecol, "age", "sex_m", "stage_num", "msi_bin", "pretreated"]].copy()
    for v in categorical_vars:
        d = pd.get_dummies(sub[v].astype(str), prefix=v, drop_first=True).astype(float)
        X = pd.concat([X, d], axis=1)
    covs = [c for c in X.columns if c not in {tcol, ecol}]
    for c in list(covs):
        if X[c].nunique(dropna=True) <= 1:
            X = X.drop(columns=c); covs.remove(c)
    strata = None
    if stage_handling == "strata" and "stage_num" in covs:
        strata = ["stage_num"]
        covs = [c for c in covs if c != "stage_num"]
    try:
        cph = CoxPHFitter()
        if strata:
            cph.fit(X, duration_col=tcol, event_col=ecol, strata=strata)
        else:
            cph.fit(X, duration_col=tcol, event_col=ecol)
        return {
            "model": name,
            "n": int(len(X)),
            "events": int(X[ecol].sum()),
            "df_model": len(covs),
            "log_likelihood": float(cph.log_likelihood_),
            "AIC_partial": float(cph.AIC_partial_),
            "c_index": float(cph.concordance_index_),
            "penalizer": 0.0,
            "status": "ok",
        }, cph
    except Exception as e:
        last = str(e)[:100]
        return {"model": name, "n": len(X), "events": int(X[ecol].sum()), "status": "failed: " + last}, None


def model_comparison(df):
    """
    FIX (invalid LRT): the previous version fit "Clinical" once on its own
    maximal sample and compared every augmented model's log-likelihood
    against that single baseline via a chi-squared LRT. This is invalid --
    Wilks' theorem (Wilks 1938, Ann. Math. Statist.) requires the two
    models being compared to be fit on the identical sample. "Clinical" was
    fit on ~948 Stage I-III patients (no categorical requirement), while
    e.g. "Clinical + CMS" additionally drops the 192 patients with
    CMS="Undefined" -- comparing log-likelihoods across different sample
    sizes produces a meaningless statistic, not merely an underpowered one.

    FIX: for every augmented spec, restrict_sample() is applied ONCE to get
    the exact analysis sample for that spec (row-filtered by every variable
    the augmented model needs). BOTH the reduced ("Clinical"-only) and full
    ("Clinical + X") model are then fit on that SAME restricted sample, so
    every LRT is a valid nested comparison. This means "Clinical" appears
    multiple times in the output table -- once matched to each augmented
    spec's sample -- which is intentional and required for validity.
    """
    from scipy.stats import chi2 as chi2_dist

    augmented_specs = {
        "Clinical + WGS": ["WGS"],
        "Clinical + CRPS": ["CRPS"],
        "Clinical + CMS": ["CMS"],
        "Clinical + iCMS": ["iCMS"],
        "Clinical + CRPS + WGS": ["CRPS", "WGS"],
        "Clinical + CMS + WGS": ["CMS", "WGS"],
        "Clinical + iCMS + WGS": ["iCMS", "WGS"],
    }

    rows = []
    # Reference-only row: "Clinical" on its own maximal sample (no categorical
    # requirement). NOT used in any LRT -- shown only for descriptive context.
    ref_sub = restrict_sample(df, [])
    ref_row, _ = fit_model(ref_sub, "Clinical (reference, max sample)", [])
    rows.append(ref_row)

    for name, cats in augmented_specs.items():
        print(f"Fitting {name}")
        matched_sub = restrict_sample(df, cats)
        aug_row, _ = fit_model(matched_sub, name, cats)
        clin_row, _ = fit_model(matched_sub, f"Clinical [matched to {name}]", [])
        rows.append(clin_row)
        rows.append(aug_row)

        if aug_row.get("status") == "ok" and clin_row.get("status") == "ok":
            stat = 2 * (float(aug_row["log_likelihood"]) - float(clin_row["log_likelihood"]))
            ddf = max(float(aug_row["df_model"]) - float(clin_row["df_model"]), 1)
            aug_row["LRT_stat"] = stat
            aug_row["LRT_df"] = ddf
            aug_row["LRT_vs_matched_clinical_p"] = float(chi2_dist.sf(stat, ddf))
        else:
            aug_row["LRT_vs_matched_clinical_p"] = np.nan

    out = pd.DataFrame(rows)
    out.to_csv(TABDIR / "model_comparison_rfs.csv", index=False)
    ok = out[out["LRT_vs_matched_clinical_p"].notna()] if "LRT_vs_matched_clinical_p" in out.columns else out.iloc[0:0]
    if len(ok):
        print("\nValid same-sample LRT results (each row vs its matched Clinical-only model):")
        print(ok[["model", "n", "events", "LRT_vs_matched_clinical_p"]].to_string(index=False))


def model_comparison_os(df):
    """
    ADDED — analysis E. The overall-survival counterpart of model_comparison().

    model_comparison() is deliberately left untouched; this function reuses
    the same restrict_sample()/fit_model() helpers through their new endpoint
    arguments, so the two arms share one implementation and cannot drift.

    Why this exists. The manuscript's prognostic claim rests on OS, and C4 is
    enriched for CMS4 (35.1% vs 26.1% cohort-wide). A referee will therefore
    ask whether C4 is a genomic restatement of CMS4. Only a nested OS model
    containing BOTH CMS and the WGS subtypes can answer that, in two parts:
      (i)  the likelihood-ratio test for adding the 4-level WGS factor to a
           clinical + CMS model -- does the taxonomy add anything at all;
      (ii) the C4-vs-C1 hazard ratio WITHIN that model -- does the specific
           C4 association survive adjustment for CMS.
    Part (ii) is why every model's coefficients are dumped, not just its fit
    statistics.

    Every LRT here is a valid nested comparison: both models in a pair are
    fit on the identical restricted sample, exactly as in model_comparison().
    """
    from scipy.stats import chi2 as chi2_dist

    augmented_specs = {
        "Clinical + WGS": ["WGS"],
        "Clinical + CRPS": ["CRPS"],
        "Clinical + CMS": ["CMS"],
        "Clinical + iCMS": ["iCMS"],
        "Clinical + CRPS + WGS": ["CRPS", "WGS"],
        "Clinical + CMS + WGS": ["CMS", "WGS"],
        "Clinical + iCMS + WGS": ["iCMS", "WGS"],
    }
    # Head-to-head pairs, both directions. The first three are the analysis
    # that decides the paper: does the WGS taxonomy add prognostic information
    # ON TOP OF each expression classifier? The second three are its mirror --
    # does the classifier add anything on top of the taxonomy? Reporting only
    # one direction would judge the taxonomy by a standard its comparators
    # were never held to, or vice versa.
    reverse_specs = {
        "Clinical + CMS + WGS  [vs Clinical + CMS]":  (["CMS", "WGS"],  ["CMS"]),
        "Clinical + CRPS + WGS [vs Clinical + CRPS]": (["CRPS", "WGS"], ["CRPS"]),
        "Clinical + iCMS + WGS [vs Clinical + iCMS]": (["iCMS", "WGS"], ["iCMS"]),
        "Clinical + WGS + CMS  [vs Clinical + WGS]":  (["WGS", "CMS"],  ["WGS"]),
        "Clinical + WGS + CRPS [vs Clinical + WGS]":  (["WGS", "CRPS"], ["WGS"]),
        "Clinical + WGS + iCMS [vs Clinical + WGS]":  (["WGS", "iCMS"], ["WGS"]),
    }

    rows, coef_rows = [], []

    def _record_coefs(cph, model_name, stage_handling, n, events):
        if cph is None:
            return
        # lifelines' summary column names contain spaces and '%', so index by
        # label rather than via itertuples (which mangles them into valid
        # Python identifiers and then cannot find them).
        s = cph.summary
        lo = "coef lower 95%" if "coef lower 95%" in s.columns else "coef_lower_95"
        hi = "coef upper 95%" if "coef upper 95%" in s.columns else "coef_upper_95"
        for cov in s.index:
            coef_rows.append({
                "endpoint": "OS", "stage_handling": stage_handling,
                "model": model_name, "n": n, "events": events,
                "covariate": str(cov),
                "HR": float(np.exp(s.loc[cov, "coef"])),
                "HR_lower95": float(np.exp(s.loc[cov, lo])),
                "HR_upper95": float(np.exp(s.loc[cov, hi])),
                "p": float(s.loc[cov, "p"]),
            })

    for stage_handling in ["covariate", "strata"]:
        print(f"\n=== OS model comparison (stage as {stage_handling}) ===")
        ref_sub = restrict_sample(df, [], endpoint="OS")
        ref_row, ref_cph = fit_model(ref_sub, "Clinical (reference, max sample)", [],
                                     endpoint="OS", stage_handling=stage_handling)
        ref_row["stage_handling"] = stage_handling
        ref_row["comparison"] = "reference only (not used in any LRT)"
        rows.append(ref_row)
        _record_coefs(ref_cph, "Clinical (reference, max sample)", stage_handling,
                      ref_row.get("n"), ref_row.get("events"))

        for name, cats in augmented_specs.items():
            print(f"Fitting {name}")
            matched_sub = restrict_sample(df, cats, endpoint="OS")
            aug_row, aug_cph = fit_model(matched_sub, name, cats,
                                         endpoint="OS", stage_handling=stage_handling)
            clin_row, clin_cph = fit_model(matched_sub, f"Clinical [matched to {name}]", [],
                                           endpoint="OS", stage_handling=stage_handling)
            for r, nm in ((clin_row, f"Clinical [matched to {name}]"), (aug_row, name)):
                r["stage_handling"] = stage_handling
                r["comparison"] = f"{name} vs matched Clinical"
            rows.append(clin_row)
            rows.append(aug_row)
            _record_coefs(aug_cph, name, stage_handling, aug_row.get("n"), aug_row.get("events"))

            if aug_row.get("status") == "ok" and clin_row.get("status") == "ok":
                stat = 2 * (float(aug_row["log_likelihood"]) - float(clin_row["log_likelihood"]))
                ddf = max(float(aug_row["df_model"]) - float(clin_row["df_model"]), 1)
                aug_row["LRT_stat"] = stat
                aug_row["LRT_df"] = ddf
                aug_row["LRT_vs_matched_clinical_p"] = float(chi2_dist.sf(stat, ddf))
            else:
                aug_row["LRT_vs_matched_clinical_p"] = np.nan

        for name, (full_cats, red_cats) in reverse_specs.items():
            print(f"Fitting {name}")
            matched_sub = restrict_sample(df, full_cats, endpoint="OS")
            full_row, full_cph = fit_model(matched_sub, name, full_cats,
                                           endpoint="OS", stage_handling=stage_handling)
            red_row, _ = fit_model(matched_sub, f"[reduced] {'+'.join(red_cats)}", red_cats,
                                   endpoint="OS", stage_handling=stage_handling)
            for r in (red_row, full_row):
                r["stage_handling"] = stage_handling
                r["comparison"] = name
            rows.append(red_row)
            rows.append(full_row)
            _record_coefs(full_cph, name, stage_handling, full_row.get("n"), full_row.get("events"))
            if full_row.get("status") == "ok" and red_row.get("status") == "ok":
                stat = 2 * (float(full_row["log_likelihood"]) - float(red_row["log_likelihood"]))
                ddf = max(float(full_row["df_model"]) - float(red_row["df_model"]), 1)
                full_row["LRT_stat"] = stat
                full_row["LRT_df"] = ddf
                full_row["LRT_vs_matched_clinical_p"] = float(chi2_dist.sf(stat, ddf))
            else:
                full_row["LRT_vs_matched_clinical_p"] = np.nan

    out = pd.DataFrame(rows)
    out.to_csv(TABDIR / "model_comparison_os.csv", index=False)
    coefs = pd.DataFrame(coef_rows)
    coefs.to_csv(TABDIR / "model_coefficients_os.csv", index=False)

    ok = out[out["LRT_vs_matched_clinical_p"].notna()] if "LRT_vs_matched_clinical_p" in out.columns else out.iloc[0:0]
    if len(ok):
        print("\nOS: valid same-sample LRT results")
        print(ok[["stage_handling", "model", "n", "events", "c_index",
                  "LRT_vs_matched_clinical_p"]].to_string(index=False))
    if len(coefs):
        c4 = coefs[coefs["covariate"].astype(str).str.contains("WGS_C4")]
        if len(c4):
            print("\nOS: C4-vs-C1 hazard ratio in every model that contains the WGS factor")
            print("(the row for 'Clinical + CMS + WGS' is the CMS-adjusted C4 estimate)")
            print(c4[["stage_handling", "model", "n", "events",
                      "HR", "HR_lower95", "HR_upper95", "p"]].to_string(index=False))


def main():
    df = load_all()
    print(f"Merged samples: {len(df)}")
    print("WGS counts:", df["WGS"].value_counts().sort_index().to_dict())
    concordance_metrics(df)
    coverage, _ = assignment_coverage(df)
    print("\nComparator callability and WGS assignment of undefined cases:")
    print(coverage.to_string(index=False))
    alluvial_crps_wgs_cms(df)
    icms_stacked(df)
    for col, label in [("WGS", "WGS"), ("CRPS", "CRPS"), ("CMS", "CMS"), ("iCMS", "iCMS")]:
        plot_km_group(df, col, label)
    model_comparison(df)
    model_comparison_os(df)  # ADDED — analysis E, primary endpoint
    print(f"Done. Outputs: {OUTDIR}")


if __name__ == "__main__":
    main()
