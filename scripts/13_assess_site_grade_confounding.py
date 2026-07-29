#!/usr/bin/env python3
"""Test whether primary tumour site and grade explain the C4 mortality signal.

Stage-stratified C4-versus-C1 Cox models are fitted from the common complete-case
cohort with site and grade added separately and together.
"""
import os
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

BASE = Path(os.environ.get("CRC_BASE", Path(__file__).resolve().parents[1])).resolve()
LOCKED = BASE / "module05_06_loocv_results" / "labels" / "NMF_k4_LOOCV.csv"
SUPP = BASE / "crc_heterogeneity_data" / "Supplementary_Table_01.xlsx"
OUT = BASE / "module26b_site_grade_adjusted_cox"
(OUT / "tables").mkdir(parents=True, exist_ok=True)


def extract_id(x):
    m = re.search(r"((?:U|UM)\d+)", str(x))
    return m.group(1) if m else None


lab = pd.read_csv(LOCKED)
lab["sid"] = lab["sample_id"].map(extract_id)
cl = pd.to_numeric(lab["cluster"], errors="coerce").astype(int)
lab["WGS"] = "C" + (cl if cl.min() >= 1 else cl + 1).astype(str)
lab = lab[["sid", "WGS"]]
print(f"Locked labels: {lab.WGS.value_counts().sort_index().to_dict()}")

st = pd.read_excel(SUPP)
scol = "DNA Tumor Sample Barcode" if "DNA Tumor Sample Barcode" in st.columns else st.columns[0]
st["sid"] = st[scol].map(extract_id)
df = pd.DataFrame({"sid": st["sid"]})
df["age"] = pd.to_numeric(st["Age at diagnosis"], errors="coerce")
df["sex_m"] = st["Sex"].astype(str).str.lower().str[0].map({"m": 1.0, "f": 0.0})
df["msi_bin"] = st["MSI Status"].astype(str).str.upper().str.contains("MSI").astype(float)
df["pretreated"] = st["Pre-Treated"].astype(str).str.lower().isin(["yes", "1", "true"]).astype(float)
df["stage_num"] = st["Tumour Stage"].astype(str).str.extract(r"Stage\s*(IV|III|II|I)")[0].map(
    {"I": 1.0, "II": 2.0, "III": 3.0, "IV": 4.0})
df["site"] = st["Tumour Site"].astype(str)
df["grade"] = st["Tumour Grade"].astype(str)
df["OS_months"] = pd.to_numeric(st["Overall survival days"], errors="coerce") / 30.44
df["OS_event"] = st["Vital Status"].astype(str).str.lower().map({"dead": 1.0, "alive": 0.0})

df = df.merge(lab, on="sid", how="inner")
d = df[df.WGS.isin(["C1", "C4"])].copy()
d["C4"] = (d.WGS == "C4").astype(float)
print(f"C1 vs C4 merged cohort: {len(d)}  (C1={int((d.WGS=='C1').sum())}, C4={int((d.WGS=='C4').sum())})")

site_dist = d.groupby("WGS")["site"].value_counts(normalize=True).unstack().round(3) * 100
grade_dist = d.groupby("WGS")["grade"].value_counts(normalize=True).unstack().round(3) * 100
print("\nSite distribution, C1 vs C4 (%):"); print(site_dist)
print("\nGrade distribution, C1 vs C4 (%):"); print(grade_dist)


def fit(extra_covs, name):
    cols = ["OS_months", "OS_event", "C4", "age", "sex_m", "stage_num"] + extra_covs
    dd = d[cols].dropna()
    dd = dd[dd.OS_months > 0]
    Xcols = ["OS_months", "OS_event", "C4", "age"]
    model_df = dd[Xcols].copy()
    for c in ["sex_m"] + extra_covs:
        if c in ("site", "grade"):
            dummies = pd.get_dummies(dd[c], prefix=c, drop_first=True).astype(float)
            model_df = pd.concat([model_df, dummies], axis=1)
        else:
            model_df[c] = dd[c]
    model_df["stage_num"] = dd["stage_num"]
    # Near-constant binary covariates (e.g. msi_bin: C1/C4 are 99.3%/95.7% MSS)
    # can stall Newton-Raphson in this small a contrast; drop them here rather
    # than lose the whole model, matching Module 26's approach.
    drop = [c for c in model_df.columns
            if c not in {"OS_months", "OS_event", "C4", "stage_num"}
            and (model_df[c].nunique() < 2
                 or (model_df[c].isin([0, 1]).all()
                     and min(model_df[c].mean(), 1 - model_df[c].mean()) < 0.02))]
    model_df = model_df.drop(columns=drop)
    cph = None
    for pen in (0.0, 0.1, 0.5):
        try:
            cph = CoxPHFitter(penalizer=pen)
            cph.fit(model_df, duration_col="OS_months", event_col="OS_event", strata=["stage_num"])
            break
        except Exception:
            cph = None
    if cph is None:
        return {"model": name, "n": int(len(model_df)), "events": int(model_df.OS_event.sum()),
                "HR_C4_vs_C1": np.nan, "HR_lower95": np.nan, "HR_upper95": np.nan,
                "p": np.nan, "c_index": np.nan, "dropped": "+".join(drop)}
    s = cph.summary
    lo = "coef lower 95%" if "coef lower 95%" in s.columns else "coef_lower_95"
    hi = "coef upper 95%" if "coef upper 95%" in s.columns else "coef_upper_95"
    return {"model": name, "n": int(len(model_df)), "events": int(model_df.OS_event.sum()),
            "HR_C4_vs_C1": float(np.exp(s.loc["C4", "coef"])),
            "HR_lower95": float(np.exp(s.loc["C4", lo])),
            "HR_upper95": float(np.exp(s.loc["C4", hi])),
            "p": float(s.loc["C4", "p"]), "c_index": float(cph.concordance_index_),
            "dropped": "+".join(drop) if drop else "none"}


rows = [
    fit([], "base (age, sex; stage-stratified)"),
    fit(["site"], "+ primary site (right/left/rectum)"),
    fit(["grade"], "+ grade"),
    fit(["site", "grade"], "+ site + grade"),
    fit(["msi_bin", "pretreated"], "+ MSI + pre-treatment (matches primary manuscript model)"),
    fit(["msi_bin", "pretreated", "site", "grade"], "+ MSI + pre-treatment + site + grade (fullest model)"),
]
out = pd.DataFrame(rows)
out.to_csv(OUT / "tables" / "cox_C4_vs_C1_site_grade_adjusted.csv", index=False)
print("\n" + out.to_string(index=False))
print(f"\n  -> {OUT / 'tables' / 'cox_C4_vs_C1_site_grade_adjusted.csv'}")
