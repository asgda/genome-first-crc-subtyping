#!/usr/bin/env python3
"""Evaluate subtype reconstruction after dropping genomic modalities.

All seven non-empty SNV, CNV and SV combinations are assessed by repeated
cross-fitting, held-out assignment and patient-paired bootstrap confidence
intervals. ARI measures the complete partition and Jaccard measures C4 recovery.
"""

import os
import re
import zipfile
import warnings
import xml.etree.ElementTree as ET
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")

BASE = os.environ.get("CRC_BASE", str(Path(__file__).resolve().parents[1]))
M1 = os.environ.get("CRC_M1_MATRIX", f"{BASE}/module1_results/module1_discovery_binary_matrix.csv")
M2 = os.environ.get("CRC_M2_MATRIX", f"{BASE}/module2_results/module2_discovery_binary_matrix.csv")
M3 = os.environ.get("CRC_M3_MATRIX", f"{BASE}/module3_results/module3_discovery_binary_matrix.csv")
M4 = os.environ.get("CRC_FEATURE_MATRIX", f"{BASE}/module4_results/module4_unified_discovery_matrix.csv")
SUPP_TABLE = os.environ.get("CRC_SUPP_TABLE", f"{BASE}/crc_heterogeneity_data/Supplementary_Table_01.xlsx")
CLUSTER_FILE = os.environ.get("CRC_CLUSTER_FILE", f"{BASE}/module05_06_loocv_results/labels/NMF_k4_LOOCV.csv")
OUTDIR = Path(os.environ.get("CRC_M20_OUT", f"{BASE}/module20_modality_dropout_robustness_C1C4_ppt"))
FIGDIR = OUTDIR / "figures"
TABDIR = OUTDIR / "tables"
FIGDIR.mkdir(parents=True, exist_ok=True)
TABDIR.mkdir(parents=True, exist_ok=True)

N_REPEATS = int(os.environ.get("CRC_M20_N_REPEATS", "30"))
N_COMPONENTS = int(os.environ.get("CRC_M20_K", "4"))
MAX_ITER = int(os.environ.get("CRC_M20_NMF_MAXITER", "600"))
MASTER_SEED = int(os.environ.get("CRC_M20_SEED", "42"))
CROSSFIT_REPEATS = int(os.environ.get("CRC_M20_CROSSFIT_REPEATS", "10"))
CROSSFIT_FOLDS = int(os.environ.get("CRC_M20_CROSSFIT_FOLDS", "5"))
BOOTSTRAPS = int(os.environ.get("CRC_M20_BOOTSTRAPS", "2000"))

CB8 = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999"]
MODALITY_COLORS = {
    "SNV": "#E69F00", "CNV": "#0072B2", "SV": "#009E73",
    "SNV+CNV": "#56B4E9", "SNV+SV": "#CC79A7", "CNV+SV": "#D55E00", "ALL": "#000000",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 18,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 16,
    "ytick.labelsize": 18,
    "legend.fontsize": 15,
    "savefig.dpi": 300,
    "axes.linewidth": 1.1,
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


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
    return f"C{int(c)+1}"


def read_xlsx_first_sheet(path):
    """Read the first XLSX worksheet without an openpyxl dependency."""
    ns_main = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    ns_rel = "{http://schemas.openxmlformats.org/package/2006/relationships}"

    def cell_text(cell, shared):
        typ = cell.attrib.get("t")
        if typ == "s":
            v = cell.find(f"{ns_main}v")
            return shared[int(v.text)] if v is not None and v.text is not None else ""
        if typ == "inlineStr":
            return "".join(t.text or "" for t in cell.findall(f".//{ns_main}t"))
        v = cell.find(f"{ns_main}v")
        return "" if v is None or v.text is None else v.text

    def col_idx(ref):
        letters = re.match(r"([A-Z]+)", ref).group(1)
        value = 0
        for ch in letters:
            value = value * 26 + ord(ch) - 64
        return value - 1

    with zipfile.ZipFile(path) as zf:
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        relmap = {x.attrib["Id"]: x.attrib["Target"] for x in rels.findall(f"{ns_rel}Relationship")}
        sheet = wb.find(f"{ns_main}sheets").find(f"{ns_main}sheet")
        rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        target = "xl/" + relmap[rid].lstrip("/")
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.findall(f".//{ns_main}t"))
                      for si in root.findall(f"{ns_main}si")]
        root = ET.fromstring(zf.read(target))
        rows = []
        for row in root.findall(f".//{ns_main}row"):
            values = {col_idx(c.attrib.get("r", "A1")): cell_text(c, shared)
                      for c in row.findall(f"{ns_main}c")}
            rows.append([values.get(i, "") for i in range(max(values, default=-1) + 1)])
    if not rows:
        return pd.DataFrame()
    width = max(map(len, rows))
    header = [str(x).strip() for x in rows[0]] + [f"Unnamed: {i}" for i in range(len(rows[0]), width)]
    return pd.DataFrame([r + [""] * (width - len(r)) for r in rows[1:]], columns=header)


def read_matrix(path, prefix=None):
    if not Path(path).exists():
        raise FileNotFoundError(f"Matrix not found: {path}")
    df = pd.read_csv(path, index_col=0, low_memory=False)
    df.index = df.index.astype(str).map(normalize_sample)
    df = df.loc[df.index.notna()]
    df = df.loc[~df.index.duplicated(keep="first")]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        df[c] = (df[c] > 0).astype(np.float32)
    if prefix:
        df.columns = [f"{prefix}_{str(c).replace(prefix + '_', '')}" for c in df.columns]
    return df


def read_labels(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Cluster file not found: {path}. Set CRC_CLUSTER_FILE.")
    lab = pd.read_csv(path)
    lab["sid"] = lab["sample_id"].map(normalize_sample)
    lab = lab.dropna(subset=["sid"]).drop_duplicates("sid")
    lab["cluster"] = normalize_cluster_values(lab["cluster"])
    lab["cluster_display"] = lab["cluster"].map(display_cluster)
    return lab.set_index("sid")[["cluster", "cluster_display"]]


def align_matrices(mats, labels):
    common = set(labels.index)
    for df in mats.values():
        common &= set(df.index)
    common = sorted(common)
    if len(common) < 100:
        raise ValueError(f"Too few common samples across matrices and labels: {len(common)}")
    mats = {k: v.loc[common].copy() for k, v in mats.items()}
    labels = labels.loc[common].copy()
    return mats, labels


def nmf_labels(X, seed):
    model = NMF(
        n_components=N_COMPONENTS,
        init="nndsvda",
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=MAX_ITER,
        tol=1e-4,
        random_state=seed,
    )
    W = model.fit_transform(X + 1e-6)
    lab = np.argmax(W, axis=1).astype(int)
    return lab, float(model.reconstruction_err_)


def c4_recovery(pred, true):
    true_c4 = true == 3
    rows = []
    for k in sorted(np.unique(pred)):
        p = pred == k
        inter = int(np.logical_and(p, true_c4).sum())
        union = int(np.logical_or(p, true_c4).sum())
        precision = inter / max(int(p.sum()), 1)
        recall = inter / max(int(true_c4.sum()), 1)
        jaccard = inter / max(union, 1)
        rows.append((k, jaccard, precision, recall, int(p.sum()), inter))
    best = max(rows, key=lambda x: x[1])
    return {"best_pred_cluster_for_C4": int(best[0]), "C4_jaccard": best[1], "C4_precision": best[2], "C4_recall": best[3], "matched_cluster_size": best[4], "C4_overlap_n": best[5]}


_SURVIVAL_TABLE_CACHE = None


def try_survival_p(samples, labels, endpoint="RFS"):
    """Log-rank p-value for a derived partition. endpoint="RFS" (default,
    behaviour unchanged from before) restricts to Stage I-III, since Stage IV
    patients never had a disease-free state to recur from and RFS is
    undefined for them ("Not_Applicable" in the source table). endpoint="OS"
    (added 2026-07-15) uses Vital Status / Overall survival days instead and
    applies no stage restriction, since OS is a valid endpoint for every
    patient regardless of stage."""
    if not Path(SUPP_TABLE).exists():
        return np.nan
    try:
        global _SURVIVAL_TABLE_CACHE
        from lifelines.statistics import multivariate_logrank_test
        if _SURVIVAL_TABLE_CACHE is None:
            try:
                _SURVIVAL_TABLE_CACHE = pd.read_excel(SUPP_TABLE)
            except ImportError:
                _SURVIVAL_TABLE_CACHE = read_xlsx_first_sheet(SUPP_TABLE)
        st = _SURVIVAL_TABLE_CACHE
        sample_col = next((c for c in st.columns if "Sample" in str(c) or "Barcode" in str(c)), st.columns[0])
        st["sid"] = st[sample_col].map(normalize_sample)
        stage_col = next((c for c in st.columns if "Stage" in str(c)), None)

        if endpoint == "RFS":
            time_col = next((c for c in st.columns if "Recurrence free survival days" in str(c)), None)
            event_col = next((c for c in st.columns if str(c).strip().lower() == "recurrence"), None)
            if not (time_col and event_col and stage_col):
                return np.nan
        elif endpoint == "OS":
            time_col = next((c for c in st.columns if "Overall survival days" in str(c)), None)
            event_col = next((c for c in st.columns if str(c).strip().lower() == "vital status"), None)
            if not (time_col and event_col):
                return np.nan
        else:
            raise ValueError(f"Unknown endpoint: {endpoint!r}, expected 'RFS' or 'OS'")

        d = pd.DataFrame({"sid": samples, "label": labels})
        merge_cols = ["sid", time_col, event_col] + ([stage_col] if endpoint == "RFS" else [])
        m = d.merge(st[merge_cols], on="sid", how="left")
        m["time"] = pd.to_numeric(m[time_col], errors="coerce") / 30.44

        if endpoint == "RFS":
            m["event"] = np.where(m[event_col].astype(str).str.lower().eq("yes"), 1, 0)
            # FIX: the previous filter used stage.str.contains("Stage I|Stage II|Stage III"),
            # an unanchored regex where the "Stage I" alternative matches as a literal
            # substring PREFIX of "Stage IV" (re.search confirmed this: "Stage IV"
            # matches the pattern too). This was masked only by coincidence -- Stage IV
            # rows have RFS_days = NaN (from "Not_Applicable" failing numeric coercion)
            # and were dropped by the later .dropna(). Fixed with an explicit numeric
            # stage map, consistent with the stage_num convention used in every other
            # module in this pipeline (14, 18, 19, 21).
            stage_map = {"Stage I": 1, "Stage II": 2, "Stage III": 3, "Stage IV": 4}
            m["stage_num"] = m[stage_col].astype(str).str.strip().map(stage_map)
            m = m[m["stage_num"].isin([1, 2, 3])]
        else:
            # OS: no stage restriction -- valid for every patient including Stage IV.
            m["event"] = np.where(m[event_col].astype(str).str.lower().eq("dead"), 1, 0)

        m = m.dropna(subset=["time", "event", "label"])
        if len(m) < 100 or m["event"].sum() < 20 or m["label"].nunique() < 2:
            return np.nan
        lr = multivariate_logrank_test(m["time"], m["label"], m["event"])
        return float(lr.p_value)
    except Exception:
        return np.nan


def run_dropout(mats, labels):
    combos = {
        "SNV": mats["SNV"],
        "CNV": mats["CNV"],
        "SV": mats["SV"],
        "SNV+CNV": pd.concat([mats["SNV"], mats["CNV"]], axis=1),
        "SNV+SV": pd.concat([mats["SNV"], mats["SV"]], axis=1),
        "CNV+SV": pd.concat([mats["CNV"], mats["SV"]], axis=1),
        "ALL": mats["ALL"],
    }
    ytrue = labels["cluster"].values.astype(int)
    rows = []
    samples = labels.index.tolist()
    for name, df in combos.items():
        X = df.values.astype(np.float32)
        print(f"Running {name}: {X.shape}")
        for r in range(N_REPEATS):
            seed = MASTER_SEED + 1000 * (list(combos).index(name) + 1) + r
            try:
                ypred, err = nmf_labels(X, seed)
                rec = c4_recovery(ypred, ytrue)
                rows.append({
                    "input": name,
                    "repeat": r,
                    "n_samples": X.shape[0],
                    "n_features": X.shape[1],
                    "reconstruction_error": err,
                    "ARI_vs_final": adjusted_rand_score(ytrue, ypred),
                    "NMI_vs_final": normalized_mutual_info_score(ytrue, ypred),
                    "survival_logrank_p": try_survival_p(samples, ypred) if r == 0 else np.nan,
                    "survival_logrank_p_OS": try_survival_p(samples, ypred, endpoint="OS") if r == 0 else np.nan,
                    **rec,
                    "status": "ok",
                })
            except Exception as e:
                rows.append({"input": name, "repeat": r, "status": "failed: " + str(e)[:100]})
    return pd.DataFrame(rows)


def summarize(res):
    ok = res[res["status"] == "ok"].copy()
    agg_cols = ["ARI_vs_final", "NMI_vs_final", "C4_jaccard", "C4_precision", "C4_recall"]
    summ = ok.groupby("input").agg({c: ["mean", "std", "median", "max"] for c in agg_cols})
    summ.columns = ["_".join(x) for x in summ.columns]
    # Carry first-repeat survival p (RFS, unchanged; OS, added 2026-07-15).
    surv = ok.dropna(subset=["survival_logrank_p"]).groupby("input")["survival_logrank_p"].first()
    surv_os = ok.dropna(subset=["survival_logrank_p_OS"]).groupby("input")["survival_logrank_p_OS"].first()
    summ = summ.join(surv, how="left").join(surv_os, how="left")
    order = ["SNV", "CNV", "SV", "SNV+CNV", "SNV+SV", "CNV+SV", "ALL"]
    summ = summ.reindex(order)
    return summ.reset_index()


def plot_metric(summary, metric, ylabel, fname, logy=False):
    order = [x for x in ["SNV", "CNV", "SV", "SNV+CNV", "SNV+SV", "CNV+SV", "ALL"] if x in set(summary["input"])]
    s = summary.set_index("input").loc[order]
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    x = np.arange(len(order))
    vals = s[metric].values
    colors = [MODALITY_COLORS[o] for o in order]
    ax.bar(x, vals, color=colors, edgecolor="black", linewidth=1.0)
    for xi, v in zip(x, vals):
        if np.isfinite(v):
            ax.text(xi, v + (0.015 if not logy else 0), f"{v:.3g}", ha="center", va="bottom", fontsize=12, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=30, ha="right", fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel, fontweight="bold")
    if logy:
        ax.set_yscale("log")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIGDIR / f"{fname}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def add_fdr_column_wide(df, pcol, qcol):
    """Benjamini-Hochberg FDR correction (Benjamini and Hochberg, 1995),
    added 2026-07-16 as a permanent output column, matching the convention
    established in Module 22's ablation_survival_summary.csv. Unlike
    Module 18/22's long-format tables (one row per arm, corrected within an
    endpoint group), this table is already wide -- one row per modality,
    RFS and OS p-values in separate columns -- so each p-value column IS
    already a complete family of seven simultaneously tested hypotheses on
    its own; no additional grouping is needed."""
    df[qcol] = np.nan
    mask = df[pcol].notna()
    if mask.sum() > 0:
        _, qvals, _, _ = multipletests(df.loc[mask, pcol], alpha=0.05, method="fdr_bh")
        df.loc[mask, qcol] = qvals
    return df


def modality_combinations(mats):
    """Every non-empty modality subset, using the locked candidate universe."""
    return {
        "SNV": mats["SNV"],
        "CNV": mats["CNV"],
        "SV": mats["SV"],
        "SNV+CNV": pd.concat([mats["SNV"], mats["CNV"]], axis=1),
        "SNV+SV": pd.concat([mats["SNV"], mats["SV"]], axis=1),
        "CNV+SV": pd.concat([mats["CNV"], mats["SV"]], axis=1),
        "ALL": mats["ALL"],
    }


def select_features_in_training(Xdf, train_idx):
    """Reapply data-dependent recurrence filtering inside the training fold.

    The candidate universe remains the prespecified, passenger-filtered CRC
    discovery universe.  Only the prevalence filter is sample-dependent and
    is therefore repeated without access to held-out samples: 2%-98% for
    SNV/CNV gene events, 1%-98% for SV events.  CNV arm features are retained
    as in the discovery pipeline, where chromosome arms were included as
    prespecified CIN measurements rather than recurrence-selected genes.
    """
    tr = Xdf.iloc[train_idx]
    freq = tr.mean(axis=0)
    keep = []
    for c in Xdf.columns:
        s = str(c)
        if re.match(r"^CNV_(?:[0-9]{1,2}|X|Y)[pq]$", s):
            keep.append(c)
            continue
        lower = 0.01 if s.startswith("SV_") else 0.02
        if lower <= float(freq[c]) <= 0.98:
            keep.append(c)
    if len(keep) < N_COMPONENTS:
        # Defensive only: preserve a valid factorisation for exceptionally
        # sparse resamples, selecting the most variable training features.
        keep = tr.var(axis=0).sort_values(ascending=False).head(N_COMPONENTS).index.tolist()
    return keep


def map_components_on_training(component_labels, reference_labels):
    """One-to-one component mapping learned only from the training fold."""
    ct = np.zeros((N_COMPONENTS, N_COMPONENTS), dtype=int)
    for a, b in zip(component_labels, reference_labels):
        if 0 <= int(a) < N_COMPONENTS and 0 <= int(b) < N_COMPONENTS:
            ct[int(a), int(b)] += 1
    rows, cols = linear_sum_assignment(-ct)
    mapping = {int(r): int(c) for r, c in zip(rows, cols)}
    return mapping


def partition_metrics(y_true, y_pred):
    true_c4 = np.asarray(y_true) == 3
    pred_c4 = np.asarray(y_pred) == 3
    tp = int(np.logical_and(true_c4, pred_c4).sum())
    fp = int(np.logical_and(~true_c4, pred_c4).sum())
    fn = int(np.logical_and(true_c4, ~pred_c4).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    jaccard = tp / max(tp + fp + fn, 1)
    return {
        "ARI_vs_final": adjusted_rand_score(y_true, y_pred),
        "NMI_vs_final": normalized_mutual_info_score(y_true, y_pred),
        "C4_precision": precision,
        "C4_recall": recall,
        "C4_f1": f1,
        "C4_jaccard": jaccard,
        "C4_true_positive_n": tp,
        "C4_predicted_n": int(pred_c4.sum()),
    }


def majority_consensus(predictions, membership_scores):
    """Per-patient vote, with label-order-free NMF-strength tie breaking.

    Membership scores are transformed held-out NMF weights mapped using the
    training-fold permutation and row-normalised before aggregation.  They do
    not use the held-out reference label.
    """
    predictions = np.asarray(predictions, dtype=int)
    membership_scores = np.asarray(membership_scores, dtype=float)
    out = np.empty(predictions.shape[1], dtype=int)
    ties = 0
    for i in range(predictions.shape[1]):
        counts = np.bincount(predictions[:, i], minlength=N_COMPONENTS)
        winners = np.flatnonzero(counts == counts.max())
        if len(winners) > 1:
            ties += 1
            strength = membership_scores[:, i, :].mean(axis=0)
            out[i] = int(winners[np.argmax(strength[winners])])
        else:
            out[i] = int(winners[0])
    return out, ties


def bootstrap_metric_intervals(y_true, y_pred, n_boot=BOOTSTRAPS):
    """Paired patient-level bootstrap CIs for cross-fitted performance."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    rng = np.random.default_rng(MASTER_SEED + 90909)
    keys = ["ARI_vs_final", "NMI_vs_final", "C4_precision", "C4_recall", "C4_f1", "C4_jaccard"]
    vals = {k: [] for k in keys}
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = partition_metrics(y_true[idx], y_pred[idx])
        for k in keys:
            vals[k].append(m[k])
    return {
        f"{k}_{suffix}": float(np.quantile(vals[k], q))
        for k in keys
        for suffix, q in [("ci_low", 0.025), ("ci_high", 0.975)]
    }


def run_crossfitted_dropout(mats, labels):
    """Out-of-sample recovery with no held-out-label cluster matching.

    For each repeated split, NMF is fitted only on the training fold.  The
    component-to-C1/C4 permutation is learned from training labels via a
    Hungarian assignment and then frozen before transforming the held-out
    fold.  Consequently every reported prediction is genuinely out of fold.
    """
    combos = modality_combinations(mats)
    y = labels["cluster"].to_numpy(dtype=int)
    samples = labels.index.to_numpy()
    repeat_rows = []
    feature_rows = []
    consensus_rows = []
    prediction_table = pd.DataFrame({"sid": samples, "reference": y})

    for combo_i, (name, Xdf) in enumerate(combos.items()):
        print(f"Cross-fitting {name}: {Xdf.shape}", flush=True)
        repeated_predictions = []
        repeated_memberships = []
        for repeat in range(CROSSFIT_REPEATS):
            seed = MASTER_SEED + 10000 * (combo_i + 1) + repeat
            skf = StratifiedKFold(n_splits=CROSSFIT_FOLDS, shuffle=True, random_state=seed)
            oof = np.full(len(y), -1, dtype=int)
            oof_membership = np.zeros((len(y), N_COMPONENTS), dtype=float)
            for fold, (train_idx, test_idx) in enumerate(skf.split(Xdf, y)):
                selected = select_features_in_training(Xdf, train_idx)
                Xtr = Xdf.iloc[train_idx][selected].to_numpy(dtype=np.float32) + 1e-6
                Xte = Xdf.iloc[test_idx][selected].to_numpy(dtype=np.float32) + 1e-6
                model = NMF(
                    n_components=N_COMPONENTS, init="nndsvda", solver="mu",
                    beta_loss="kullback-leibler", max_iter=MAX_ITER, tol=1e-4,
                    random_state=seed + fold,
                )
                Wtr = model.fit_transform(Xtr)
                train_components = np.argmax(Wtr, axis=1).astype(int)
                mapping = map_components_on_training(train_components, y[train_idx])
                Wte = model.transform(Xte)
                test_components = np.argmax(Wte, axis=1).astype(int)
                oof[test_idx] = np.array([mapping[int(c)] for c in test_components], dtype=int)
                mapped_weights = np.zeros_like(Wte, dtype=float)
                for component, subtype in mapping.items():
                    mapped_weights[:, subtype] = Wte[:, component]
                mapped_weights /= np.maximum(mapped_weights.sum(axis=1, keepdims=True), 1e-12)
                oof_membership[test_idx] = mapped_weights
                feature_rows.append({
                    "input": name, "repeat": repeat, "fold": fold,
                    "n_train": len(train_idx), "n_test": len(test_idx),
                    "n_candidate_features": Xdf.shape[1],
                    "n_selected_features": len(selected),
                })
            if np.any(oof < 0):
                raise RuntimeError(f"Incomplete out-of-fold predictions for {name}, repeat {repeat}")
            repeated_predictions.append(oof)
            repeated_memberships.append(oof_membership)
            repeat_rows.append({
                "input": name, "repeat": repeat,
                **partition_metrics(y, oof),
                "survival_logrank_p": try_survival_p(samples, oof, endpoint="RFS"),
                "survival_logrank_p_OS": try_survival_p(samples, oof, endpoint="OS"),
            })

        consensus, ties = majority_consensus(repeated_predictions, repeated_memberships)
        prediction_table[f"{name}_crossfit_consensus"] = consensus
        primary = partition_metrics(y, consensus)
        cis = bootstrap_metric_intervals(y, consensus)
        rr = pd.DataFrame([r for r in repeat_rows if r["input"] == name])
        fr = pd.DataFrame([r for r in feature_rows if r["input"] == name])
        consensus_rows.append({
            "input": name,
            "n_samples": len(y),
            "crossfit_repeats": CROSSFIT_REPEATS,
            "crossfit_folds": CROSSFIT_FOLDS,
            "n_candidate_features": Xdf.shape[1],
            "n_selected_features_mean": fr["n_selected_features"].mean(),
            "n_selected_features_min": fr["n_selected_features"].min(),
            "n_selected_features_max": fr["n_selected_features"].max(),
            "consensus_ties_n": ties,
            **primary,
            **cis,
            "ARI_repeat_mean": rr["ARI_vs_final"].mean(),
            "ARI_repeat_sd": rr["ARI_vs_final"].std(),
            "C4_jaccard_repeat_mean": rr["C4_jaccard"].mean(),
            "C4_jaccard_repeat_sd": rr["C4_jaccard"].std(),
            "survival_logrank_p": try_survival_p(samples, consensus, endpoint="RFS"),
            "survival_logrank_p_repeat_median": rr["survival_logrank_p"].median(),
            "survival_logrank_p_repeat_min": rr["survival_logrank_p"].min(),
            "survival_logrank_p_repeat_max": rr["survival_logrank_p"].max(),
            "survival_logrank_p_OS": try_survival_p(samples, consensus, endpoint="OS"),
            "survival_logrank_p_OS_repeat_median": rr["survival_logrank_p_OS"].median(),
            "survival_logrank_p_OS_repeat_min": rr["survival_logrank_p_OS"].min(),
            "survival_logrank_p_OS_repeat_max": rr["survival_logrank_p_OS"].max(),
        })

    summary = pd.DataFrame(consensus_rows)
    summary = add_fdr_column_wide(summary, "survival_logrank_p", "survival_logrank_p_fdr_bh")
    summary = add_fdr_column_wide(summary, "survival_logrank_p_OS", "survival_logrank_p_OS_fdr_bh")
    return summary, pd.DataFrame(repeat_rows), pd.DataFrame(feature_rows), prediction_table


def paired_bootstrap_contrasts(prediction_table, n_boot=BOOTSTRAPS):
    """Patient-paired contrasts between every modality pair.

    The same bootstrap sample is applied to both predictions in a contrast,
    so the interval concerns the difference in recovery on identical patients.
    Two-sided sign-bootstrap P values are BH-adjusted separately for the ARI
    and C4-Jaccard families (21 contrasts each).
    """
    order = ["SNV", "CNV", "SV", "SNV+CNV", "SNV+SV", "CNV+SV", "ALL"]
    y = prediction_table["reference"].to_numpy(dtype=int)
    pred = {m: prediction_table[f"{m}_crossfit_consensus"].to_numpy(dtype=int) for m in order}
    rng = np.random.default_rng(MASTER_SEED + 70707)
    boot_indices = [rng.integers(0, len(y), size=len(y)) for _ in range(n_boot)]
    rows = []

    def c4_jaccard(a, b):
        ta, pa = a == 3, b == 3
        return np.logical_and(ta, pa).sum() / max(np.logical_or(ta, pa).sum(), 1)

    for a, b in combinations(order, 2):
        for metric in ["ARI", "C4_jaccard"]:
            if metric == "ARI":
                observed = adjusted_rand_score(y, pred[a]) - adjusted_rand_score(y, pred[b])
            else:
                observed = c4_jaccard(y, pred[a]) - c4_jaccard(y, pred[b])
            diffs = []
            for idx in boot_indices:
                if metric == "ARI":
                    d = adjusted_rand_score(y[idx], pred[a][idx]) - adjusted_rand_score(y[idx], pred[b][idx])
                else:
                    d = c4_jaccard(y[idx], pred[a][idx]) - c4_jaccard(y[idx], pred[b][idx])
                diffs.append(d)
            diffs = np.asarray(diffs, dtype=float)
            left = (np.sum(diffs <= 0) + 1) / (n_boot + 1)
            right = (np.sum(diffs >= 0) + 1) / (n_boot + 1)
            rows.append({
                "metric": metric, "input_a": a, "input_b": b,
                "difference_a_minus_b": observed,
                "ci_low": float(np.quantile(diffs, 0.025)),
                "ci_high": float(np.quantile(diffs, 0.975)),
                "paired_bootstrap_p": min(1.0, 2 * min(left, right)),
            })
    out = pd.DataFrame(rows)
    out["paired_bootstrap_fdr_bh"] = np.nan
    for metric, idx in out.groupby("metric").groups.items():
        _, q, _, _ = multipletests(out.loc[idx, "paired_bootstrap_p"], method="fdr_bh")
        out.loc[idx, "paired_bootstrap_fdr_bh"] = q
    return out


def plot_crossfit_summary(summary):
    order = ["SNV", "CNV", "SV", "SNV+CNV", "SNV+SV", "CNV+SV", "ALL"]
    s = summary.set_index("input").loc[order]
    x = np.arange(len(order))
    colors = [MODALITY_COLORS[o] for o in order]

    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    vals = s["ARI_vs_final"].to_numpy()
    lo = vals - s["ARI_vs_final_ci_low"].to_numpy()
    hi = s["ARI_vs_final_ci_high"].to_numpy() - vals
    ax.bar(x, vals, color=colors, edgecolor="black", yerr=np.vstack([lo, hi]), capsize=5)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=30, ha="right", fontweight="bold")
    ax.set_ylabel("Cross-fitted ARI vs reference")
    ax.set_title("Out-of-fold partition recovery", fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIGDIR / f"modality_dropout_crossfit_ARI.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14.0, 7.2))
    metrics = [("C4_precision", "Precision"), ("C4_recall", "Recall"), ("C4_jaccard", "Jaccard")]
    width = 0.24
    metric_colors = ["#0072B2", "#D55E00", "#009E73"]
    for j, ((metric, label), color) in enumerate(zip(metrics, metric_colors)):
        vals = s[metric].to_numpy()
        lo = vals - s[f"{metric}_ci_low"].to_numpy()
        hi = s[f"{metric}_ci_high"].to_numpy() - vals
        ax.bar(x + (j-1)*width, vals, width, label=label, color=color,
               edgecolor="black", yerr=np.vstack([lo, hi]), capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=30, ha="right", fontweight="bold")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Cross-fitted C4 recovery")
    ax.set_title("Out-of-fold C4 recovery", fontweight="bold")
    ax.legend(frameon=False, ncol=3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIGDIR / f"C4_recovery_crossfit.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    mats = {
        "SNV": read_matrix(M1, "SNV"),
        "CNV": read_matrix(M2, "CNV"),
        "SV": read_matrix(M3, "SV"),
        "ALL": read_matrix(M4),
    }
    labels = read_labels(CLUSTER_FILE)
    mats, labels = align_matrices(mats, labels)
    print(f"Common samples: {len(labels)}")
    # The legacy whole-cohort restart analysis is retained in the script for
    # reproducibility, but the manuscript-facing primary analysis is now the
    # repeated cross-fitted analysis below.  It avoids whole-cohort oracle
    # matching, provides paired patient-bootstrap uncertainty, repeats the
    # prevalence filter inside training folds, and evaluates survival on every
    # repeat plus the out-of-fold consensus rather than on repeat zero alone.
    summary, repeat_metrics, feature_metrics, predictions = run_crossfitted_dropout(mats, labels)
    summary.to_csv(TABDIR / "modality_dropout_crossfit_summary.csv", index=False)
    repeat_metrics.to_csv(TABDIR / "modality_dropout_crossfit_repeat_metrics.csv", index=False)
    feature_metrics.to_csv(TABDIR / "modality_dropout_crossfit_feature_selection.csv", index=False)
    predictions.to_csv(TABDIR / "modality_dropout_crossfit_predictions.csv", index=False)
    contrasts = paired_bootstrap_contrasts(predictions)
    contrasts.to_csv(TABDIR / "modality_dropout_crossfit_paired_contrasts.csv", index=False)
    plot_crossfit_summary(summary)
    print("\nCross-fitted summary:")
    print(summary[["input", "ARI_vs_final", "C4_precision", "C4_recall", "C4_jaccard",
                   "survival_logrank_p", "survival_logrank_p_fdr_bh",
                   "survival_logrank_p_OS", "survival_logrank_p_OS_fdr_bh"]].to_string(index=False))
    print(f"Done. Outputs: {OUTDIR}")


if __name__ == "__main__":
    main()
