#!/usr/bin/env python3
"""Evaluate held-out supervised recoverability of the locked genomic subtypes.

Eight classifier families are trained on the 371 binary features using a fixed
80:20 subtype-stratified split. Oversampling is restricted to training data;
held-out performance and model feature importance are reported.
"""

import os, re, time, warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix,
    roc_auc_score, roc_curve, auc,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import shap
from imblearn.over_sampling import RandomOverSampler

##############################################################################
# CONFIGURATION
##############################################################################
MASTER_SEED = 42
np.random.seed(MASTER_SEED)
N_JOBS = max(1, os.cpu_count() - 2)

BASE = os.environ.get("CRC_BASE", str(Path(__file__).resolve().parents[1]))
DATA_FILE = f"{BASE}/module4_results/module4_unified_discovery_matrix.csv"
CLUSTER_FILE = os.environ.get(
    "CRC_CLUSTER_FILE",
    f"{BASE}/module05_06_loocv_results/labels/NMF_k4_LOOCV.csv",
)
OUTDIR = os.environ.get("CRC_M12_OUTDIR", f"{BASE}/module12_ml_validation_C1C4_ppt")
FIGDIR = os.path.join(OUTDIR, "figures")
TABLEDIR = os.path.join(OUTDIR, "tables")
SHAPDIR = os.path.join(OUTDIR, "SHAP")
PERDIR = os.path.join(OUTDIR, "per_classifier")
for d in [OUTDIR, FIGDIR, TABLEDIR, SHAPDIR, PERDIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.20
N_CV_FOLDS = int(os.environ.get("CRC_M12_CV_FOLDS", "10"))
CNN_EPOCHS = int(os.environ.get("CRC_M12_CNN_EPOCHS", "200"))
CNN_LR_INIT = 1e-3
CNN_BATCH = 64
TOP_N_SHAP = int(os.environ.get("CRC_M12_TOP_SHAP", "15"))
TOP_N_IMPORTANCE = int(os.environ.get("CRC_M12_TOP_IMPORTANCE", "15"))

# Okabe-Ito colour-blind-safe palette
CB = {
    "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
    "purple": "#CC79A7", "grey": "#999999", "black": "#000000",
}
CLF_COLORS = {
    "RFC": CB["green"], "SVM": CB["vermillion"], "NaiveBayes": CB["orange"],
    "KNN": CB["sky"], "XGBoost": CB["purple"], "LR": CB["blue"],
    "MLP": CB["yellow"], "ResNet50_1D": CB["grey"],
}
CLASS_COLORS = [CB["orange"], CB["sky"], CB["green"], CB["purple"]]
TRAIN_COLOR = CB["blue"]
TEST_COLOR = CB["vermillion"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 18,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 17,
    "legend.title_fontsize": 18,
    "savefig.dpi": 300,
    "axes.linewidth": 1.2,
    "figure.facecolor": "white",
})

##############################################################################
# HELPERS
##############################################################################
def normalize_sample(x):
    m = re.search(r"(UM\d+|U\d+)", str(x))
    return m.group(1) if m else np.nan

def display_cluster(c):
    return f"C{int(c) + 1}"

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

def rose_resample(X, y, seed=MASTER_SEED):
    return RandomOverSampler(random_state=seed).fit_resample(X, y)

def is_sv_feature(name):
    return str(name).startswith("SV_")

def feature_type(name):
    s = str(name)
    if s.startswith("SNV_"):
        return "SNV"
    if s.startswith("CNV_"):
        return "CNV"
    if s.startswith("SV_"):
        return "SV"
    return "Other"

def clean_feature_label(name, max_len=34):
    s = re.sub(r"^(SNV_|CNV_|SV_)", "", str(name))
    s = s.replace("_", " ")
    return s if len(s) <= max_len else s[:max_len - 1] + "…"

def safe_prob(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    if hasattr(clf, "decision_function"):
        z = clf.decision_function(X)
        if z.ndim == 1:
            z = np.column_stack([-z, z])
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)
    return None

def compute_metrics(y_true, y_pred, y_prob=None, classes=None):
    if classes is None:
        classes = np.unique(y_true)
    cm_ovr = multilabel_confusion_matrix(y_true, y_pred, labels=classes)
    sens, spec = [], []
    for m in cm_ovr:
        tn, fp, fn, tp = m.ravel()
        sens.append(tp / (tp + fn) if (tp + fn) else 0.0)
        spec.append(tn / (tn + fp) if (tn + fp) else 0.0)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": float(np.mean(sens)),
        "specificity": float(np.mean(spec)),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_prob is not None:
        try:
            out["auroc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except Exception:
            out["auroc"] = np.nan
    else:
        out["auroc"] = np.nan
    return out

##############################################################################
# DATA
##############################################################################
def load_data():
    print("Loading matrix...")
    X_df = pd.read_csv(DATA_FILE, index_col=0)
    X_df.index = X_df.index.astype(str).map(normalize_sample)
    X_df = X_df.loc[~X_df.index.isna() & ~X_df.index.duplicated()]

    print("Loading labels...")
    lab = pd.read_csv(CLUSTER_FILE)
    lab["sample_id"] = lab["sample_id"].astype(str).apply(normalize_sample)
    lab = lab.dropna(subset=["sample_id"]).drop_duplicates("sample_id").set_index("sample_id")
    lab["cluster"] = normalize_cluster_values(lab["cluster"])

    common = X_df.index.intersection(lab.index)
    X_df = X_df.loc[common]
    y = lab.loc[common, "cluster"].values
    X = X_df.values.astype(np.float32)
    fnames = X_df.columns.tolist()
    counts = dict(zip(*np.unique(y, return_counts=True)))
    _expected_counts = {0: 426, 1: 274, 2: 268, 3: 94}
    _observed_counts = {int(k): int(v) for k, v in counts.items()}
    if _observed_counts != _expected_counts:
        raise ValueError(
            f"Cluster counts after 0/1-index normalization do not match the "
            f"locked solution (observed {_observed_counts}, expected "
            f"{_expected_counts}). Check CLUSTER_FILE={CLUSTER_FILE}."
        )
    print(f"  X={X.shape}; labels={counts}; displayed as C1-C4")
    return X, y, fnames

##############################################################################
# CLASSIFIERS
##############################################################################
def get_classifiers():
    return {
        "RFC": lambda s: RandomForestClassifier(
            n_estimators=500, max_features="sqrt", class_weight="balanced",
            n_jobs=N_JOBS, random_state=s),
        "SVM": lambda s: SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True,
            class_weight="balanced", random_state=s),
        "NaiveBayes": lambda s: BernoulliNB(alpha=1.0),
        "KNN": lambda s: KNeighborsClassifier(n_neighbors=7, metric="hamming", n_jobs=N_JOBS),
        "XGBoost": lambda s: xgb.XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
            nthread=N_JOBS, random_state=s, verbosity=0),
        "LR": lambda s: LogisticRegression(
            solver="lbfgs", C=1.0, max_iter=2000, class_weight="balanced",
            n_jobs=N_JOBS, random_state=s),
        "MLP": lambda s: MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), activation="relu", max_iter=500,
            early_stopping=True, validation_fraction=0.1, random_state=s),
    }

##############################################################################
# EVALUATION
##############################################################################
def cross_validate_classifier(name, factory, X_train, y_train, classes):
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=MASTER_SEED)
    rows, y_all, prob_all = [], [], []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        Xtr, Xva = X_train[tr_idx], X_train[va_idx]
        ytr, yva = y_train[tr_idx], y_train[va_idx]
        Xtr_r, ytr_r = rose_resample(Xtr, ytr, seed=MASTER_SEED + fold)
        clf = factory(MASTER_SEED + fold)
        clf.fit(Xtr_r, ytr_r)
        ypred = clf.predict(Xva)
        yprob = safe_prob(clf, Xva)
        rows.append({"fold": fold, **compute_metrics(yva, ypred, yprob, classes=classes)})
        y_all.append(yva)
        if yprob is not None:
            prob_all.append(yprob)
    cv_df = pd.DataFrame(rows)
    y_cv = np.concatenate(y_all)
    prob_cv = np.vstack(prob_all) if prob_all else None
    return cv_df, y_cv, prob_cv

def fit_test_classifier(name, factory, X_train, y_train, X_test, y_test, classes):
    Xr, yr = rose_resample(X_train, y_train)
    clf = factory(MASTER_SEED)
    clf.fit(Xr, yr)
    ypred = clf.predict(X_test)
    yprob = safe_prob(clf, X_test)
    metrics = compute_metrics(y_test, ypred, yprob, classes=classes)
    return clf, metrics, ypred, yprob

##############################################################################
# PLOTS
##############################################################################
def plot_confusion(y_true, y_pred, name, outpath):
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(7.2, 6.5))
    disp = ConfusionMatrixDisplay(cm, display_labels=[display_cluster(c) for c in classes])
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title("Confusion", fontweight="bold", pad=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    savefig(outpath)

def plot_per_class(report_df, name, outpath):
    classes = [i for i in report_df.index if str(i).startswith("C")]
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(classes)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    for i, m in enumerate(metrics):
        vals = [report_df.loc[c, m] for c in classes]
        ax.bar(x + (i-1)*w, vals, w, label=m.replace("f1-score", "F1").title(),
               color=[CB["blue"], CB["orange"], CB["green"]][i],
               edgecolor="black", linewidth=0.8)
    for yref in [0.25, 0.5, 0.75, 1.0]:
        ax.axhline(yref, color="lightgrey", ls="--", lw=0.8, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(classes, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Per-class", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.22))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    savefig(outpath)

def _clf_order(summary_df):
    return [c for c in CLF_COLORS if c in summary_df["classifier"].tolist()]

def _pretty_clf(name):
    return {"ResNet50_1D": "ResNet", "NaiveBayes": "Naive Bayes"}.get(str(name), str(name))

def plot_performance(summary_df, outpath):
    """Reference-style compact performance plots.

    Two separate files are saved from the same call:
      <outpath>_CV     : 10-fold CV/training performance
      <outpath>_Test   : held-out test performance

    Each file has four horizontal metric panels, all eight classifiers shown
    simultaneously, with large fonts and colour-blind-safe bars.
    """
    metrics = [("accuracy", "Accuracy"), ("sensitivity", "Sensitivity"),
               ("specificity", "Specificity"), ("macro_f1", "F1")]
    clf_order = _clf_order(summary_df)
    x = np.arange(len(clf_order))

    def _one(prefix, title_suffix, outfile, show_errors=False):
        fig, axes = plt.subplots(len(metrics), 1, figsize=(15.5, 10.5), sharex=True)
        colors = [CLF_COLORS[c] for c in clf_order]
        for ax, (key, label) in zip(axes, metrics):
            col = f"{prefix}_{key}"
            vals = [summary_df.loc[summary_df.classifier == c, col].values[0]
                    for c in clf_order]
            errs = None
            if show_errors:
                errs = [summary_df.loc[summary_df.classifier == c, f"cv_sd_{key}"].values[0]
                        for c in clf_order]
            bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=1.15,
                          yerr=errs, error_kw=dict(ecolor="black", lw=1.2, capsize=4),
                          zorder=3)
            for yref in [0.25, 0.50, 0.75, 1.00]:
                ax.axhline(yref, color="lightgrey", ls="--", lw=1.0, zorder=0)
            ax.axhline(0.25, color=CB["purple"], ls=":", lw=1.7, zorder=1)
            ax.set_ylim(0, 1.08)
            ax.set_ylabel("Score", fontweight="bold")
            ax.set_title(label, fontsize=18, fontweight="bold", pad=4)
            ax.tick_params(axis="both", labelsize=16)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            # Add compact values only for the top panel to avoid clutter.
            if key == "accuracy":
                for b, v in zip(bars, vals):
                    if np.isfinite(v):
                        ax.text(b.get_x()+b.get_width()/2, min(v+0.025, 1.045),
                                f"{v:.2f}", ha="center", va="bottom",
                                fontsize=11.5, rotation=90, fontweight="bold")
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels([_pretty_clf(c) for c in clf_order],
                                 rotation=35, ha="right", fontsize=16, fontweight="bold")
        handles = [Patch(facecolor=CLF_COLORS[c], edgecolor="black", label=_pretty_clf(c))
                   for c in clf_order]
        fig.legend(handles=handles, title="Classifier", ncol=4, frameon=False,
                   loc="upper center", bbox_to_anchor=(0.5, 0.995),
                   fontsize=14, title_fontsize=15)
        fig.suptitle(title_suffix, fontsize=26, fontweight="bold", y=1.045)
        savefig(outfile, tight=False)

    _one("cv_mean", "CV", f"{outpath}_CV", show_errors=True)
    _one("test", "Test", f"{outpath}_Test", show_errors=False)

def _macro_roc_curve(y_true, y_prob):
    if y_prob is None:
        return None, None, np.nan
    n_cls = y_prob.shape[1]
    Y = label_binarize(y_true, classes=list(range(n_cls)))
    if Y.shape[1] == 1:
        Y = np.hstack([1 - Y, Y])
    mean_fpr = np.linspace(0, 1, 300)
    tprs = []
    for i in range(min(n_cls, Y.shape[1])):
        try:
            fpr, tpr, _ = roc_curve(Y[:, i], y_prob[:, i])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
        except Exception:
            continue
    if not tprs:
        return None, None, np.nan
    mean_tpr = np.mean(tprs, axis=0)
    return mean_fpr, mean_tpr, auc(mean_fpr, mean_tpr)

def plot_auroc_curves(roc_data, outpath, title="AUROC"):
    """Reference-style multi-classifier macro-OvR AUROC curve."""
    fig, ax = plt.subplots(figsize=(9.8, 8.4))
    any_curve = False
    for clf_name in [c for c in CLF_COLORS if c in roc_data]:
        y_true, y_prob = roc_data[clf_name]
        fpr, tpr, au = _macro_roc_curve(y_true, y_prob)
        if fpr is None:
            continue
        any_curve = True
        ax.plot(fpr, tpr, lw=3.0, color=CLF_COLORS[clf_name],
                label=f"{_pretty_clf(clf_name)} (AUC={au:.3f})")
        # value tag near the right side; clipped safely inside axes
        x_tag = 0.72
        y_tag = float(np.interp(x_tag, fpr, tpr))
        ax.text(x_tag+0.012, min(y_tag, 0.98), f"{au:.2f}",
                color=CLF_COLORS[clf_name], fontsize=12.5, fontweight="bold",
                va="center")
    ax.plot([0, 1], [0, 1], "--", color="dimgrey", lw=1.5, label="Chance")
    ax.set_xlabel("1-Specificity", fontweight="bold")
    ax.set_ylabel("Sensitivity", fontweight="bold")
    ax.set_title(title, fontsize=26, fontweight="bold", pad=10)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(True, ls="--", color="lightgrey", lw=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=17)
    ax.legend(frameon=True, edgecolor="lightgrey", fontsize=12.5,
              loc="lower right", ncol=1)
    if any_curve:
        savefig(outpath, tight=True)
    else:
        plt.close(fig)

def plot_auroc_bars(summary_df, outpath):
    """Retained wrapper for backward compatibility; creates compact AUROC bars."""
    clf_order = _clf_order(summary_df)
    x = np.arange(len(clf_order)); w = 0.36
    fig, ax = plt.subplots(figsize=(14.5, 7.5))
    cv_vals = [summary_df.loc[summary_df.classifier == c, "cv_mean_auroc"].values[0] for c in clf_order]
    cv_errs = [summary_df.loc[summary_df.classifier == c, "cv_sd_auroc"].values[0] for c in clf_order]
    te_vals = [summary_df.loc[summary_df.classifier == c, "test_auroc"].values[0] for c in clf_order]
    ax.bar(x-w/2, cv_vals, w, color=TRAIN_COLOR, edgecolor="black", linewidth=1.1,
           yerr=cv_errs, error_kw=dict(ecolor="black", lw=1.1, capsize=4), label="CV")
    ax.bar(x+w/2, te_vals, w, color=TEST_COLOR, edgecolor="black", linewidth=1.1,
           hatch="//", label="Test")
    for xi, v in zip(x-w/2, cv_vals):
        if np.isfinite(v): ax.text(xi, min(v+0.018, 1.02), f"{v:.2f}", ha="center", va="bottom", fontsize=11, rotation=90)
    for xi, v in zip(x+w/2, te_vals):
        if np.isfinite(v): ax.text(xi, min(v+0.018, 1.02), f"{v:.2f}", ha="center", va="bottom", fontsize=11, rotation=90)
    for yref in [0.5, 0.75, 1.0]:
        ax.axhline(yref, color="lightgrey", ls="--", lw=0.9, zorder=0)
    ax.axhline(0.5, color=CB["purple"], ls=":", lw=1.7)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("AUROC", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([_pretty_clf(c) for c in clf_order], rotation=30, ha="right", fontweight="bold")
    ax.set_title("AUROC", fontsize=26, fontweight="bold")
    ax.legend(frameon=False, ncol=2)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    savefig(outpath, tight=True)

def plot_macro_roc(y_true, y_prob, name, outpath):
    if y_prob is None:
        return
    n_cls = y_prob.shape[1]
    Y = label_binarize(y_true, classes=list(range(n_cls)))
    mean_fpr = np.linspace(0, 1, 250)
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    for i in range(min(n_cls, Y.shape[1])):
        fpr, tpr, _ = roc_curve(Y[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, lw=2.2, color=CLASS_COLORS[i % len(CLASS_COLORS)],
                label=f"{display_cluster(i)} AUC={auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1.3)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("ROC", fontweight="bold")
    ax.legend(frameon=False, loc="lower right", fontsize=14)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, ls="--", color="lightgrey", lw=0.6, alpha=0.7)
    savefig(outpath)

def plot_importance(importances, feature_names, name, outpath, top_n=TOP_N_IMPORTANCE):
    if importances is None:
        return
    idx = np.argsort(importances)[::-1][:top_n]
    idx = idx[::-1]
    labels = [clean_feature_label(feature_names[i]) for i in idx]
    colors = [{"SNV": CB["orange"], "CNV": CB["blue"], "SV": CB["green"]}.get(feature_type(feature_names[i]), CB["grey"]) for i in idx]
    fig, ax = plt.subplots(figsize=(9.5, max(6.0, 0.42 * len(idx) + 2.0)))
    ax.barh(np.arange(len(idx)), importances[idx], color=colors, edgecolor="black", linewidth=0.7)
    ax.set_yticks(np.arange(len(idx)))
    ax.set_yticklabels(labels, fontsize=16)
    ax.set_xlabel("Importance")
    ax.set_title("Importance", fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(handles=[Patch(color=CB["orange"], label="SNV"), Patch(color=CB["blue"], label="CNV"),
                       Patch(color=CB["green"], label="SV")],
              frameon=False, ncol=3, loc="lower right")
    savefig(outpath)

def plot_shap_bar(shap_mean, feature_names, name, outpath, top_n=TOP_N_SHAP):
    idx = np.argsort(shap_mean)[::-1][:top_n]
    idx = idx[::-1]
    labels = [clean_feature_label(feature_names[i]) for i in idx]
    colors = [{"SNV": CB["orange"], "CNV": CB["blue"], "SV": CB["green"]}.get(feature_type(feature_names[i]), CB["grey"]) for i in idx]
    fig, ax = plt.subplots(figsize=(9.8, max(6.0, 0.48 * len(idx) + 2.2)))
    ax.barh(np.arange(len(idx)), shap_mean[idx], color=colors, edgecolor="black", linewidth=0.8)
    ax.set_yticks(np.arange(len(idx)))
    ax.set_yticklabels(labels, fontsize=17)
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title("SHAP", fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", ls="--", color="lightgrey", lw=0.6, alpha=0.7)
    ax.legend(handles=[Patch(color=CB["orange"], label="SNV"), Patch(color=CB["blue"], label="CNV"),
                       Patch(color=CB["green"], label="SV")],
              frameon=False, ncol=3, loc="lower right")
    savefig(outpath)
    return pd.DataFrame({
        "rank": range(1, len(idx) + 1),
        "feature": [feature_names[i] for i in idx[::-1]],
        "feature_type": [feature_type(feature_names[i]) for i in idx[::-1]],
        "mean_abs_shap": shap_mean[idx[::-1]],
    })


def plot_importance_combined(importance_dict, feature_names, outpath, top_n=20):
    """Reference-style grouped feature-importance plot across classifiers.

    v3 layout-only update:
      - selected features are still the same top-N union-max features
      - features are grouped by modality (SNV / CNV / SV / Other)
      - light background bands + top labels demarcate each feature class
      - model legend is moved outside the plotting area so it no longer
        clashes with the short title.
    """
    if not importance_dict:
        return
    model_order = [m for m in ["LR", "RFC", "XGBoost", "NaiveBayes", "MLP"] if m in importance_dict]
    if not model_order:
        return

    mat = np.vstack([importance_dict[m] for m in model_order])

    # Keep the same selection rule as before: top-N by union-max importance.
    top_idx_raw = np.argsort(np.nanmax(mat, axis=0))[::-1][:top_n]

    # New display-only ordering: group selected features by genomic modality.
    # Within each modality, retain decreasing union-max importance.
    type_order = {"SNV": 0, "CNV": 1, "SV": 2, "Other": 3}
    union_max = np.nanmax(mat, axis=0)
    top_idx = sorted(
        top_idx_raw,
        key=lambda i: (type_order.get(feature_type(feature_names[i]), 99), -union_max[i])
    )

    labels = [clean_feature_label(feature_names[i]) for i in top_idx]
    feat_types = [feature_type(feature_names[i]) for i in top_idx]
    x = np.arange(len(top_idx))
    width = min(0.80 / max(len(model_order), 1), 0.18)

    type_colors = {
        "SNV": CB["orange"],
        "CNV": CB["sky"],
        "SV":  CB["green"],
        "Other": CB["grey"],
    }

    fig_w = max(21.5, 0.82 * len(top_idx) + 10.5)
    fig, ax = plt.subplots(figsize=(fig_w, 8.9))

    # Leave explicit, in-canvas room for the right-side legends.
    # The legends are placed with fig.legend() in figure coordinates so they
    # are never clipped by bbox_inches='tight'.
    fig.subplots_adjust(top=0.86, right=0.76, bottom=0.30, left=0.055)

    # Feature-type demarcation blocks behind bars.
    blocks = []
    start_i = 0
    for i in range(1, len(feat_types) + 1):
        if i == len(feat_types) or feat_types[i] != feat_types[start_i]:
            blocks.append((feat_types[start_i], start_i, i - 1))
            start_i = i

    for typ, a, b in blocks:
        col = type_colors.get(typ, CB["grey"])
        ax.axvspan(a - 0.5, b + 0.5, color=col, alpha=0.12, zorder=0)
        ax.axvline(a - 0.5, color="black", lw=1.0, alpha=0.35, zorder=1)
        ax.axvline(b + 0.5, color="black", lw=1.0, alpha=0.35, zorder=1)
        ax.text((a + b) / 2, 1.025, typ,
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=17, fontweight="bold", color=col, clip_on=False)

    # Classifier bars.
    for j, m in enumerate(model_order):
        vals = importance_dict[m][top_idx]
        offset = (j - len(model_order)/2 + 0.5) * width
        ax.bar(x + offset, vals, width*0.92, color=CLF_COLORS.get(m, CB["grey"]),
               edgecolor="black", linewidth=0.75, label=_pretty_clf(m), zorder=3)

    # Horizontal reference gridlines.
    ymax = max(1.05, float(np.nanmax([np.nanmax(importance_dict[m][top_idx]) for m in model_order])) * 1.18)
    for yref in np.arange(0.1, min(ymax, 1.05), 0.1):
        ax.axhline(yref, color="lightgrey", ls="--", lw=0.8, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=42, ha="right", fontsize=15, fontweight="bold")
    ax.set_ylabel("Importance", fontweight="bold")
    ax.set_title("Importance", fontsize=28, fontweight="bold", pad=22)
    ax.tick_params(axis="y", labelsize=17)
    ax.set_ylim(0, ymax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Model legend in the reserved right margin.  Using fig.legend with
    # figure coordinates prevents the right edge from being cut off.
    model_handles = [Patch(facecolor=CLF_COLORS.get(m, CB["grey"]), edgecolor="black",
                           label=_pretty_clf(m)) for m in model_order]
    fig.legend(handles=model_handles, title="Model", frameon=False,
               loc="upper left", bbox_to_anchor=(0.785, 0.80),
               bbox_transform=fig.transFigure, fontsize=15,
               title_fontsize=16, borderaxespad=0.0, handlelength=1.7)

    # Separate feature-type legend for SNV/CNV/SV demarcation, also inside
    # the reserved right margin.
    present_types = [t for t in ["SNV", "CNV", "SV", "Other"] if t in feat_types]
    type_handles = [Patch(facecolor=type_colors[t], edgecolor="black", alpha=0.55, label=t)
                    for t in present_types]
    fig.legend(handles=type_handles, title="Feature", frameon=False,
               loc="upper left", bbox_to_anchor=(0.785, 0.49),
               bbox_transform=fig.transFigure, fontsize=15,
               title_fontsize=16, borderaxespad=0.0, handlelength=1.7)

    # Save with a small pad so all figure-level legend text remains visible.
    for ext in ["png", "pdf"]:
        plt.savefig(f"{outpath}.{ext}", bbox_inches="tight", dpi=300, pad_inches=0.25)
    plt.close()
    print(f"  → {outpath}.png/pdf")

    pd.DataFrame({
        "rank": np.arange(1, len(top_idx)+1),
        "feature": [feature_names[i] for i in top_idx],
        "display_label": labels,
        "feature_type": feat_types,
        **{m: importance_dict[m][top_idx] for m in model_order},
    }).to_csv(f"{outpath}_top_features.csv", index=False)

##############################################################################
# FEATURE IMPORTANCE / SHAP
##############################################################################
def extract_native_importance(clf, name):
    try:
        if name in ["RFC", "XGBoost"]:
            imp = clf.feature_importances_
        elif name == "LR":
            imp = np.abs(clf.coef_).mean(axis=0)
        elif name == "NaiveBayes":
            imp = clf.feature_log_prob_.var(axis=0)
        elif name == "MLP":
            imp = np.abs(clf.coefs_[0]).sum(axis=1)
        else:
            return None
        lo, hi = np.nanmin(imp), np.nanmax(imp)
        return (imp - lo) / (hi - lo + 1e-12)
    except Exception:
        return None

def shap_rfc(clf, X_bg):
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_bg)
    arr = np.asarray(sv)
    # multi-class may be list -> (classes, samples, features) or array (samples, features, classes)
    if arr.ndim == 3 and arr.shape[0] != len(np.unique(clf.classes_)):
        arr = arr.transpose(2, 0, 1)
    return np.abs(arr).mean(axis=(0, 1))

def shap_xgb_native(clf, X_bg):
    dmat = xgb.DMatrix(X_bg)
    sv = clf.get_booster().predict(dmat, pred_contribs=True)
    if sv.ndim == 3:
        arr = sv[:, :, :-1].transpose(1, 0, 2)
    else:
        arr = sv[:, :-1][np.newaxis, :, :]
    return np.abs(arr).mean(axis=(0, 1))

##############################################################################
# RESNET
##############################################################################
def build_resnet50_1d(n_feat, n_classes):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    def res_block(x, filters, stride=1):
        shortcut = x
        if stride != 1 or x.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, 1, strides=stride, padding="same",
                                     kernel_initializer="lecun_normal")(shortcut)
        for j in range(3):
            x = layers.Conv1D(filters, 3, strides=(stride if j == 0 else 1), padding="same",
                              kernel_initializer="lecun_normal")(x)
            x = layers.Activation("selu")(x)
        x = layers.Add()([x, shortcut])
        return layers.Activation("selu")(x)

    inp = keras.Input(shape=(n_feat, 1))
    x = layers.Conv1D(64, 7, padding="same", kernel_initializer="lecun_normal")(inp)
    x = layers.Activation("selu")(x)
    x = layers.MaxPooling1D(3, strides=2, padding="same")(x)
    for filters, n_blocks in [(64, 3), (128, 4), (256, 3), (512, 3)]:
        x = res_block(x, filters, stride=2)
        for _ in range(n_blocks - 1):
            x = res_block(x, filters)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="selu", kernel_initializer="lecun_normal")(x)
    x = layers.Dropout(0.3, seed=MASTER_SEED)(x)
    out = layers.Dense(n_classes, activation="softmax", kernel_initializer="lecun_normal")(x)
    return keras.Model(inp, out, name="ResNet50_1D")

def lr_schedule(epoch):
    if epoch < 5:
        return CNN_LR_INIT * (epoch + 1) / 5
    prog = (epoch - 5) / max(1, CNN_EPOCHS - 5)
    return CNN_LR_INIT / 100 + 0.5 * (CNN_LR_INIT - CNN_LR_INIT / 100) * (1 + np.cos(np.pi * prog))

def run_resnet(X_train, y_train, X_test, y_test, classes):
    print("\n" + "=" * 70 + "\n  ResNet50_1D\n" + "=" * 70)
    import tensorflow as tf
    from tensorflow import keras
    tf.random.set_seed(MASTER_SEED)

    n_feat = X_train.shape[1]
    n_cls = len(classes)
    cdir = os.path.join(PERDIR, "ResNet50_1D")
    Path(cdir).mkdir(parents=True, exist_ok=True)

    Xtr_r, ytr_r = rose_resample(X_train, y_train)
    Xtr, Xval, ytr, yval = train_test_split(
        Xtr_r, ytr_r, test_size=0.125, stratify=ytr_r, random_state=MASTER_SEED)

    def to3d(a):
        return a.reshape(-1, n_feat, 1)

    model = build_resnet50_1d(n_feat, n_cls)
    model.compile(optimizer=keras.optimizers.Nadam(learning_rate=CNN_LR_INIT),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    ckpt = os.path.join(cdir, "best_model.keras")

    class BestF1(keras.callbacks.Callback):
        def __init__(self):
            super().__init__(); self.best = 0.0
        def on_epoch_end(self, epoch, logs=None):
            yp = np.argmax(self.model.predict(to3d(Xval), verbose=0), axis=1)
            f1 = f1_score(yval, yp, average="macro", zero_division=0)
            if f1 > self.best:
                self.best = f1; self.model.save(ckpt)

    hist = model.fit(
        to3d(Xtr), keras.utils.to_categorical(ytr, n_cls),
        validation_data=(to3d(Xval), keras.utils.to_categorical(yval, n_cls)),
        epochs=CNN_EPOCHS, batch_size=CNN_BATCH, verbose=0,
        callbacks=[keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0),
                   BestF1(), keras.callbacks.EarlyStopping(monitor="val_loss", patience=30,
                                                           restore_best_weights=True, verbose=0)]
    )
    if os.path.exists(ckpt):
        model = keras.models.load_model(ckpt)

    prob_val = model.predict(to3d(Xval), verbose=0)
    pred_val = np.argmax(prob_val, axis=1)
    cv_metrics = compute_metrics(yval, pred_val, prob_val, classes=classes)

    prob_test = model.predict(to3d(X_test), verbose=0)
    pred_test = np.argmax(prob_test, axis=1)
    test_metrics = compute_metrics(y_test, pred_test, prob_test, classes=classes)

    # History
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(hist.history["loss"], color=TRAIN_COLOR, lw=2.5, label="Train")
    axes[0].plot(hist.history["val_loss"], color=TEST_COLOR, lw=2.5, label="Val")
    axes[0].set_title("Loss", fontweight="bold"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[1].plot(hist.history["accuracy"], color=TRAIN_COLOR, lw=2.5, label="Train")
    axes[1].plot(hist.history["val_accuracy"], color=TEST_COLOR, lw=2.5, label="Val")
    axes[1].set_title("Accuracy", fontweight="bold"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    for ax in axes:
        ax.legend(frameon=False); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    savefig(os.path.join(cdir, "training_history"))

    plot_confusion(y_test, pred_test, "ResNet50_1D", os.path.join(cdir, "confusion"))
    rpt = classification_report(y_test, pred_test, target_names=[display_cluster(c) for c in classes], output_dict=True)
    rpt_df = pd.DataFrame(rpt).T
    rpt_df.to_csv(os.path.join(cdir, "classification_report.csv"))
    plot_per_class(rpt_df, "ResNet50_1D", os.path.join(cdir, "per_class"))
    plot_macro_roc(y_test, prob_test, "ResNet50_1D", os.path.join(cdir, "roc"))
    return cv_metrics, test_metrics, pred_test, prob_test, yval, prob_val

##############################################################################
# MAIN
##############################################################################
def main():
    print("=" * 70)
    print("MODULE 12 — ML VALIDATION OF NMF k=4 SUBTYPES")
    print("=" * 70)
    print(f"N_JOBS={N_JOBS}; CV folds={N_CV_FOLDS}; output={OUTDIR}")
    t0 = time.time()

    X, y, fnames = load_data()
    classes = sorted(np.unique(y))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=MASTER_SEED)
    print(f"Train={X_train.shape}; Test={X_test.shape}")

    rows = []
    roc_test = {}
    roc_cv = {}
    trained = {}
    classifiers = get_classifiers()

    for name, factory in classifiers.items():
        print("\n" + "=" * 70 + f"\n  {name}\n" + "=" * 70)
        cdir = os.path.join(PERDIR, name)
        Path(cdir).mkdir(parents=True, exist_ok=True)
        try:
            cv_df, y_cv, prob_cv = cross_validate_classifier(name, factory, X_train, y_train, classes)
            cv_df.to_csv(os.path.join(cdir, "cv_metrics.csv"), index=False)

            clf, test_m, y_pred, y_prob = fit_test_classifier(name, factory, X_train, y_train, X_test, y_test, classes)
            trained[name] = clf
            roc_test[name] = (y_test, y_prob)
            if prob_cv is not None:
                roc_cv[name] = (y_cv, prob_cv)

            plot_confusion(y_test, y_pred, name, os.path.join(cdir, "confusion"))
            rpt = classification_report(y_test, y_pred, target_names=[display_cluster(c) for c in classes], output_dict=True)
            rpt_df = pd.DataFrame(rpt).T
            rpt_df.to_csv(os.path.join(cdir, "classification_report.csv"))
            plot_per_class(rpt_df, name, os.path.join(cdir, "per_class"))
            plot_macro_roc(y_test, y_prob, name, os.path.join(cdir, "roc"))

            row = {"classifier": name}
            for m in ["accuracy", "sensitivity", "specificity", "macro_f1", "auroc"]:
                row[f"cv_mean_{m}"] = cv_df[m].mean() if m in cv_df else np.nan
                row[f"cv_sd_{m}"] = cv_df[m].std() if m in cv_df else np.nan
                row[f"test_{m}"] = test_m.get(m, np.nan)
            rows.append(row)
            print("  CV:   " + "  ".join(f"{m}={row[f'cv_mean_{m}']:.3f}" for m in ["accuracy", "sensitivity", "specificity", "macro_f1", "auroc"]))
            print("  Test: " + "  ".join(f"{m}={row[f'test_{m}']:.3f}" for m in ["accuracy", "sensitivity", "specificity", "macro_f1", "auroc"]))
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    # ResNet as eighth classifier
    try:
        cv_m, test_m, pred, prob, y_val_resnet, prob_val_resnet = run_resnet(X_train, y_train, X_test, y_test, classes)
        row = {"classifier": "ResNet50_1D"}
        for m in ["accuracy", "sensitivity", "specificity", "macro_f1", "auroc"]:
            row[f"cv_mean_{m}"] = cv_m.get(m, np.nan)
            row[f"cv_sd_{m}"] = 0.0
            row[f"test_{m}"] = test_m.get(m, np.nan)
        rows.append(row)
        roc_test["ResNet50_1D"] = (y_test, prob)
        roc_cv["ResNet50_1D"] = (y_val_resnet, prob_val_resnet)
    except Exception as e:
        print(f"  ResNet50_1D FAILED: {e}")
        import traceback; traceback.print_exc()

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(TABLEDIR, "summary_metrics.csv"), index=False)

    if not summary.empty:
        plot_performance(summary, os.path.join(FIGDIR, "performance"))
        plot_auroc_bars(summary, os.path.join(FIGDIR, "auroc_bar"))
    if roc_cv:
        plot_auroc_curves(roc_cv, os.path.join(FIGDIR, "auroc_CV"), title="CV AUROC")
    if roc_test:
        plot_auroc_curves(roc_test, os.path.join(FIGDIR, "auroc_Test"), title="Test AUROC")

    # Feature importance and SHAP
    print("\n" + "=" * 70 + "\nFEATURE IMPORTANCE / SHAP\n" + "=" * 70)
    Xr, yr = rose_resample(X_train, y_train)
    combined_importances = {}
    for name in ["RFC", "XGBoost", "LR", "NaiveBayes", "MLP"]:
        if name not in get_classifiers():
            continue
        try:
            clf = get_classifiers()[name](MASTER_SEED)
            clf.fit(Xr, yr)
            imp = extract_native_importance(clf, name)
            if imp is not None:
                combined_importances[name] = imp
                plot_importance(imp, fnames, name, os.path.join(FIGDIR, f"importance_{name}"))
                pd.DataFrame({"feature": fnames, "importance": imp}).sort_values("importance", ascending=False).to_csv(
                    os.path.join(TABLEDIR, f"{name}_importance.csv"), index=False)
            if name == "RFC":
                sm = shap_rfc(clf, X_test)
                st = plot_shap_bar(sm, fnames, name, os.path.join(SHAPDIR, "RFC_shap"))
                if st is not None:
                    st.to_csv(os.path.join(TABLEDIR, "RFC_shap_top.csv"), index=False)
            if name == "XGBoost":
                sm = shap_xgb_native(clf, X_test)
                st = plot_shap_bar(sm, fnames, name, os.path.join(SHAPDIR, "XGBoost_shap"))
                if st is not None:
                    st.to_csv(os.path.join(TABLEDIR, "XGBoost_shap_top.csv"), index=False)
        except Exception as e:
            print(f"  {name} importance/SHAP skipped: {e}")

    if combined_importances:
        plot_importance_combined(combined_importances, fnames, os.path.join(FIGDIR, "importance_combined"), top_n=20)

    print("\nSUMMARY")
    if not summary.empty:
        print(summary[["classifier", "cv_mean_accuracy", "test_accuracy", "cv_mean_macro_f1", "test_macro_f1", "cv_mean_auroc", "test_auroc"]].to_string(index=False, float_format="%.4f"))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")
    print(f"Outputs: {OUTDIR}")

if __name__ == "__main__":
    main()
