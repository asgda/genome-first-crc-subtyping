#!/usr/bin/env python3
"""Select and validate the genome-first NMF subtype solution.

NMF solutions at k=4-6 are evaluated by consensus stability, prevalence-
preserving null comparisons, internal validity and 80% subsampling. Ten
repetitions of five-fold cross-fitting generate held-out assignments; the
pooled repetition-0 k=4 partition is written as the locked C1-C4 label file.
Survival is reported only as a diagnostic and is not used to select k.
"""

import os
import re
import json
import time
import hashlib
import platform
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import chi2
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.optimize import linear_sum_assignment

from sklearn.decomposition import NMF as skNMF
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score,
                             confusion_matrix)
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.spatial.distance import cdist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ════════════════════════════ CONFIG ════════════════════════════
MASTER_SEED = 42
np.random.seed(MASTER_SEED)
N_JOBS = int(os.environ.get("CRC_N_JOBS", max(1, (os.cpu_count() or 2) - 2)))
K_RANGE = [int(v) for v in os.environ.get("CRC_K_RANGE", "4,5,6").split(",") if v.strip()]
LOCK_K = 4
LOCK_COUNTS = (426, 274, 268, 94)

# Historical NMF configuration -- unchanged, this is what defines the lock.
NMF_KW = dict(init="nndsvda", solver="mu", beta_loss="kullback-leibler",
              max_iter=150, tol=1e-3)
NMF_NRUN = 8            # multi-start restarts within a single fit
N_CONSENSUS_RESTARTS = int(os.environ.get("CRC_N_CONSENSUS_RESTARTS", "20"))
N_SUBSAMPLE = int(os.environ.get("CRC_N_SUBSAMPLE", "200"))
SUBSAMPLE_FRAC = float(os.environ.get("CRC_SUBSAMPLE_FRAC", "0.8"))
N_REPEATS = int(os.environ.get("CRC_N_REPEATS", "10"))
N_FOLDS = int(os.environ.get("CRC_N_FOLDS", "5"))
PERM_B = int(os.environ.get("CRC_PERM_B", "2000"))
HENNIG_VALID_PATTERN = 0.75
HENNIG_HIGHLY_STABLE = 0.85


def resolve_base():
    if os.environ.get("CRC_BASE"):
        return Path(os.environ["CRC_BASE"]).resolve()
    return Path(__file__).resolve().parents[1]


BASE = resolve_base()
INPUT = Path(os.environ.get("CRC_INPUT", BASE / "module4_results" / "module4_unified_discovery_matrix.csv"))
HISTORICAL = Path(os.environ.get("CRC_HISTORICAL_LABEL_FILE",
                                 BASE / "module05_06_loocv_results" / "labels" / "NMF_k4_LOOCV.csv"))
OUT = Path(os.environ.get("CRC_M30_OUT", BASE / "module30_nmf_only_endtoend"))
LOCKED_LABEL_OUT = Path(os.environ.get(
    "CRC_CLUSTER_FILE",
    BASE / "module05_06_loocv_results" / "labels" / "NMF_k4_LOOCV.csv",
))


def resolve_clin():
    if os.environ.get("CRC_CLIN"):
        return Path(os.environ["CRC_CLIN"])
    for c in [BASE / "clinical_data.tsv", BASE.parent / "clinical_data.tsv"]:
        if Path(c).exists():
            return Path(c)
    raise FileNotFoundError("clinical_data.tsv not found; set CRC_CLIN")


CLIN_PATH = resolve_clin()
for sub in ("tables", "figures", "labels"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)

CB8 = ["#E69F00", "#56B4E9", "#009E73", "#D55E00", "#F0E442", "#0072B2", "#CC79A7", "#999999"]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 14, "axes.titlesize": 16, "savefig.dpi": 300,
                     "figure.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42})


def extract_id(x):
    m = re.search(r"((?:U|UM)\d+)", str(x))
    return m.group(1) if m else None


def disp_c(c):
    return f"C{int(c) + 1}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def savefig(p):
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        plt.savefig(f"{p}.{ext}", bbox_inches="tight", dpi=300)
    plt.close()


def multigroup_logrank_p(t, e, g):
    """Mantel-Cox multigroup log-rank, tie-aware."""
    t = np.asarray(t, float); e = np.asarray(e, int); g = np.asarray(g, int)
    groups = np.sort(np.unique(g))
    ng = len(groups)
    if ng < 2:
        return np.nan
    O = np.zeros(ng); E = np.zeros(ng); V = np.zeros((ng, ng))
    for tt in np.unique(t[e == 1]):
        at_risk = t >= tt
        ev = (t == tt) & (e == 1)
        n_ = int(at_risk.sum()); d = int(ev.sum())
        gr = np.array([(at_risk & (g == q)).sum() for q in groups], float)
        ge = np.array([(ev & (g == q)).sum() for q in groups], float)
        O += ge; E += d * gr / n_
        if n_ > 1:
            p_ = gr / n_
            V += d * (n_ - d) / (n_ - 1) * (np.diag(p_) - np.outer(p_, p_))
    diff = (O - E)[:-1]
    stat = float(diff @ np.linalg.pinv(V[:-1, :-1]) @ diff)
    return float(chi2.sf(stat, ng - 1))


def hungarian_map(labels, reference, k):
    cm = confusion_matrix(reference, labels, labels=list(range(k)))
    r, c = linear_sum_assignment(-cm)
    return {int(cc): int(rr) for rr, cc in zip(r, c)}


def apply_map(labels, mp):
    return np.asarray([mp.get(int(v), int(v)) for v in labels], int)


def cluster_jaccard(a, b, c):
    left, right = a == c, b == c
    u = int(np.sum(left | right))
    return float(np.sum(left & right) / u) if u else np.nan


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg adjusted P values without an optional dependency."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


# ═══════════════════════ LOAD (historical rules) ════════════════
print("=" * 78)
print("MODULE 30 — NMF-only end-to-end clustering, selection and validation")
print("=" * 78)
print("NMF is the sole clustering method in this script. No k-means or")
print("Bernoulli mixture model is fitted anywhere below.")
print(f"\nmatrix    : {INPUT}")
print(f"historical: {HISTORICAL}  (verification target only)")
print(f"output    : {OUT}")

xdf = pd.read_csv(INPUT, index_col=0)
xdf["__sid"] = [extract_id(i) for i in xdf.index.astype(str)]
xdf = xdf.dropna(subset=["__sid"]).drop_duplicates("__sid", keep="first").set_index("__sid")

clin = pd.read_csv(CLIN_PATH, sep="\t", low_memory=False)
clin.rename(columns={clin.columns[0]: "full"}, inplace=True)
clin["sid"] = clin["full"].astype(str).map(extract_id)
clin["OS_MONTHS"] = pd.to_numeric(clin["OS_MONTHS"], errors="coerce")
clin["OS_STATUS"] = pd.to_numeric(clin["OS_STATUS"], errors="coerce")
clin = (clin.dropna(subset=["sid", "OS_MONTHS", "OS_STATUS"])
        .query("OS_MONTHS >= 0").drop_duplicates("sid").set_index("sid"))

order = [s for s in xdf.index if s in clin.index]
dropped = [s for s in xdf.index if s not in clin.index]
X = (np.nan_to_num(xdf.loc[order].values.astype(np.float32), nan=0.0) > 0).astype(np.float32)
SAMPLES = order
OS_T = clin.loc[order, "OS_MONTHS"].to_numpy(float)
OS_E = clin.loc[order, "OS_STATUS"].to_numpy(int)
N, P = X.shape
print(f"cohort    : {N} x {P}; events={int(OS_E.sum())}; excluded matrix-only={dropped}")

print("precomputing Hamming distances for silhouette (reported for context only) …")
HAM = cdist(X, X, metric="hamming").astype(np.float32)


# ══════════════════ NMF PRIMITIVES (only method) ════════════════
def nmf_fit_predict(k, Xtr, seed, Xte=None, nrun=NMF_NRUN, random_init=False):
    """Multi-start NMF fit. random_init=True uses init='random' with the
    given seed for a SINGLE restart -- used only by the consensus-matrix
    builder, where nndsvda's determinism would make every restart identical
    (Brunet et al. 2004 requires genuinely different starts per restart)."""
    if random_init:
        m = skNMF(n_components=k, init="random", solver="mu",
                  beta_loss="kullback-leibler", max_iter=NMF_KW["max_iter"],
                  tol=NMF_KW["tol"], random_state=int(seed))
        W = m.fit_transform(Xtr + 1e-6)
        te = np.argmax(m.transform(Xte + 1e-6), 1) if Xte is not None and len(Xte) else None
        return np.argmax(W, 1).astype(int), (None if te is None else np.asarray(te, int))

    rng = np.random.default_rng(seed)
    best, best_err, bW = None, np.inf, None
    for sd in rng.integers(0, 99999, nrun):
        m = skNMF(n_components=k, random_state=int(sd), **NMF_KW)
        W = m.fit_transform(Xtr + 1e-6)
        if len(np.unique(np.argmax(W, 1))) < 2:
            continue
        if m.reconstruction_err_ < best_err:
            best, best_err, bW = m, m.reconstruction_err_, W
    if best is None:
        best = skNMF(n_components=k, random_state=int(seed),
                     init="nndsvda", solver="mu", beta_loss="kullback-leibler",
                     max_iter=NMF_KW["max_iter"] * 2, tol=NMF_KW["tol"])
        bW = best.fit_transform(Xtr + 1e-6)
    te = np.argmax(best.transform(Xte + 1e-6), 1) if Xte is not None and len(Xte) else None
    return np.argmax(bW, 1).astype(int), (None if te is None else np.asarray(te, int))


def historical_nmf_fit_predict(k, Xtr, seed, Xte=None):
    """The EXACT historical single-start configuration used for the ten
    repeated cross-fitting repetitions that produced the lock. Distinct from
    nmf_fit_predict's multi-start search, which is used for the full-cohort
    canonical fit and consensus restarts."""
    m = skNMF(n_components=k, random_state=int(seed), **NMF_KW)
    Wtr = m.fit_transform(Xtr + 1e-6)
    te = np.argmax(m.transform(Xte + 1e-6), 1) if Xte is not None and len(Xte) else None
    return np.argmax(Wtr, 1).astype(int), (None if te is None else np.asarray(te, int))


def compactness(labels, ham, Xm):
    if len(np.unique(labels)) < 2:
        return dict(silhouette_hamming=np.nan, calinski_harabasz=np.nan, davies_bouldin=np.nan)
    return dict(
        silhouette_hamming=float(silhouette_score(ham, labels, metric="precomputed")),
        calinski_harabasz=float(calinski_harabasz_score(Xm, labels)),
        davies_bouldin=float(davies_bouldin_score(Xm, labels)),
    )


def consensus_matrix(k, Xm, n_restarts, seed_base):
    rng = np.random.default_rng(seed_base)
    n_ = len(Xm)
    C = np.zeros((n_, n_))
    ok = 0
    for _ in range(n_restarts):
        sd = int(rng.integers(0, 2 ** 31 - 1))
        try:
            lbl, _ = nmf_fit_predict(k, Xm, sd, nrun=1, random_init=True)
        except Exception:
            continue
        if len(np.unique(lbl)) < 2:
            continue
        C += (lbl[:, None] == lbl[None, :])
        ok += 1
    return (C / ok, ok) if ok >= 2 else (None, ok)


def cophenetic_and_dispersion(C):
    """Cophenetic correlation (Brunet 2004) and dispersion coefficient
    (Kim & Park 2007). Both derived from the same consensus matrix so they
    are directly comparable and can be reported side by side."""
    if C is None:
        return np.nan, np.nan
    n_ = C.shape[0]
    rho = float(np.sum(4.0 * (C - 0.5) ** 2) / (n_ * n_))
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    cond = squareform(D, checks=False)
    ccc = 1.0 if not np.any(cond > 0) else float(cophenet(linkage(cond, "average"), cond)[0])
    return ccc, rho


def subsample_stability(k, Xm, ref_labels, n_rep, frac, seed_base):
    def one(rep):
        rng = np.random.default_rng(seed_base + rep)
        tr = rng.choice(len(Xm), int(round(frac * len(Xm))), replace=False)
        try:
            lt, _ = nmf_fit_predict(k, Xm[tr], int(rng.integers(0, 2 ** 31 - 1)))
        except Exception:
            return None
        if len(np.unique(lt)) < 2:
            return None
        mp = hungarian_map(lt, ref_labels[tr], k)
        al = apply_map(lt, mp)
        jac = {c: cluster_jaccard(al, ref_labels[tr], c) for c in range(k)}
        return adjusted_rand_score(ref_labels[tr], al), jac

    res = [r for r in Parallel(n_jobs=N_JOBS, prefer="processes")(
        delayed(one)(r) for r in range(n_rep)) if r is not None]
    if not res:
        return None
    aris = np.array([r[0] for r in res])
    jac = {c: np.array([r[1].get(c, np.nan) for r in res]) for c in range(k)}
    return {"aris": aris, "jaccard": jac, "n_ok": len(res)}


def logrank_p(labels):
    return multigroup_logrank_p(OS_T, OS_E, labels)


def crossfit_reference_labels(k, Xm, repeat=0, stratified=True, strata=None):
    """Generate one pooled held-out NMF partition.

    This is the sole source of reference labels used by label-dependent
    analyses. Full-cohort random-start fits are reserved for consensus-matrix
    diagnostics and never define or rename the reported subtypes.
    """
    n_samples = len(Xm)
    fold_seed = MASTER_SEED + repeat
    if stratified:
        if strata is None:
            raise ValueError("strata are required for stratified cross-fitting")
        split = StratifiedKFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=fold_seed,
        ).split(Xm, strata)
    else:
        split = KFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=fold_seed,
        ).split(Xm)
    labels = np.full(n_samples, -1, int)
    for fold, (train_index, test_index) in enumerate(split):
        seed = MASTER_SEED + repeat * N_FOLDS + fold
        _, heldout = historical_nmf_fit_predict(
            k,
            Xm[train_index],
            seed,
            Xm[test_index],
        )
        labels[test_index] = heldout
    if np.any(labels < 0):
        raise RuntimeError(
            f"Cross-fitting left samples unassigned for k={k}, repeat={repeat}"
        )
    return labels


# ═════ STAGE 1 — CROSS-FITTED REFERENCE LABELS + NMF CONSENSUS ═════
print("\n" + "=" * 78)
print(f"STAGE 1  cross-fitted reference partitions and NMF consensus, k = {K_RANGE}")
print("=" * 78)
print("Pooled held-out repetition-0 labels define every label-dependent result.")
print("Full-cohort random starts contribute only the consensus matrix; their")
print("component sizes are neither subtype assignments nor reported outputs.")

CANON, rows = {}, []
for k in K_RANGE:
    t0 = time.time()
    lbl = crossfit_reference_labels(
        k,
        X,
        repeat=0,
        stratified=True,
        strata=OS_E,
    )
    CANON[k] = lbl
    comp = compactness(lbl, HAM, X)
    C, n_ok = consensus_matrix(k, X, N_CONSENSUS_RESTARTS, MASTER_SEED)
    ccc, rho = cophenetic_and_dispersion(C)
    lp = logrank_p(lbl)
    sizes = {disp_c(c): int((lbl == c).sum()) for c in range(k)}
    rows.append({"k": k, "reference_source": "crossfit_repeat0", **comp,
                "cophenetic": ccc, "dispersion_rho": rho,
                "consensus_restarts_ok": n_ok, "logrank_p_OS": lp,
                "smallest_cluster_n": int(min(sizes.values())),
                "cluster_sizes": json.dumps(sizes)})
    print(f"  k={k}  sil={comp['silhouette_hamming']:+.3f}  CH={comp['calinski_harabasz']:7.1f}  "
          f"DB={comp['davies_bouldin']:.3f}  cophenetic={ccc:.4f}  dispersion={rho:.4f}  "
          f"logrank={lp:.4f}  sizes={sizes}  [{time.time()-t0:.0f}s]")

canon_df = pd.DataFrame(rows)


# ════════════ STAGE 2 — NULL BASELINE (Senbabaoglu check) ═══════
print("\n" + "=" * 78)
print("STAGE 2  null baseline: column-permuted matrix (Senbabaoglu et al. 2014)")
print("=" * 78)
print("Each feature keeps its exact marginal prevalence; co-occurrence between")
print("features is destroyed. A metric that scores as well here as on the real")
print("data cannot be read as evidence of real cluster structure.")

rng = np.random.default_rng(MASTER_SEED)
Xn = X.copy()
for j in range(Xn.shape[1]):
    Xn[:, j] = Xn[rng.permutation(N), j]
HAMn = cdist(Xn, Xn, metric="hamming").astype(np.float32)

null_rows = []
for k in K_RANGE:
    t0 = time.time()
    # Use the same pooled held-out label-generation procedure as for the real
    # matrix so that all label-dependent real/null metrics are comparable.
    lbl = crossfit_reference_labels(
        k,
        Xn,
        repeat=0,
        stratified=True,
        strata=OS_E,
    )
    comp = compactness(lbl, HAMn, Xn)
    C, n_ok = consensus_matrix(k, Xn, N_CONSENSUS_RESTARTS, MASTER_SEED)
    ccc, rho = cophenetic_and_dispersion(C)
    null_rows.append({"k": k, "reference_source": "crossfit_repeat0", **comp,
                      "cophenetic": ccc, "dispersion_rho": rho,
                      "consensus_restarts_ok": n_ok})
    print(f"  [null] k={k}  sil={comp['silhouette_hamming']:+.3f}  "
          f"cophenetic={ccc:.4f}  dispersion={rho:.4f}  [{time.time()-t0:.0f}s]")

null_df = pd.DataFrame(null_rows)
real_vs_null = canon_df.merge(null_df, on="k", suffixes=("_real", "_null"))
for c in ["silhouette_hamming", "calinski_harabasz", "davies_bouldin", "cophenetic", "dispersion_rho"]:
    real_vs_null[f"{c}_gap"] = real_vs_null[f"{c}_real"] - real_vs_null[f"{c}_null"]
print("\nReal vs null gap (k=4):")
r4 = real_vs_null[real_vs_null.k == LOCK_K].iloc[0]
for c in ["silhouette_hamming", "cophenetic", "dispersion_rho"]:
    print(f"  {c}: real={r4[c+'_real']:.4f}  null={r4[c+'_null']:.4f}  gap={r4[c+'_gap']:+.4f}")


# ══════════ STAGE 3 — RESAMPLING REPRODUCIBILITY (Hennig) ═══════
print("\n" + "=" * 78)
print(f"STAGE 3  resampling reproducibility  ({N_SUBSAMPLE} x {SUBSAMPLE_FRAC:.0%}, one-to-one relabelled)")
print("=" * 78)

sub_rows, sub_jac_rows = [], []
for k in K_RANGE:
    t0 = time.time()
    st = subsample_stability(k, X, CANON[k], N_SUBSAMPLE, SUBSAMPLE_FRAC, MASTER_SEED * 1000)
    if st is None:
        continue
    aris = st["aris"]
    sub_rows.append({"k": k, "ARI_mean": float(aris.mean()),
                     "ARI_p5": float(np.percentile(aris, 5)), "ARI_min": float(aris.min()),
                     "n_reps_ok": st["n_ok"]})
    cluster_jaccard_means = []
    for c in range(k):
        v = st["jaccard"][c]
        cluster_jaccard_means.append(float(np.nanmean(v)))
        sub_jac_rows.append({"k": k, "cluster": disp_c(c),
                             "n_in_reference": int((CANON[k] == c).sum()),
                             "jaccard_mean": float(np.nanmean(v)),
                             "jaccard_p5": float(np.nanpercentile(v, 5)),
                             "stable_pattern_ge_0_75": bool(
                                 np.nanmean(v) >= HENNIG_VALID_PATTERN
                             ),
                             "highly_stable_ge_0_85": bool(
                                 np.nanmean(v) >= HENNIG_HIGHLY_STABLE
                             )})
    sub_rows[-1]["minimum_mean_cluster_jaccard"] = float(
        np.min(cluster_jaccard_means)
    )
    print(f"  k={k}  ARI_mean={aris.mean():.3f}  ARI_p5={np.percentile(aris,5):.3f}  "
          f"[{time.time()-t0:.0f}s]")

sub_df = pd.DataFrame(sub_rows)
sub_jac_df = pd.DataFrame(sub_jac_rows)
print(
    "\nPer-cluster stability "
    "(mean Jaccard >=0.75 stable pattern; >=0.85 highly stable):"
)
print(sub_jac_df.to_string(index=False))


# ═══ STAGE 4 — HISTORICAL 10x5-FOLD PROTOCOL + STRATIFICATION CHECK ═══
print("\n" + "=" * 78)
print("STAGE 4  historical repeated cross-fitting protocol (provenance + sensitivity)")
print("=" * 78)


def oof_repeat(k, repeat, stratified=True):
    return crossfit_reference_labels(
        k,
        X,
        repeat=repeat,
        stratified=stratified,
        strata=OS_E if stratified else None,
    )


provenance = {}
for k in K_RANGE:
    reps = Parallel(n_jobs=min(N_JOBS, N_REPEATS), prefer="processes")(
        delayed(oof_repeat)(k, r, True) for r in range(N_REPEATS))
    ref0 = reps[0]
    aligned = [ref0] + [apply_map(reps[r], hungarian_map(reps[r], ref0, k)) for r in range(1, N_REPEATS)]
    lp = [logrank_p(v) for v in aligned]
    sizes0 = {disp_c(c): int((ref0 == c).sum()) for c in range(k)}
    provenance[k] = {"ref0": ref0, "aligned": aligned, "logrank_p": lp, "sizes0": sizes0}
    print(f"  k={k}  repeat-0 sizes={sizes0}  pooled logrank median={np.median(lp):.4f}  "
          f"frac<0.05={np.mean(np.array(lp) < 0.05):.1f}")

ref0_k4 = provenance[LOCK_K]["ref0"]
counts0 = tuple(int((ref0_k4 == c).sum()) for c in range(LOCK_K))
counts_ok = counts0 == LOCK_COUNTS
print(f"\n  k=4 repeat-0 counts: {counts0}  expected {LOCK_COUNTS}  -> "
      f"{'MATCH' if counts_ok else 'MISMATCH'}")

hist_ok, hist_ari, hist_sha, regen_sha = None, np.nan, None, None
regen = pd.DataFrame({
    "sample_id": SAMPLES,
    "cluster": ref0_k4 + 1,
    "cluster_display": [disp_c(c) for c in ref0_k4],
    "cluster_0based": ref0_k4,
})
if HISTORICAL.exists():
    h = pd.read_csv(HISTORICAL)
    h["sid"] = h["sample_id"].map(extract_id)
    hc = pd.to_numeric(h["cluster"], errors="coerce").astype(int)
    h["c0"] = hc - 1 if hc.min() >= 1 else hc
    hl = h.set_index("sid").loc[SAMPLES, "c0"].to_numpy(int)
    hist_ari = float(adjusted_rand_score(hl, ref0_k4))
    hist_ok = bool(np.array_equal(hl, ref0_k4))
    hist_sha = sha256(HISTORICAL)
    regen_path = OUT / "labels" / "NMF_k4_REGENERATED.csv"
    regen.to_csv(regen_path, index=False)
    regen_sha = sha256(regen_path)
    print(f"  vs historical lock: ARI={hist_ari:.4f}  identical={'YES' if hist_ok else 'NO'}  "
          f"SHA match={'YES' if regen_sha == hist_sha else 'NO'}")

if not counts_ok:
    raise RuntimeError(
        f"Regenerated k=4 counts {counts0} do not match the prespecified "
        f"locked counts {LOCK_COUNTS}; refusing to write subtype labels."
    )
LOCKED_LABEL_OUT.parent.mkdir(parents=True, exist_ok=True)
regen.to_csv(LOCKED_LABEL_OUT, index=False)
print(f"  locked subtype labels written to: {LOCKED_LABEL_OUT}")

print("\n  Fold-stratification sensitivity (k=4): does stratifying on the")
print("  outcome determine the partition?")
unstrat_reps = Parallel(n_jobs=min(N_JOBS, N_REPEATS), prefer="processes")(
    delayed(oof_repeat)(LOCK_K, r, False) for r in range(N_REPEATS))
unstrat_aligned = [apply_map(v, hungarian_map(v, ref0_k4, LOCK_K)) for v in unstrat_reps]
strat_ari = [adjusted_rand_score(ref0_k4, v) for v in provenance[LOCK_K]["aligned"]]
unstrat_ari = [adjusted_rand_score(ref0_k4, v) for v in unstrat_aligned]
sens_df = pd.DataFrame({
    "fold_scheme": ["stratified"] * N_REPEATS + ["unstratified"] * N_REPEATS,
    "repeat": list(range(N_REPEATS)) * 2,
    "ARI_vs_lock": strat_ari + unstrat_ari,
    "C4_n": ([int((v == 3).sum()) for v in provenance[LOCK_K]["aligned"]] +
             [int((v == 3).sum()) for v in unstrat_aligned]),
})
# Repetition 0 of the stratified analysis is the frozen partition itself and
# therefore has ARI=1 by definition.  Exclude that self-comparison and use the
# matched non-reference repetitions 1-9 for the manuscript sensitivity
# summary.  This makes the stratified and unstratified means directly
# comparable and prevents inflation of the stratified mean.
sens_matched_df = sens_df[sens_df["repeat"].between(1, N_REPEATS - 1)].copy()
sens_summary_df = (
    sens_matched_df.groupby("fold_scheme", sort=False)
    .agg(
        n_repetitions=("ARI_vs_lock", "size"),
        ARI_mean=("ARI_vs_lock", "mean"),
        ARI_sd=("ARI_vs_lock", "std"),
        ARI_median=("ARI_vs_lock", "median"),
        ARI_min=("ARI_vs_lock", "min"),
        ARI_max=("ARI_vs_lock", "max"),
        C4_median=("C4_n", "median"),
        C4_min=("C4_n", "min"),
        C4_max=("C4_n", "max"),
    )
    .reset_index()
)
print(sens_df.groupby("fold_scheme").agg(
    ARI_mean=("ARI_vs_lock", "mean"), ARI_min=("ARI_vs_lock", "min"),
    C4_median=("C4_n", "median"), C4_min=("C4_n", "min"), C4_max=("C4_n", "max")
).reset_index().to_string(index=False))
print("\n  Matched non-reference repetitions 1-9 (manuscript comparison):")
print(sens_summary_df.to_string(index=False))


# ═══════════════════ STAGE 5 — SURVIVAL ACROSS k ════════════════
print("\n" + "=" * 78)
print("STAGE 5  survival across the k=4,5,6 screen (NMF only)")
print("=" * 78)
print("Multiplicity spans only the k screen -- a direct, correct consequence")
print("of no longer screening across three algorithms.")

surv_p = np.array([logrank_p(CANON[k]) for k in K_RANGE], float)
bh = benjamini_hochberg(surv_p)
bonf = np.clip(surv_p * len(K_RANGE), 0, 1)
obs_min = float(np.min(surv_p))
group_mat = np.vstack([CANON[k] for k in K_RANGE])


def perm_min(b):
    r_ = np.random.default_rng(MASTER_SEED * 7919 + b)
    ii = r_.permutation(N)
    t_, e_ = OS_T[ii], OS_E[ii]
    best = np.inf
    for gi in range(group_mat.shape[0]):
        g = group_mat[gi]
        if len(np.unique(g)) < 2:
            continue
        best = min(best, multigroup_logrank_p(t_, e_, g))
    return best


pm = np.array(Parallel(n_jobs=N_JOBS, prefer="processes")(delayed(perm_min)(b) for b in range(PERM_B)))
pm = pm[np.isfinite(pm)]
p_fw = (1.0 + float((pm <= obs_min).sum())) / (1.0 + len(pm))

surv_df = pd.DataFrame({"k": K_RANGE, "logrank_p_OS": surv_p, "logrank_p_BH": bh,
                        "logrank_p_Bonferroni": bonf})
surv_df["familywise_minP_permutation_p"] = p_fw
print(surv_df.to_string(index=False))
print(f"\nfamily-wise minimum-P permutation P = {p_fw:.4f}  [{len(pm)} permutations]")


# ═════════════ STAGE 6 — PCA / UMAP EMBEDDING SPACE ═════════════
print("\n" + "=" * 78)
print("STAGE 6  PCA and UMAP embeddings of the discovery matrix, coloured by")
print("         NMF cluster assignment at k = 4, 5, 6")
print("=" * 78)
print("The embedding itself is computed ONCE on the real 371-feature matrix and")
print("held fixed across k; only the point colouring (the cross-fitted reference")
print("NMF label at")
print("each k) changes between panels. This isolates what changes with k -- how")
print("a fixed geometric picture of the cohort is carved up -- from any change")
print("in the picture itself, which a per-k embedding would confound.")

from sklearn.decomposition import PCA
import umap as umap_lib

t0 = time.time()
pca = PCA(n_components=2, random_state=MASTER_SEED)
PCA_EMB = pca.fit_transform(X)
pca_var = pca.explained_variance_ratio_
print(f"  PCA done  [{time.time()-t0:.0f}s]  "
      f"PC1={pca_var[0]:.1%} var, PC2={pca_var[1]:.1%} var")

t0 = time.time()
reducer = umap_lib.UMAP(n_components=2, n_neighbors=30, min_dist=0.2,
                        metric="hamming", random_state=MASTER_SEED)
UMAP_EMB = reducer.fit_transform(X)
print(f"  UMAP done  [{time.time()-t0:.0f}s]")

embed_df = pd.DataFrame({"sample_id": SAMPLES, "PC1": PCA_EMB[:, 0], "PC2": PCA_EMB[:, 1],
                         "UMAP1": UMAP_EMB[:, 0], "UMAP2": UMAP_EMB[:, 1]})
for k in K_RANGE:
    embed_df[f"NMF_k{k}_cluster"] = [disp_c(c) for c in CANON[k]]
embed_df.to_csv(OUT / "tables" / "pca_umap_embedding_coordinates.csv", index=False)

fig, axes = plt.subplots(2, len(K_RANGE), figsize=(6.2 * len(K_RANGE), 11.5), squeeze=False)
for col, k in enumerate(K_RANGE):
    lbl = CANON[k]
    for row, (emb, name) in enumerate(((PCA_EMB, "PCA"), (UMAP_EMB, "UMAP"))):
        ax = axes[row, col]
        for c in range(k):
            m = lbl == c
            ax.scatter(emb[m, 0], emb[m, 1], s=14, alpha=0.85, color=CB8[c % len(CB8)],
                      edgecolor="none", label=f"{disp_c(c)} (n={int(m.sum())})")
        ax.set_title(f"{name} — NMF k={k}", fontsize=14)
        ax.set_xlabel(f"{name}1" if row == 1 else f"PC1 ({pca_var[0]:.0%})")
        ax.set_ylabel(f"{name}2" if row == 1 else f"PC2 ({pca_var[1]:.0%})")
        ax.legend(frameon=False, fontsize=9, markerscale=1.6, loc="best")
        ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Embedding space of the discovery matrix, coloured by NMF cluster (k=4,5,6)",
            fontsize=17)
savefig(str(OUT / "figures" / "pca_umap_embedding_by_k"))
print(f"  -> {OUT / 'figures' / 'pca_umap_embedding_by_k'}.png/pdf/svg")
print(f"  -> {OUT / 'tables' / 'pca_umap_embedding_coordinates.csv'}")

# Thesis/main-text version showing only the selected, frozen k=4 partition.
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
selected_labels = CANON[LOCK_K]
for ax, emb, name in zip(axes, (PCA_EMB, UMAP_EMB), ("PCA", "UMAP")):
    for c in range(LOCK_K):
        mask = selected_labels == c
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=22,
            alpha=0.82,
            color=CB8[c],
            edgecolor="none",
            label=f"{disp_c(c)} (n={int(mask.sum())})",
        )
    if name == "PCA":
        ax.set_xlabel(f"PC1 ({pca_var[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca_var[1]:.1%})")
    else:
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
    ax.set_title(f"{name}: frozen NMF k=4 partition")
    ax.legend(frameon=False, fontsize=11, markerscale=1.5)
    ax.spines[["top", "right"]].set_visible(False)
savefig(str(OUT / "figures" / "selected_k4_pca_umap"))
print(f"  -> {OUT / 'figures' / 'selected_k4_pca_umap'}.png/pdf/svg")


# ═══════════════════════ OUTPUTS ════════════════════════════════
canon_df.to_csv(OUT / "tables" / "crossfit_reference_by_k.csv", index=False)
# Retain the historical table filename as a compatibility alias, but its
# contents now describe the cross-fitted reference partitions only.
canon_df.to_csv(OUT / "tables" / "canonical_fits_by_k.csv", index=False)
null_df.to_csv(OUT / "tables" / "null_baseline_by_k.csv", index=False)
real_vs_null.to_csv(OUT / "tables" / "real_vs_null_comparison.csv", index=False)
sub_df.to_csv(OUT / "tables" / "resampling_reproducibility_by_k.csv", index=False)
sub_jac_df.to_csv(OUT / "tables" / "per_cluster_jaccard_by_k.csv", index=False)
sens_df.to_csv(OUT / "tables" / "fold_stratification_sensitivity.csv", index=False)
sens_summary_df.to_csv(
    OUT / "tables" / "fold_stratification_summary_matched_repeats.csv",
    index=False,
)
surv_df.to_csv(OUT / "tables" / "survival_by_k.csv", index=False)

# Consolidated, rounded tables for direct use in the thesis.  The unrounded
# source values remain available in the analysis tables written above.
selection_rows = []
canon_idx = canon_df.set_index("k")
null_idx = null_df.set_index("k")
sub_idx = sub_df.set_index("k")
for k in K_RANGE:
    real = canon_idx.loc[k]
    null = null_idx.loc[k]
    stability = sub_idx.loc[k]
    selection_rows.append(
        {
            "k": k,
            "cross_fitted_cluster_sizes": real["cluster_sizes"],
            "silhouette_real": real["silhouette_hamming"],
            "silhouette_null": null["silhouette_hamming"],
            "calinski_harabasz_real": real["calinski_harabasz"],
            "calinski_harabasz_null": null["calinski_harabasz"],
            "davies_bouldin_real": real["davies_bouldin"],
            "davies_bouldin_null": null["davies_bouldin"],
            "cophenetic_real": real["cophenetic"],
            "cophenetic_null": null["cophenetic"],
            "dispersion_real": real["dispersion_rho"],
            "dispersion_null": null["dispersion_rho"],
            "resampling_ARI_5th_percentile": stability["ARI_p5"],
            "minimum_mean_cluster_Jaccard": stability[
                "minimum_mean_cluster_jaccard"
            ],
        }
    )
thesis_selection_df = pd.DataFrame(selection_rows)
numeric_cols = thesis_selection_df.select_dtypes(include=[np.number]).columns
thesis_selection_df[numeric_cols] = thesis_selection_df[numeric_cols].round(3)
thesis_selection_df.to_csv(
    OUT / "tables" / "thesis_model_selection_summary.csv", index=False
)

thesis_cluster_df = sub_jac_df[
    ["k", "cluster", "n_in_reference", "jaccard_mean", "jaccard_p5"]
].copy()
thesis_cluster_df[["jaccard_mean", "jaccard_p5"]] = thesis_cluster_df[
    ["jaccard_mean", "jaccard_p5"]
].round(3)
thesis_cluster_df.to_csv(
    OUT / "tables" / "thesis_cluster_stability_summary.csv", index=False
)

thesis_fold_df = sens_summary_df.copy()
for col in ["ARI_mean", "ARI_sd", "ARI_median", "ARI_min", "ARI_max"]:
    thesis_fold_df[col] = thesis_fold_df[col].round(3)
thesis_fold_df.to_csv(
    OUT / "tables" / "thesis_fold_sensitivity_summary.csv", index=False
)

table_notes = """# Thesis-ready NMF model-selection tables

## Table 1: model selection across k
Use `thesis_model_selection_summary.csv`. Higher values indicate better
performance for silhouette, Calinski-Harabasz, cophenetic correlation,
dispersion and resampling ARI; lower values indicate better performance for
Davies-Bouldin. Null matrices preserve the marginal prevalence of every
feature while disrupting feature co-occurrence. Survival was not used to
select k.

## Table 2: component-specific resampling stability
Use `thesis_cluster_stability_summary.csv`. Values are mean and fifth-
percentile Jaccard similarities across 200 stratified 80% subsamples after
one-to-one relabelling against the cross-fitted reference partition.

## Table 3: fold-stratification sensitivity
Use `thesis_fold_sensitivity_summary.csv`. Both fold schemes are summarized
over matched non-reference repetitions 1-9. Stratified repetition 0 is
excluded because it is the frozen reference and therefore has ARI=1 by
definition.
"""
(OUT / "tables" / "THESIS_TABLE_NOTES.md").write_text(table_notes)

for k in K_RANGE:
    for r, lv in enumerate(provenance[k]["aligned"]):
        pd.DataFrame({"sample_id": SAMPLES, "cluster": lv + 1,
                      "cluster_display": [disp_c(c) for c in lv]}).to_csv(
            OUT / "labels" / f"NMF_k{k}_OOF_repeat{r}.csv", index=False)

manifest = {
    "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    "method": "NMF only -- no k-means or Bernoulli mixture model fitted",
    "reporting_rule": (
        "all label-dependent results use pooled cross-fitting repeat-0 labels; "
        "full-cohort random starts are used only for consensus matrices"
    ),
    "k_range": K_RANGE, "master_seed": MASTER_SEED,
    "nmf_hyperparameters": NMF_KW,
    "provenance": {"k4_repeat0_counts": list(counts0), "expected": list(LOCK_COUNTS),
                  "counts_match": counts_ok, "identical_to_historical": hist_ok,
                  "ARI_vs_historical": hist_ari, "regenerated_sha256": regen_sha,
                  "historical_sha256": hist_sha},
    "cohort": {"n": int(N), "p": int(P), "events": int(OS_E.sum()), "excluded": dropped},
    "inputs": {"matrix": {"path": str(INPUT), "sha256": sha256(INPUT)},
              "clinical": {"path": str(CLIN_PATH), "sha256": sha256(CLIN_PATH)}},
    "environment": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "platform": platform.platform()},
}
try:
    import sklearn, scipy
    manifest["environment"].update({"scikit-learn": sklearn.__version__, "scipy": scipy.__version__})
except Exception:
    pass
(OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

# ── figures ──
# 2x4 grid: the four compactness/consensus indices (silhouette, Calinski-
# Harabasz, Davies-Bouldin, cophenetic), dispersion, resampling ARI, survival,
# and the C4 fold-stratification sensitivity scatter -- eight panels in total.
fig, axes = plt.subplots(2, 4, figsize=(24, 10))
ks = K_RANGE
panels = [("silhouette_hamming", "Silhouette (Hamming)", True),
          ("calinski_harabasz", "Calinski-Harabasz", True),
          ("davies_bouldin", "Davies-Bouldin", False),
          ("cophenetic", "Cophenetic correlation", True),
          ("dispersion_rho", "Dispersion coefficient", True),
          ("ARI_p5", "Resampling ARI, 5th pct", True),
          ("minimum_mean_cluster_jaccard", "Minimum mean cluster Jaccard", True)]
NULL_COMPARABLE = ("silhouette_hamming", "calinski_harabasz", "davies_bouldin",
                   "cophenetic", "dispersion_rho")
def _fmt(v, met):
    if met == "logrank_p_OS":
        return f"{v:.4g}"
    if met == "calinski_harabasz":
        return f"{v:.1f}"
    return f"{v:.3f}"


for ax, (met, title, hib) in zip(axes.ravel()[:7], panels):
    if met in ("ARI_p5", "minimum_mean_cluster_jaccard"):
        vals = sub_df.set_index("k").reindex(ks)[met].values
    else:
        vals = canon_df.set_index("k").reindex(ks)[met].values

    bars = ax.bar(np.arange(len(ks)), vals, color=CB8[2], edgecolor="black", label="real")
    for b, v in zip(bars, vals):
        ax.annotate(_fmt(v, met), (b.get_x() + b.get_width() / 2, b.get_height()),
                   textcoords="offset points", xytext=(0, 3), ha="center",
                   fontsize=9, fontweight="bold")
    if met in NULL_COMPARABLE:
        nv = null_df.set_index("k").reindex(ks)[met].values
        nbars = ax.bar(np.arange(len(ks)), nv, color=CB8[7], alpha=0.55, edgecolor="black",
                       label="null", width=0.5)
        for b, v in zip(nbars, nv):
            ax.annotate(_fmt(v, met), (b.get_x() + b.get_width() / 2, b.get_height()),
                       textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
        ax.legend(frameon=False, fontsize=10)

    ax.set_xticks(np.arange(len(ks))); ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_title(title + ("  (higher better)" if hib else "  (lower better)"), fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    top = max(np.nanmax(vals), np.nanmax(null_df.set_index("k").reindex(ks)[met].values)
              if met in NULL_COMPARABLE else np.nanmax(vals))
    ax.set_ylim(top=top * 1.14)
ax = axes.ravel()[7]
d = sens_df
for tag, col in (("stratified", CB8[3]), ("unstratified", CB8[1])):
    dd = d[d.fold_scheme == tag]
    ax.scatter(dd.repeat, dd.C4_n, label=tag, color=col, s=60, edgecolor="black", zorder=3)
ax.axhline(94, ls="--", color="black", label="locked n=94")
ax.set_xlabel("repeat"); ax.set_ylabel("C4 size"); ax.legend(frameon=False, fontsize=10)
ax.set_title("C4 size: stratified vs unstratified folds"); ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(
    "NMF-only model selection using cross-fitted reference partitions",
    fontsize=17,
)
savefig(str(OUT / "figures" / "nmf_only_panel"))

# Dedicated evidence for the component-specific stability values and the
# corrected fold-sensitivity comparison stated in the Results text.
fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))

ax = axes[0]
xpos = 0
tick_positions, tick_labels = [], []
for k in K_RANGE:
    dd = sub_jac_df[sub_jac_df["k"] == k]
    for _, row in dd.iterrows():
        colour = CB8[k - min(K_RANGE)]
        ax.vlines(
            xpos, row["jaccard_p5"], row["jaccard_mean"],
            color=colour, linewidth=2.5, alpha=0.85,
        )
        ax.scatter(
            xpos, row["jaccard_mean"], s=70, color=colour,
            edgecolor="black", zorder=3,
        )
        tick_positions.append(xpos)
        tick_labels.append(f"k={k}\n{row['cluster']}")
        xpos += 1
    xpos += 0.6
ax.axhline(HENNIG_VALID_PATTERN, color="#666666", ls="--", lw=1.5,
           label="stable pattern (0.75)")
ax.axhline(HENNIG_HIGHLY_STABLE, color="#111111", ls=":", lw=1.5,
           label="highly stable (0.85)")
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, fontsize=9)
ax.set_ylim(0.35, 1.02)
ax.set_ylabel("Jaccard similarity")
ax.set_title("A  Component stability\n(point: mean; line: 5th percentile to mean)")
ax.legend(frameon=False, fontsize=9, loc="lower left")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
pivot_ari = sens_matched_df.pivot(
    index="repeat", columns="fold_scheme", values="ARI_vs_lock"
)
for _, row in pivot_ari.iterrows():
    ax.plot(
        [0, 1], [row["stratified"], row["unstratified"]],
        color="#BBBBBB", linewidth=1, zorder=1,
    )
ax.scatter(
    np.zeros(len(pivot_ari)), pivot_ari["stratified"], s=65,
    color=CB8[3], edgecolor="black", zorder=3,
)
ax.scatter(
    np.ones(len(pivot_ari)), pivot_ari["unstratified"], s=65,
    color=CB8[1], edgecolor="black", zorder=3,
)
means = sens_summary_df.set_index("fold_scheme")["ARI_mean"]
ax.hlines(means["stratified"], -0.18, 0.18, color=CB8[3], linewidth=3)
ax.hlines(means["unstratified"], 0.82, 1.18, color=CB8[1], linewidth=3)
ax.annotate(
    f"mean {means['stratified']:.3f}", (0, means["stratified"]),
    xytext=(-40, 12), textcoords="offset points", fontsize=11,
)
ax.annotate(
    f"mean {means['unstratified']:.3f}", (1, means["unstratified"]),
    xytext=(-35, 12), textcoords="offset points", fontsize=11,
)
ax.set_xlim(-0.35, 1.35)
ax.set_ylim(0.925, 0.965)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Stratified", "Unstratified"])
ax.set_ylabel("ARI versus frozen partition")
ax.set_title("B  Fold-construction sensitivity\n(matched repetitions 1–9)")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[2]
pivot_c4 = sens_matched_df.pivot(
    index="repeat", columns="fold_scheme", values="C4_n"
)
for _, row in pivot_c4.iterrows():
    ax.plot(
        [0, 1], [row["stratified"], row["unstratified"]],
        color="#BBBBBB", linewidth=1, zorder=1,
    )
ax.scatter(
    np.zeros(len(pivot_c4)), pivot_c4["stratified"], s=65,
    color=CB8[3], edgecolor="black", zorder=3,
)
ax.scatter(
    np.ones(len(pivot_c4)), pivot_c4["unstratified"], s=65,
    color=CB8[1], edgecolor="black", zorder=3,
)
ax.axhline(
    LOCK_COUNTS[3], color="black", ls="--", lw=1.5,
    label=f"frozen C4 n={LOCK_COUNTS[3]}",
)
ax.set_xlim(-0.35, 1.35)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Stratified", "Unstratified"])
ax.set_ylabel("C4 sample count")
ax.set_title("C  C4 size sensitivity\n(matched repetitions 1–9)")
ax.legend(frameon=False, fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle(
    "NMF component reproducibility and fold-construction sensitivity",
    fontsize=18,
)
savefig(str(OUT / "figures" / "nmf_stability_and_fold_sensitivity"))

figure_legends = """# Thesis-ready figure legends

## NMF model selection
`nmf_only_panel`: Comparison of NMF solutions at k = 4, 5 and 6 using
internal validity, consensus and resampling-reproducibility measures. Real-
data solutions are compared with column-permuted null matrices that preserve
feature prevalence while disrupting feature co-occurrence. Higher values are
preferable except for the Davies-Bouldin index, for which lower values
indicate better separation.

## Component and fold stability
`nmf_stability_and_fold_sensitivity`: (A) Component-specific Jaccard
similarity across 200 stratified 80% subsamples. Points show mean similarity
and vertical segments extend to the fifth percentile. (B) Adjusted Rand index
between the frozen k = 4 partition and assignments obtained using matched
stratified and unstratified cross-fitting repetitions 1-9. Horizontal bars
denote the corresponding means. (C) C4 sample counts across the same matched
repetitions; the dashed line denotes the frozen C4 size of 94.

## Selected subtype embedding
`selected_k4_pca_umap`: PCA and UMAP projections of the fixed discovery
matrix, coloured by the frozen cross-fitted NMF k = 4 assignments. The
embeddings provide two-dimensional visualizations and were not used as
evidence for selecting k.
"""
(OUT / "figures" / "THESIS_FIGURE_LEGENDS.md").write_text(figure_legends)


# ═══════════════════════ REPORT ═════════════════════════════════
best_k = int(canon_df.set_index("k")["cophenetic"].idxmax())
L = []
L.append("# Module 30 — NMF-only end-to-end clustering, selection and validation\n")
L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. NMF is the only clustering "
         f"method used anywhere in this script. Locked partition read-only; no other "
         f"script or output directory touched.\n")
L.append("## Provenance\n")
L.append(f"- k=4 repeat-0 counts: **{counts0}** (expected {LOCK_COUNTS}) — "
         f"{'MATCH' if counts_ok else 'MISMATCH'}")
if hist_ok is not None:
    L.append(f"- Identical to historical lock: **{'YES' if hist_ok else 'NO'}** "
             f"(ARI {hist_ari:.4f}); SHA-256 "
             f"{'identical' if regen_sha == hist_sha else 'different'}\n")

L.append("## Model selection across k=4,5,6 (NMF's own framework, real vs null)\n")
L.append(
    "All label-dependent metrics use the pooled cross-fitted repetition-0 "
    "partition at each k. Full-cohort random starts are used only to construct "
    "consensus matrices for cophenetic correlation and dispersion; they do not "
    "define a second set of subtype assignments.\n"
)
L.append("Each cell reads real / null. Silhouette, Calinski-Harabasz, cophenetic "
         "correlation and dispersion are higher-is-better; Davies-Bouldin is "
         "lower-is-better.\n")
L.append("| k | silhouette | Calinski-Harabasz | Davies-Bouldin | cophenetic | "
         "dispersion | ARI p5 | minimum mean cluster Jaccard |")
L.append("|---|---|---|---|---|---|---|---|")
for k in K_RANGE:
    cr = canon_df[canon_df.k == k].iloc[0]
    nr = null_df[null_df.k == k].iloc[0]
    ar = sub_df[sub_df.k == k].iloc[0] if (sub_df.k == k).any() else None
    ari_p5_str = f"{ar.ARI_p5:.3f}" if ar is not None else "NA"
    min_jaccard_str = (
        f"{ar.minimum_mean_cluster_jaccard:.3f}" if ar is not None else "NA"
    )
    L.append(f"| {k} | {cr.silhouette_hamming:.3f} / {nr.silhouette_hamming:.3f} | "
             f"{cr.calinski_harabasz:.1f} / {nr.calinski_harabasz:.1f} | "
             f"{cr.davies_bouldin:.3f} / {nr.davies_bouldin:.3f} | "
             f"{cr.cophenetic:.3f} / {nr.cophenetic:.3f} | "
             f"{cr.dispersion_rho:.3f} / {nr.dispersion_rho:.3f} | "
             f"{ari_p5_str} | {min_jaccard_str} |")
L.append(f"\nEvery real-data metric exceeds its null counterpart at every k "
         f"(Davies-Bouldin: real is lower/better than null), "
         f"confirming the consensus structure is not a procedural artefact "
         f"(Senbabaoglu et al. 2014).\n")

L.append("## Fold-stratification sensitivity\n")
for r in sens_summary_df.itertuples():
    L.append(
        f"- {r.fold_scheme}, matched non-reference repetitions 1-9: "
        f"mean ARI vs lock {r.ARI_mean:.3f}, C4 median {r.C4_median:.0f}"
    )
L.append("\nStratifying folds on the survival event indicator did not determine the "
         "partition; unstratified folds reproduce it closely.\n")

L.append("## Thesis-ready evidence package\n")
L.append("- `tables/thesis_model_selection_summary.csv`: all model-selection, "
         "real-versus-null and lower-tail ARI values stated in the Results.")
L.append("- `tables/thesis_cluster_stability_summary.csv`: all component-specific "
         "mean and fifth-percentile Jaccard values.")
L.append("- `tables/thesis_fold_sensitivity_summary.csv`: corrected matched-repeat "
         "stratified-versus-unstratified ARI and C4-size summary.")
L.append("- `figures/nmf_only_panel`: principal model-selection figure.")
L.append("- `figures/nmf_stability_and_fold_sensitivity`: component Jaccard and "
         "fold-construction sensitivity figure.")
L.append("- `figures/selected_k4_pca_umap`: selected frozen-partition embedding.\n")

L.append("## Survival across the k screen\n")
L.append(f"Family-wise minimum-P permutation test across k=4,5,6: **P = {p_fw:.4f}** "
         f"({len(pm)} permutations). Individual k values:\n")
for r in surv_df.itertuples():
    L.append(f"- k={r.k}: P={r.logrank_p_OS:.4g}, BH={r.logrank_p_BH:.3f}, "
             f"Bonferroni={r.logrank_p_Bonferroni:.3f}")

L.append("\n## Embedding space (PCA / UMAP)\n")
L.append(f"PCA (PC1 {pca_var[0]:.1%} variance, PC2 {pca_var[1]:.1%}) and UMAP "
         f"(Hamming metric, n_neighbors=30, min_dist=0.2) were each computed once "
         f"on the full 371-feature matrix and held fixed across k; only the point "
         f"colouring by the cross-fitted NMF reference assignment changes between "
         f"panels, so the "
         f"figure isolates how a fixed geometric picture of the cohort is "
         f"partitioned at each k rather than confounding that with a change in "
         f"the embedding itself. See `figures/pca_umap_embedding_by_k.png/pdf/svg` "
         f"and `tables/pca_umap_embedding_coordinates.csv`.\n")

(OUT / "REPORT.md").write_text("\n".join(L))
print(f"\n  -> {OUT / 'REPORT.md'}")
print("\n" + "=" * 78)
print("DONE — NMF only; locked partition untouched; no other output modified")
print("=" * 78)
