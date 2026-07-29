#!/usr/bin/env python3
"""Assess cross-platform portability of the genomic subtypes in TCGA COAD/READ.

Coverage-matched feature transfer and exploratory de novo NMF are evaluated for
OS and DFS. The analysis treats TCGA as a portability stress test because SV
coverage is incomplete, not as like-for-like external WGS validation.
"""

import os
import re
import gzip
import json
import time
import zlib
import warnings
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, adjusted_rand_score
from sklearn.decomposition import NMF
from statsmodels.stats.multitest import multipletests

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
from lifelines.utils import concordance_index

warnings.filterwarnings("ignore")

##############################################################################
# STYLE — publication-grade, large fonts, colour-blind-safe palette (PPT)
##############################################################################
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 18,
    "axes.titlesize": 24,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 15,
    "savefig.dpi": 300,
    "axes.linewidth": 1.1,
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Dedicated per-cluster identity colours, locked to the project convention:
# C1=orange, C2=sky-blue, C3=bluish-green, C4=vermillion (high-risk).
CLUSTER_COLORS = {"C1": "#E69F00", "C2": "#56B4E9", "C3": "#009E73", "C4": "#D55E00"}
# De-novo clusters get a neutral qualitative set (identity is arbitrary).
DENOVO_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#D55E00", "#0072B2", "#CC79A7"]
ABLATION_COLORS = {
    "SNV":        "#E69F00",
    "CNV":        "#0072B2",
    "SV":         "#009E73",
    "SNV+CNV":    "#56B4E9",
    "SNV+SV":     "#CC79A7",
    "CNV+SV":     "#D55E00",
    "SNV+CNV+SV": "#000000",
}

##############################################################################
# CONFIGURATION  (all paths absolute; override via environment variables)
##############################################################################
BASE = os.environ.get("CRC_BASE", str(Path(__file__).resolve().parents[1]))

DISCOVERY_MATRIX = os.environ.get(
    "CRC_DISCOVERY_MATRIX",
    f"{BASE}/module4_results/module4_unified_discovery_matrix.csv",
)
CLUSTER_FILE = os.environ.get(
    "CRC_CLUSTER_FILE",
    f"{BASE}/module05_06_loocv_results/labels/NMF_k4_LOOCV.csv",
)

OUTDIR    = os.environ.get("CRC_M22_OUT", f"{BASE}/module22_tcga_ablation_validation_C1C4_ppt")
CACHE_DIR = os.environ.get("CRC_M22_CACHE", f"{OUTDIR}/tcga_raw")
FIGDIR    = f"{OUTDIR}/figures"
TABDIR    = f"{OUTDIR}/tables"
for d in (OUTDIR, CACHE_DIR, FIGDIR, TABDIR):
    os.makedirs(d, exist_ok=True)

# Analysis parameters
CNV_SEG_LOG2_THRESH = float(os.environ.get("CRC_M22_SEG_THRESH", "0.30"))  # |log2| for arm segment "altered"
ARM_ALTERED_FRAC    = float(os.environ.get("CRC_M22_ARM_FRAC", "0.50"))    # arm called if >= this fraction altered
N_BOOTSTRAP         = int(os.environ.get("CRC_M22_NBOOT", "1000"))
RANDOM_STATE        = int(os.environ.get("CRC_M22_SEED", "0"))
K_SUBTYPES          = 4
TRY_CBIOPORTAL_SV   = os.environ.get("CRC_M22_TRY_SV", "1") == "1"
DOWNLOAD_TIMEOUT    = int(os.environ.get("CRC_M22_TIMEOUT", "120"))
DOWNLOAD_RETRIES    = int(os.environ.get("CRC_M22_RETRIES", "3"))

XENA_TCGA = "https://tcga.xenahubs.net/download"
XENA_GDC  = "https://gdc.xenahubs.net/download"
CBIO_API  = "https://www.cbioportal.org/api"
CBIO_STUDY = "coadread_tcga_pan_can_atlas_2018"

# MC3 MAF Variant_Classification -> discovery "functional" (Module 1 VEP terms)
FUNCTIONAL_MAF = {
    "Missense_Mutation",       # missense_variant
    "Frame_Shift_Del",         # frameshift_variant
    "Frame_Shift_Ins",         # frameshift_variant
    "Nonsense_Mutation",       # stop_gained
    "Splice_Site",             # splice_acceptor/donor_variant
    "Splice_Region",           # (kept; borderline but functional)
    "In_Frame_Del",            # inframe_deletion
    "In_Frame_Ins",            # inframe_insertion
    "Translation_Start_Site",  # start_lost
    "Nonstop_Mutation",        # stop_lost
    "Protein_Altering",        # protein_altering_variant (rare label variant)
}

# -----------------------------------------------------------------------------
# GENOME BUILD  (critical for the arm-level CNV rule)
# -----------------------------------------------------------------------------
# The arm-CNV rule needs chromosome/centromere coordinates, which are
# BUILD-SPECIFIC. The build is dictated by WHERE the segment file comes from,
# not by "TCGA" in the abstract:
#   * UCSC Xena LEGACY TCGA hub (tcga.xenahubs.net, the SNP6_nocnv_genomicSegment
#     datasets this script uses by default)  -> hg19 / GRCh37.
#   * UCSC Xena GDC hub (gdc.xenahubs.net, TCGA-*.masked_cnv_DNAcopy.tsv) or
#     anything downloaded from the GDC portal / TCGAbiolinks -> hg38 / GRCh38.
# Set CRC_M22_GENOME_BUILD to "hg19" (default, matches the legacy-hub segment
# files below) or "hg38" if you swap the CNV source to a GDC/hg38 segment file.
# A runtime sanity check (see check_build_consistency) flags any mismatch
# between the declared build and the actual segment coordinates so this can
# never fail silently.
GENOME_BUILD = os.environ.get("CRC_M22_GENOME_BUILD", "hg19").lower()

# hg19 / GRCh37 chromosome lengths (bp) and approximate centromere midpoints.
HG19_CHR_LEN = {
    "1": 249250621, "2": 243199373, "3": 198022430, "4": 191154276,
    "5": 180915260, "6": 171115067, "7": 159138663, "8": 146364022,
    "9": 141213431, "10": 135534747, "11": 135006516, "12": 133851895,
    "13": 115169878, "14": 107349540, "15": 102531392, "16": 90354753,
    "17": 81195210, "18": 78077248, "19": 59128983, "20": 63025520,
    "21": 48129895, "22": 51304566, "X": 155270560,
}
HG19_CENTROMERE = {
    "1": 125000000, "2": 93300000, "3": 91000000, "4": 50400000,
    "5": 48400000, "6": 61000000, "7": 59900000, "8": 45600000,
    "9": 49000000, "10": 40200000, "11": 53700000, "12": 35800000,
    "13": 17900000, "14": 17600000, "15": 19000000, "16": 36600000,
    "17": 24000000, "18": 17200000, "19": 26500000, "20": 27500000,
    "21": 13200000, "22": 14700000, "X": 60600000,
}
# hg38 / GRCh38 chromosome lengths (UCSC hg38; verified against the CNAqc and
# rCGH GRCh38 tables) and centromere midpoints (UCSC cytoBand acen midpoints).
HG38_CHR_LEN = {
    "1": 248956422, "2": 242193529, "3": 198295559, "4": 190214555,
    "5": 181538259, "6": 170805979, "7": 159345973, "8": 145138636,
    "9": 138394717, "10": 133797422, "11": 135086622, "12": 133275309,
    "13": 114364328, "14": 107043718, "15": 101991189, "16": 90338345,
    "17": 83257441, "18": 80373285, "19": 58617616, "20": 64444167,
    "21": 46709983, "22": 50818468, "X": 156040895,
}
HG38_CENTROMERE = {
    "1": 123400000, "2": 93900000, "3": 90900000, "4": 50000000,
    "5": 48800000, "6": 59800000, "7": 60100000, "8": 45200000,
    "9": 43000000, "10": 39800000, "11": 53400000, "12": 35500000,
    "13": 17700000, "14": 17200000, "15": 19000000, "16": 36800000,
    "17": 25100000, "18": 18500000, "19": 26200000, "20": 28100000,
    "21": 12000000, "22": 15000000, "X": 61000000,
}


def genome_coords(build=None):
    """Return (chr_len, centromere) dicts for the requested build."""
    b = (build or GENOME_BUILD).lower()
    if b in ("hg19", "grch37", "b37"):
        return HG19_CHR_LEN, HG19_CENTROMERE
    if b in ("hg38", "grch38"):
        return HG38_CHR_LEN, HG38_CENTROMERE
    raise ValueError(f"Unknown genome build '{b}'. Use 'hg19' or 'hg38'.")


def check_build_consistency(seg, build=None):
    """Guard against a silent hub/build mismatch. For each declared build we
    score how well the observed maximum segment coordinate per chromosome fits
    that build's chromosome lengths, then compare the fit of the DECLARED build
    against the alternative build. If the alternative fits clearly better, the
    wrong build was declared and we raise with an actionable message.

    Scoring handles both failure directions:
      * coordinates OVERFLOWING the declared lengths (declared build too small,
        e.g. hg38 data declared as hg19 on chr17/18/20), and
      * coordinates falling systematically SHORT of the declared telomeres
        (declared build too large), which catches the reverse case that a pure
        overflow test would miss."""
    declared = (build or GENOME_BUILD).lower()
    alt = "hg38" if declared.startswith(("hg19", "grch37", "b37")) else "hg19"

    def fit_error(bld):
        """Mean relative distance between observed per-chrom max and that
        build's chromosome length (lower = better fit). Overflow is penalised
        harder than undershoot."""
        chr_len, _ = genome_coords(bld)
        errs = []
        for chrom, g in seg.groupby("chrom"):
            c = _norm_chrom(chrom)
            if c not in chr_len:
                continue
            mx = float(g["end"].max())
            L = float(chr_len[c])
            if mx > L:                       # overflow: impossible under this build
                errs.append(2.0 * (mx - L) / L)
            else:                            # undershoot: allowed but scored
                errs.append((L - mx) / L)
        return float(np.mean(errs)) if errs else np.nan, len(errs)

    e_declared, n = fit_error(declared)
    e_alt, _ = fit_error(alt)
    if n < 5 or not np.isfinite(e_declared) or not np.isfinite(e_alt):
        return
    # Hard overflow of the declared build is decisive. Threshold is 2 (not 3):
    # hg19 and hg38 chromosome lengths are close, so a real mismatch may
    # overflow only a couple of chromosomes (e.g. hg19-as-hg38 overflows only
    # chr9 and chr21). Requiring the alternative build to also fit better
    # guards against false positives from stray telomere-spanning segments.
    chr_len, _ = genome_coords(declared)
    hard_overflow = sum(
        1 for chrom, g in seg.groupby("chrom")
        if _norm_chrom(chrom) in chr_len
        and float(g["end"].max()) > chr_len[_norm_chrom(chrom)] * 1.02)
    if hard_overflow >= 2 and e_alt < e_declared:
        raise RuntimeError(
            f"Genome-build mismatch: {hard_overflow} chromosomes have segment "
            f"coordinates exceeding the declared build '{declared}', and the "
            f"data fit '{alt}' better (fit error {e_alt:.3f} vs {e_declared:.3f}). "
            f"Set CRC_M22_GENOME_BUILD={alt}. "
            f"Legacy Xena TCGA hub = hg19; GDC hub = hg38.")
    # Soft signal: alternative fits markedly better with no overflow either way.
    if e_alt + 0.03 < e_declared:
        print(f"  [warn] segment coordinates fit '{alt}' (err {e_alt:.3f}) better "
              f"than the declared '{declared}' (err {e_declared:.3f}). "
              f"Verify CRC_M22_GENOME_BUILD; legacy Xena=hg19, GDC=hg38.")

##############################################################################
# DOWNLOAD LAYER — robust, cached, offline-after-first-run
##############################################################################
def _local_path(dataset):
    """Cache filename derived from a Xena dataset path (slashes -> __)."""
    safe = dataset.replace("/", "__")
    return os.path.join(CACHE_DIR, safe)


def download_xena(hub, dataset, required=True):
    """Download {hub}/download/{dataset} (or its .gz), cache, return local path.
    If the file already exists in the cache, the network is not touched."""
    local = _local_path(dataset)
    if os.path.exists(local) and os.path.getsize(local) > 0:
        print(f"  [cache] {dataset}")
        return local

    last_err = None
    for suffix in ("", ".gz"):
        url = f"{hub}/{dataset}{suffix}"
        for attempt in range(1, DOWNLOAD_RETRIES + 1):
            try:
                print(f"  [get ] {url}  (try {attempt}/{DOWNLOAD_RETRIES})")
                req = urllib.request.Request(url, headers={"User-Agent": "crc-validation/1.0"})
                with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as r:
                    raw = r.read()
                if suffix == ".gz":
                    raw = gzip.decompress(raw)
                # sanity: must look like text with at least one line
                if len(raw) == 0:
                    raise ValueError("empty payload")
                with open(local, "wb") as fh:
                    fh.write(raw)
                print(f"  [ok  ] {dataset}  ({len(raw)/1e6:.1f} MB)")
                return local
            except (urllib.error.URLError, urllib.error.HTTPError,
                    TimeoutError, ValueError, OSError) as e:
                last_err = e
                time.sleep(1.5 * attempt)
        # try next suffix
    msg = (f"Failed to download '{dataset}' from {hub}\n"
           f"  last error: {last_err}\n"
           f"  If this host is unreachable from your environment, download the "
           f"file manually into {CACHE_DIR}/ as '{os.path.basename(local)}'.")
    if required:
        raise RuntimeError(msg)
    print("  [warn] " + msg)
    return None


def read_table(path, **kw):
    """Read a possibly-gzipped TSV/CSV into a DataFrame."""
    if path is None:
        return None
    opener = gzip.open if path.endswith(".gz") else open
    # sniff separator on first line
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")) as fh:
        first = fh.readline()
    sep = "\t" if first.count("\t") >= first.count(",") else ","
    return pd.read_csv(path, sep=sep, low_memory=False, **kw)


##############################################################################
# TCGA BARCODE NORMALISATION
##############################################################################
def tcga_patient(barcode):
    """TCGA-XX-XXXX  (12-char patient id)."""
    m = re.match(r"(TCGA-[0-9A-Za-z]{2}-[0-9A-Za-z]{4})", str(barcode))
    return m.group(1) if m else None


def tcga_sample_type(barcode):
    """Two-digit sample-type code from TCGA-XX-XXXX-YY... ; None if absent."""
    m = re.match(r"TCGA-[0-9A-Za-z]{2}-[0-9A-Za-z]{4}-(\d{2})", str(barcode))
    return m.group(1) if m else None


def is_primary_tumor(barcode):
    """Primary solid tumour = sample type 01. If no type suffix, keep it
    (Xena matrices are often already primary-tumour only)."""
    st = tcga_sample_type(barcode)
    return (st is None) or (st == "01")


##############################################################################
# DISCOVERY FEATURE SPACE
##############################################################################
def load_discovery():
    disc = pd.read_csv(DISCOVERY_MATRIX, index_col=0)
    disc = disc.astype(np.int8)
    cols = disc.columns.tolist()
    snv_genes = [c[4:] for c in cols if c.startswith("SNV_")]
    cnv_all   = [c[4:] for c in cols if c.startswith("CNV_")]
    sv_genes  = [c[3:] for c in cols if c.startswith("SV_")]
    # split CNV into arm-level vs gene-level using the arm pattern
    arm_re = re.compile(r"^(?:[0-9]{1,2}|X|Y)[pq]$")
    cnv_arms  = [c for c in cnv_all if arm_re.match(c)]
    cnv_genes = [c for c in cnv_all if not arm_re.match(c)]
    print(f"Discovery matrix: {disc.shape[0]} samples x {disc.shape[1]} features")
    print(f"  SNV genes : {len(snv_genes)}")
    print(f"  CNV genes : {len(cnv_genes)}   CNV arms: {len(cnv_arms)}")
    print(f"  SV  genes : {len(sv_genes)}")
    return disc, cols, snv_genes, cnv_genes, cnv_arms, sv_genes


def normalize_cluster_values(series):
    """0-based internal labels regardless of source 0/1 indexing (matches
    Modules 16-21)."""
    s = pd.to_numeric(series, errors="raise").astype(int)
    vals = sorted(s.dropna().unique().tolist())
    if vals and min(vals) == 1 and max(vals) <= 8 and 0 not in vals:
        s = s - 1
    return s


def display_cluster(c):
    return f"C{int(c) + 1}"


def load_discovery_labels():
    lab = pd.read_csv(CLUSTER_FILE)
    if not {"sample_id", "cluster"}.issubset(lab.columns):
        raise ValueError("Cluster CSV must contain sample_id and cluster columns.")
    lab = lab[["sample_id", "cluster"]].copy()
    lab["cluster"] = normalize_cluster_values(lab["cluster"])  # 0..3
    lab = lab.dropna(subset=["sample_id"]).drop_duplicates("sample_id")
    return lab.set_index("sample_id")["cluster"]


##############################################################################
# TCGA PARSERS  ->  per-sample feature evidence
##############################################################################
def parse_mc3_snv(path, snv_genes):
    """Return (hits, present_patients).
    hits            : patient_id -> set(genes with a functional mutation)
    present_patients: patient_id set for EVERY primary-tumour patient that
                      appears anywhere in the MC3 file (any gene, any
                      effect) -- i.e. patients MC3 actually sequenced/called,
                      regardless of whether they hit the 210-gene panel.
    This distinction matters: a patient absent from MC3 entirely has NO
    mutation data (missing), which is not the same as a patient present in
    MC3 with zero functional hits in the panel (a genuine negative call).
    Silently treating "absent from MC3" as "SNV-negative" biases every
    downstream analysis that uses the SNV feature block."""
    df = read_table(path)
    cols = {c.lower(): c for c in df.columns}
    scol = cols.get("sample") or cols.get("sampleid") or list(df.columns)[0]
    gcol = cols.get("gene") or cols.get("hugo_symbol") or cols.get("symbol")
    ecol = (cols.get("effect") or cols.get("variant_classification")
            or cols.get("consequence"))
    if gcol is None or ecol is None:
        raise ValueError(f"MC3 file missing gene/effect columns; have {list(df.columns)[:12]}")
    gene_set = set(snv_genes)
    hits = {}
    present = set()
    for samp, gene, eff in zip(df[scol], df[gcol], df[ecol]):
        if not is_primary_tumor(samp):
            continue
        pid = tcga_patient(samp)
        if pid is None:
            continue
        present.add(pid)                        # sequenced/called by MC3, regardless of gene/effect
        g = str(gene).strip().upper()
        if g not in gene_set:
            continue
        if str(eff).strip() not in FUNCTIONAL_MAF:
            continue
        hits.setdefault(pid, set()).add(g)
    return hits, present


def parse_gistic_gene_cnv(path, cnv_genes):
    """Return (hits, wanted_genes, present_patients).
    present_patients: every patient that is a COLUMN in the GISTIC file,
    i.e. genuinely has a GISTIC call, regardless of whether any of the 84
    panel genes happened to be altered for them."""
    df = read_table(path)
    idx_col = df.columns[0]                       # 'Gene Symbol' or 'sample'
    df = df.set_index(idx_col)
    df.index = df.index.astype(str).str.upper()
    wanted = [g for g in cnv_genes if g.upper() in df.index]
    hits = {}
    present = set()
    if not wanted:
        return hits, wanted, present
    sub = df.loc[[g.upper() for g in wanted]]     # genes x samples
    for samp in sub.columns:
        if not is_primary_tumor(samp):
            continue
        pid = tcga_patient(samp)
        if pid is None:
            continue
        present.add(pid)
        col = pd.to_numeric(sub[samp], errors="coerce").fillna(0)
        altered = set(col.index[col != 0])
        if altered:
            hits.setdefault(pid, set()).update(
                g for g in wanted if g.upper() in altered)
    return hits, wanted, present


def _norm_chrom(x):
    x = str(x).replace("chr", "").strip()
    return x


def parse_segment_arm_cnv(paths, cnv_arms):
    """Compute arm-level CNV binary from one or more Xena segment files.
    Rule (matches Module 2): an arm is 'altered' for a sample if the
    fraction of that arm's overlapping segments with |value| > threshold
    is >= ARM_ALTERED_FRAC.
    Return (hits, arm_names, present_patients); present_patients is every
    patient with >=1 segment record, i.e. genuinely has copy-number data,
    regardless of whether any arm crossed the alteration threshold."""
    frames = []
    for p in paths:
        if p is None:
            continue
        d = read_table(p)
        frames.append(d)
    if not frames:
        return {}, [], set()
    seg = pd.concat(frames, ignore_index=True)
    cols = {c.lower(): c for c in seg.columns}
    scol = cols.get("sample") or list(seg.columns)[0]
    ccol = cols.get("chrom") or cols.get("chr") or cols.get("chromosome")
    stcol = cols.get("start")
    encol = cols.get("end")
    vcol = cols.get("value") or cols.get("segment_mean") or cols.get("seg.mean")
    if None in (ccol, stcol, encol, vcol):
        raise ValueError(f"Segment file missing columns; have {list(seg.columns)[:10]}")

    seg = seg[[scol, ccol, stcol, encol, vcol]].copy()
    seg.columns = ["sample", "chrom", "start", "end", "value"]
    seg["chrom"] = seg["chrom"].map(_norm_chrom)
    seg["start"] = pd.to_numeric(seg["start"], errors="coerce")
    seg["end"] = pd.to_numeric(seg["end"], errors="coerce")
    seg["value"] = pd.to_numeric(seg["value"], errors="coerce")
    seg = seg.dropna(subset=["start", "end", "value"])
    seg = seg[seg["sample"].map(is_primary_tumor)]
    seg["patient"] = seg["sample"].map(tcga_patient)
    seg = seg.dropna(subset=["patient"])
    present = set(seg["patient"].unique())

    # Guard: verify the declared genome build actually matches these segments.
    check_build_consistency(seg)
    chr_len, centromere = genome_coords()
    print(f"  arm-CNV using genome build: {GENOME_BUILD}")

    # arm intervals
    arm_iv = {}
    for arm in cnv_arms:
        chrom = arm[:-1]
        side = arm[-1]
        if chrom not in chr_len:
            continue
        cen = centromere[chrom]
        if side == "p":
            arm_iv[arm] = (chrom, 0, cen)
        else:
            arm_iv[arm] = (chrom, cen, chr_len[chrom])

    hits = {}
    for pid, g in seg.groupby("patient"):
        altered_arms = set()
        for arm, (chrom, a0, a1) in arm_iv.items():
            sub = g[(g["chrom"] == chrom) & (g["end"] >= a0) & (g["start"] <= a1)]
            if len(sub) == 0:
                continue
            frac = float(np.mean(np.abs(sub["value"].values) > CNV_SEG_LOG2_THRESH))
            if frac >= ARM_ALTERED_FRAC:
                altered_arms.add(arm)
        if altered_arms:
            hits[pid] = altered_arms
    return hits, list(arm_iv.keys()), present


def fetch_cbioportal_sv(sv_genes):
    """Fetch and cache structural-variant/fusion records from cBioPortal.

    Missing API data are never converted to negative SV calls. A successful
    raw response is cached for deterministic offline reruns; if neither the
    cache nor the API is available, the analysis stops rather than silently
    changing the SV arms to all-zero matrices.
    """
    if not TRY_CBIOPORTAL_SV:
        print("  cBioPortal SV fetch disabled (CRC_M22_TRY_SV=0)")
        return {}, []
    profile = f"{CBIO_STUDY}_structural_variants"
    sample_list = f"{CBIO_STUDY}_all"
    cache_path = f"{CACHE_DIR}/cbioportal_structural_variants.json"
    # cBioPortal accepts a study-wide fetch via sampleListId query param.
    endpoints = [
        f"{CBIO_API}/structural-variant/fetch?sampleListId={sample_list}",
        f"{CBIO_API}/structural-variants/fetch?sampleListId={sample_list}",
    ]
    data = None
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 2:
        try:
            with open(cache_path, "r", encoding="utf-8") as handle:
                cached = json.load(handle)
            if isinstance(cached, list) and cached:
                data = cached
                print(f"  [cache] {os.path.basename(cache_path)}")
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] invalid cBioPortal cache ignored: {e}")
    if data is None:
        for url in endpoints:
            try:
                payload = json.dumps({"molecularProfileIds": [profile]}).encode()
                req = urllib.request.Request(
                    url, data=payload,
                    headers={"Content-Type": "application/json",
                             "Accept": "application/json",
                             "User-Agent": "crc-validation/1.0"},
                    method="POST")
                with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as response:
                    fetched = json.loads(response.read())
                if not isinstance(fetched, list) or not fetched:
                    continue
                data = fetched
                with open(cache_path, "w", encoding="utf-8") as handle:
                    json.dump(data, handle)
                break
            except Exception as e:                   # noqa: BLE001
                print(f"  [warn] cBioPortal SV attempt failed: {e}")
                continue
    if data is None:
        raise RuntimeError(
            "No cBioPortal structural-variant data were available from cache "
            "or API; refusing to encode unmeasured SV features as wild-type.")

    gene_set = {g.upper() for g in sv_genes}
    hits, covered = {}, set()
    for sv in data:
        samp = sv.get("sampleId") or sv.get("patientId")
        pid = tcga_patient(samp) if samp else None
        if pid is None:
            continue
        for key in ("site1HugoSymbol", "site2HugoSymbol"):
            g = str(sv.get(key, "")).strip().upper()
            if g in gene_set:
                hits.setdefault(pid, set()).add(g)
                covered.add(g)
    if not hits:
        raise RuntimeError(
            "The cBioPortal response contained no usable SV calls for the "
            "prespecified discovery genes; SV ablation is not estimable.")
    print(f"  cBioPortal SV: {len(hits)} patients, "
          f"{len(covered)}/{len(sv_genes)} SV genes covered")
    return hits, sorted(covered)


##############################################################################
# BUILD TCGA BINARY MATRIX IN THE DISCOVERY FEATURE SPACE
##############################################################################
def build_tcga_matrix(disc_cols, snv_hits, cnv_gene_hits, arm_hits, sv_hits,
                      cnv_genes_covered, arm_covered, sv_genes_covered):
    """Assemble a patients x 371 binary matrix with the SAME columns as the
    discovery matrix. Returns (matrix, coverage_dict)."""
    patients = set()
    for h in (snv_hits, cnv_gene_hits, arm_hits, sv_hits):
        patients.update(h.keys())
    patients = sorted(patients)

    mat = pd.DataFrame(0, index=patients, columns=disc_cols, dtype=np.int8)

    for pid, genes in snv_hits.items():
        for g in genes:
            col = f"SNV_{g}"
            if col in mat.columns:
                mat.at[pid, col] = 1
    for pid, genes in cnv_gene_hits.items():
        for g in genes:
            col = f"CNV_{g}"
            if col in mat.columns:
                mat.at[pid, col] = 1
    for pid, arms in arm_hits.items():
        for a in arms:
            col = f"CNV_{a}"
            if col in mat.columns:
                mat.at[pid, col] = 1
    for pid, genes in sv_hits.items():
        for g in genes:
            col = f"SV_{g}"
            if col in mat.columns:
                mat.at[pid, col] = 1

    # coverage = which discovery features are even measurable in TCGA
    snv_cov = [c for c in disc_cols if c.startswith("SNV_")]  # MC3 = all genes measurable
    coverage = {
        "SNV": (len(snv_cov), len(snv_cov)),
        "CNV_gene": (len(cnv_genes_covered),
                     len([c for c in disc_cols if c.startswith("CNV_")
                          and not re.match(r"^CNV_(?:[0-9]{1,2}|X|Y)[pq]$", c)])),
        "CNV_arm": (len(arm_covered),
                    len([c for c in disc_cols
                         if re.match(r"^CNV_(?:[0-9]{1,2}|X|Y)[pq]$", c)])),
        "SV": (len(sv_genes_covered),
               len([c for c in disc_cols if c.startswith("SV_")])),
    }
    return mat, coverage


##############################################################################
# CLINICAL / SURVIVAL
##############################################################################
STAGE_MAP = {
    "STAGE I": 1, "STAGE IA": 1, "STAGE IB": 1,
    "STAGE II": 2, "STAGE IIA": 2, "STAGE IIB": 2, "STAGE IIC": 2,
    "STAGE III": 3, "STAGE IIIA": 3, "STAGE IIIB": 3, "STAGE IIIC": 3,
    "STAGE IV": 4, "STAGE IVA": 4, "STAGE IVB": 4, "STAGE IVC": 4,
}


def parse_stage(x):
    """Roman-numeral stage -> 1..4 via named lookup (avoids the classic
    gsub('[^0-9]','') failure on Roman numerals)."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    if s in STAGE_MAP:
        return STAGE_MAP[s]
    for k, v in STAGE_MAP.items():
        if s.startswith(k):
            return v
    return np.nan


def load_clinical_survival(clinical_path, survival_paths):
    """Return DataFrame indexed by patient with columns:
    duration, event, age, sex, stage (missing covariates simply absent)."""
    # --- survival (OS, OS.time) from GDC survival files ---
    surv_frames = []
    for p in survival_paths:
        if p is None:
            continue
        d = read_table(p)
        cols = {c.lower(): c for c in d.columns}
        scol = cols.get("sample") or list(d.columns)[0]
        oscol = cols.get("os")
        otcol = cols.get("os.time") or cols.get("os_time")
        if oscol is None or otcol is None:
            continue
        d = d[[scol, oscol, otcol]].copy()
        d.columns = ["sample", "event", "duration"]
        surv_frames.append(d)
    if not surv_frames:
        raise RuntimeError("No usable survival file (need OS and OS.time columns).")
    surv = pd.concat(surv_frames, ignore_index=True)
    surv = surv[surv["sample"].map(is_primary_tumor)]
    surv["patient"] = surv["sample"].map(tcga_patient)
    surv = surv.dropna(subset=["patient"])
    surv["event"] = pd.to_numeric(surv["event"], errors="coerce")
    surv["duration"] = pd.to_numeric(surv["duration"], errors="coerce")
    surv = surv.dropna(subset=["event", "duration"])
    surv = surv[surv["duration"] >= 0]
    surv = surv.sort_values("duration").drop_duplicates("patient", keep="last")
    surv = surv.set_index("patient")[["duration", "event"]]

    # --- covariates from clinical matrix ---
    out = surv.copy()
    if clinical_path is not None:
        cl = read_table(clinical_path)
        idc = cl.columns[0]
        cl = cl.rename(columns={idc: "sample"})
        cl = cl[cl["sample"].map(is_primary_tumor)]
        cl["patient"] = cl["sample"].map(tcga_patient)
        cl = cl.dropna(subset=["patient"]).drop_duplicates("patient")
        cl = cl.set_index("patient")
        lc = {c.lower(): c for c in cl.columns}

        agecol = next((lc[k] for k in lc if "age_at_initial" in k or k == "age"), None)
        sexcol = next((lc[k] for k in lc if k in ("gender", "sex")), None)
        stgcol = next((lc[k] for k in lc
                       if "pathologic_stage" in k or "ajcc_pathologic_tumor_stage" in k
                       or k == "stage"), None)
        if agecol:
            out["age"] = pd.to_numeric(cl[agecol], errors="coerce").reindex(out.index)
        if sexcol:
            out["sex"] = (cl[sexcol].astype(str).str.upper().str.startswith("M")
                          ).astype(float).reindex(out.index)
        if stgcol:
            out["stage"] = cl[stgcol].map(parse_stage).reindex(out.index)
    return out


def build_dfs_survival(clinical_path):
    """Secondary endpoint, added 2026-07-16: disease-free survival (DFS) for
    TCGA COAD/READ, built from the legacy Xena clinical matrix's 'new tumor
    event after initial treatment' field -- the standard TCGA-derived
    recurrence/progression proxy used when a dedicated, curated DFI/PFI
    field is not available (cf. Liu et al. 2018, Cell, TCGA Pan-Cancer
    Clinical Data Resource, which constructs its own DFI from this same
    underlying field for cancer types lacking a curated value).

    NOTE ON NAMING: this is deliberately called DFS, not RFS, even though it
    plays the same "primary-endpoint-of-the-rest-of-the-paper" role that RFS
    plays for the discovery cohort. TCGA's 'new tumor event' field captures
    any new tumour event after initial treatment (locoregional recurrence,
    distant metastasis, or new primary), which is a broader construct than
    the discovery cohort's curated 'Recurrence' field. Calling it RFS would
    overstate the equivalence between the two endpoints; DFS is the accurate
    name for what this field actually measures.

    Event = 1 if a new tumor event was recorded; duration = days to that
    event for event-positive patients, or days to last follow-up for
    event-free patients (proper right-censoring, rather than truncating
    censored patients at their OS time, which would silently import OS
    information into a nominally separate endpoint). Patients with missing
    event status (no YES/NO value) are excluded rather than imputed as
    event-free, the same "absence is not evidence of absence" discipline
    already used throughout this module for genomic feature coverage."""
    if clinical_path is None:
        raise RuntimeError("DFS endpoint requires the clinical matrix (missing).")
    cl = read_table(clinical_path)
    idc = cl.columns[0]
    cl = cl.rename(columns={idc: "sample"})
    cl = cl[cl["sample"].map(is_primary_tumor)]
    cl["patient"] = cl["sample"].map(tcga_patient)
    cl = cl.dropna(subset=["patient"]).drop_duplicates("patient")
    cl = cl.set_index("patient")
    lc = {c.lower(): c for c in cl.columns}

    evcol = lc.get("new_tumor_event_after_initial_treatment")
    daycol = lc.get("days_to_new_tumor_event_after_initial_treatment")
    fucol = lc.get("days_to_last_followup")
    if evcol is None or fucol is None:
        raise RuntimeError(
            "DFS fields not found in clinical matrix (need "
            "new_tumor_event_after_initial_treatment and days_to_last_followup).")

    ev = cl[evcol].astype(str).str.strip().str.upper()
    out = pd.DataFrame(index=cl.index)
    out["event"] = np.where(ev.eq("YES"), 1.0, np.where(ev.eq("NO"), 0.0, np.nan))
    day_event = (pd.to_numeric(cl[daycol], errors="coerce")
                 if daycol else pd.Series(np.nan, index=cl.index))
    day_fu = pd.to_numeric(cl[fucol], errors="coerce")
    out["duration"] = np.where(out["event"] == 1, day_event, day_fu)
    out = out.dropna(subset=["event", "duration"])
    out = out[out["duration"] >= 0]
    out["event"] = out["event"].astype(int)

    agecol = next((lc[k] for k in lc if "age_at_initial" in k or k == "age"), None)
    sexcol = next((lc[k] for k in lc if k in ("gender", "sex")), None)
    stgcol = next((lc[k] for k in lc
                   if "pathologic_stage" in k or "ajcc_pathologic_tumor_stage" in k
                   or k == "stage"), None)
    if agecol:
        out["age"] = pd.to_numeric(cl[agecol], errors="coerce").reindex(out.index)
    if sexcol:
        out["sex"] = (cl[sexcol].astype(str).str.upper().str.startswith("M")
                      ).astype(float).reindex(out.index)
    if stgcol:
        out["stage"] = cl[stgcol].map(parse_stage).reindex(out.index)
    return out


##############################################################################
# ABLATION ENGINE
##############################################################################
FEATURE_SETS = {
    "SNV+CNV+SV": ("SNV_", "CNV_", "SV_"),
    "SNV+CNV":    ("SNV_", "CNV_"),
    "SNV+SV":     ("SNV_", "SV_"),
    "CNV+SV":     ("CNV_", "SV_"),
    "SNV":        ("SNV_",),
    "CNV":        ("CNV_",),
    "SV":         ("SV_",),
}


def subset_columns(all_cols, prefixes):
    return [c for c in all_cols if c.startswith(prefixes)]


def transfer_labels(disc_sub, disc_labels, tcga_sub):
    """Train RF on discovery(subset)->C1-C4, predict TCGA(subset).
    Returns:
      pred      : hard predicted label per TCGA patient (for KM visualisation)
      c4_score  : P(C4) probability per TCGA patient (stable continuous risk
                  score for the Cox model — avoids the small-n instability of
                  a Cox HR on a handful of hard C4 calls)
      cv_f1, cv_acc : discovery 5-fold CV macro-F1 / accuracy (how separable
                  the classes are with this feature set)."""
    X = disc_sub.values
    y = disc_labels.values
    clf = RandomForestClassifier(
        n_estimators=600, class_weight="balanced_subsample",
        random_state=RANDOM_STATE, n_jobs=-1)
    # within-discovery CV separability
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_pred = cross_val_predict(clf, X, y, cv=skf, n_jobs=-1)
    cv_f1 = f1_score(y, cv_pred, average="macro")
    cv_acc = accuracy_score(y, cv_pred)
    # fit on all discovery, predict TCGA (hard label + P(C4))
    clf.fit(X, y)
    pred = clf.predict(tcga_sub.values)
    classes = list(clf.classes_)
    c4_idx = classes.index(3) if 3 in classes else None
    proba = clf.predict_proba(tcga_sub.values)
    c4_score = proba[:, c4_idx] if c4_idx is not None else np.zeros(len(pred))
    return pred, c4_score, cv_f1, cv_acc


def pam_kmedoids(D, k, n_restarts=10, max_iter=100, random_state=RANDOM_STATE):
    """K-medoids (PAM) clustering on a precomputed distance matrix D.

    This replaces hierarchical/agglomerative clustering for the de-novo
    TCGA arm on purpose: the discovery pipeline's own methodology explicitly
    rejected consensus hierarchical clustering because it degenerates into
    one dominant cluster plus near-singleton outliers on this exact kind of
    sparse binary genomic matrix (near-zero entropy, ~0.02, one giant
    cluster). PAM/k-medoids on Jaccard distance is one of the pipeline's
    three validated primary discovery methods (alongside BMM and Spectral
    clustering) precisely because it does not have that failure mode.

    Uses k-means++-style distance-weighted seeding and multiple restarts,
    keeping the lowest-cost (total within-cluster distance) solution --
    mirroring the project's "5 restarts, keep best" convention used for BMM.
    """
    n = D.shape[0]
    rng = np.random.default_rng(random_state)
    best_labels, best_cost = None, np.inf

    for restart in range(n_restarts):
        # k-means++-style seeding: spreads initial medoids apart instead of
        # picking them uniformly at random, which matters a lot on sparse
        # binary data where most pairwise Jaccard distances are close to 1.
        medoids = [rng.integers(n)]
        for _ in range(k - 1):
            d2 = np.min(D[:, medoids], axis=1) ** 2
            total = d2.sum()
            probs = (d2 / total) if total > 0 else np.full(n, 1.0 / n)
            medoids.append(rng.choice(n, p=probs))
        medoids = np.array(sorted(set(medoids)))
        # top up if seeding collided onto fewer than k unique points
        while len(medoids) < k:
            candidate = rng.integers(n)
            if candidate not in medoids:
                medoids = np.append(medoids, candidate)

        labels = np.argmin(D[:, medoids], axis=1)
        for _ in range(max_iter):
            changed = False
            # update: best medoid per cluster = point minimizing total
            # in-cluster distance
            new_medoids = medoids.copy()
            for ci in range(k):
                members = np.where(labels == ci)[0]
                if len(members) == 0:
                    # empty cluster: reseed from the point currently
                    # farthest from its own medoid (classic PAM repair)
                    far_point = np.argmax(np.min(D[:, medoids], axis=1))
                    new_medoids[ci] = far_point
                    continue
                sub = D[np.ix_(members, members)]
                costs = sub.sum(axis=1)
                new_medoids[ci] = members[np.argmin(costs)]
            if not np.array_equal(new_medoids, medoids):
                changed = True
            medoids = new_medoids
            new_labels = np.argmin(D[:, medoids], axis=1)
            if np.array_equal(new_labels, labels) and not changed:
                labels = new_labels
                break
            labels = new_labels

        cost = sum(D[i, medoids[labels[i]]] for i in range(n))
        if cost < best_cost:
            best_cost, best_labels = cost, labels.copy()

    return best_labels


def denovo_cluster(tcga_sub, k=K_SUBTYPES, n_restarts=20, random_state=RANDOM_STATE):
    """De-novo clustering of TCGA on the subset via NMF (multiplicative
    update, Kullback-Leibler divergence) -- matching Brunet et al. 2004's
    original NMF-clustering algorithm, i.e. the SAME algorithm family used
    to derive the locked discovery C1-C4 labels (NMF-Brunet k=4). This is a
    deliberate methodological-consistency choice: de-novo clustering TCGA
    with a DIFFERENT algorithm than the one that defines the discovery
    subtypes would confound "does prognostic structure exist in TCGA" with
    "does this other algorithm happen to find structure here," which isn't
    the comparison this ablation is meant to make.

    Binary features are non-negative by construction, so NMF applies
    directly (no data transform needed). Multiple random restarts are run
    and the lowest-reconstruction-error solution is kept, matching the
    project's "N restarts, keep best" convention used elsewhere (e.g. BMM).
    Hard cluster assignment is argmax over each patient's row of W (the
    standard "metagene" assignment rule from Brunet et al. 2004).

    Note: hierarchical/Jaccard-average-linkage clustering was deliberately
    NOT used here. It was tried first and produced a degenerate ~90%-in-one-
    cluster split on real TCGA data -- the exact "one giant cluster, near-
    zero entropy" failure mode this project's own discovery-phase method
    comparison already documented and rejected hierarchical clustering for.
    PAM/k-medoids on Jaccard distance (pam_kmedoids(), above) was tried next
    and works well (balanced, non-degenerate splits) -- it remains available
    as a cross-check -- but NMF is used here to match the actual discovery
    algorithm rather than just any binary-data-appropriate method."""
    X = tcga_sub.values.astype(float)
    n = X.shape[0]
    if X.shape[1] == 0 or n < k:
        return np.zeros(n, dtype=int), {
            "restarts_ok": 0, "pairwise_ari_mean": np.nan,
            "pairwise_ari_min": np.nan, "reconstruction_error": np.nan}
    rng = np.random.default_rng(random_state)
    best_W, best_err = None, np.inf
    restart_labels = []
    for _ in range(n_restarts):
        seed = int(rng.integers(0, 2**31 - 1))
        try:
            model = NMF(n_components=k, init="random", solver="mu",
                        beta_loss="kullback-leibler", max_iter=400,
                        random_state=seed)
            W = model.fit_transform(X)
            err = model.reconstruction_err_
            raw_labels = np.argmax(W, axis=1)
            restart_labels.append(raw_labels)
        except Exception:
            continue
        if err < best_err:
            best_err, best_W = err, W
    if best_W is None:
        # Degenerate fallback (should not happen on real data with n>=k):
        # all restarts failed to converge to a finite error.
        return np.zeros(n, dtype=int), {
            "restarts_ok": 0, "pairwise_ari_mean": np.nan,
            "pairwise_ari_min": np.nan, "reconstruction_error": np.nan}
    labels = np.argmax(best_W, axis=1)
    # Relabel by descending cluster size for a stable, reproducible ordering
    # (argmax component index is otherwise arbitrary across restarts).
    order = pd.Series(labels).value_counts().index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    final_labels = np.array([remap[l] for l in labels])
    pairwise_ari = []
    for i in range(len(restart_labels)):
        for j in range(i + 1, len(restart_labels)):
            pairwise_ari.append(adjusted_rand_score(
                restart_labels[i], restart_labels[j]))
    stability = {
        "restarts_ok": len(restart_labels),
        "pairwise_ari_mean": (float(np.mean(pairwise_ari))
                              if pairwise_ari else np.nan),
        "pairwise_ari_min": (float(np.min(pairwise_ari))
                             if pairwise_ari else np.nan),
        "reconstruction_error": float(best_err),
    }
    return final_labels, stability


MIN_EVENTS_TOTAL = 10       # minimum events overall to attempt a Cox fit
MIN_INDICATOR_ARM_N = 10    # both arms must contain enough patients
MIN_EVENTS_HR_ARM = 5       # both arms must contain enough events
COX_FALLBACK_PENALTIES = (0.01, 0.1, 1.0)  # convergence fallback only
HR_CLIP = (1e-2, 1e2)       # clip absurd HRs before summarising (safety net)
MIN_CLUSTER_SIZE_FOR_PROFILE_MATCH = 10  # ignore small de-novo clusters
                                         # as centroid-matching candidates (a
                                         # tiny cluster's mean feature vector is
                                         # too noisy to trust a correlation on)


def _build_surv_df(assign, surv, risk_col_name="risk", high_risk_label=None,
                   continuous_score=None):
    """Merge an assignment / score with survival + covariates.
    - For de-novo indicator: pass high_risk_label -> adds 0/1 `risk` column.
    - For transfer continuous score: pass continuous_score (Series) -> adds a
      standardised `risk` column (HR is then per 1 SD of the score)."""
    df = surv.copy()
    idx = df.index.intersection(assign.index)
    df = df.loc[idx]
    df["group"] = assign.reindex(df.index)
    if continuous_score is not None:
        s = continuous_score.reindex(df.index).astype(float)
        sd = s.std(ddof=0)
        df[risk_col_name] = (s - s.mean()) / (sd if sd > 0 else 1.0)  # z-score
    else:
        df[risk_col_name] = (df["group"] == high_risk_label).astype(int)
    df = df.dropna(subset=["group", "duration", "event", risk_col_name])
    return df


def _prepare_cox_data(df, risk_col, covars):
    """Complete-case Cox data with stage represented categorically.

    Treating AJCC stage as a single numeric covariate assumes a constant
    log-hazard increment from I to II, II to III, and III to IV. That is not
    clinically required and is avoided here by using indicator variables,
    with Stage I as the reference whenever it is present.
    """
    use = ["duration", "event", risk_col] + [c for c in covars if c in df.columns]
    d = df[use].dropna().reset_index(drop=True)
    cov_cols = []
    for c in [x for x in covars if x in d.columns]:
        if c == "stage":
            stage = pd.Categorical(d[c], categories=[1.0, 2.0, 3.0, 4.0])
            stage_dummies = pd.get_dummies(
                stage, prefix="stage", drop_first=True, dtype=float)
            stage_dummies.index = d.index
            d = pd.concat([d.drop(columns=[c]), stage_dummies], axis=1)
            cov_cols.extend([x for x in stage_dummies.columns
                             if stage_dummies[x].nunique() > 1])
        elif d[c].nunique() > 1:
            cov_cols.append(c)
    keep = ["duration", "event", risk_col] + cov_cols
    return d[keep], cov_cols


def _indicator_is_estimable(d, risk_col):
    values = set(pd.unique(d[risk_col]))
    if not values.issubset({0, 1}):
        return True
    for arm in (0, 1):
        arm_df = d[d[risk_col] == arm]
        if len(arm_df) < MIN_INDICATOR_ARM_N:
            return False
        if int(arm_df["event"].sum()) < MIN_EVENTS_HR_ARM:
            return False
    return True


def _fit_cox_pair(d, risk_col, cov_cols, penalizer):
    full_cols = ["duration", "event", risk_col] + cov_cols
    cov_model_cols = ["duration", "event"] + cov_cols
    cph_full = CoxPHFitter(penalizer=penalizer)
    cph_full.fit(d[full_cols], duration_col="duration", event_col="event")
    if cov_cols:
        cph_cov = CoxPHFitter(penalizer=penalizer)
        cph_cov.fit(d[cov_model_cols], duration_col="duration", event_col="event")
    else:
        cph_cov = None
    return cph_full, cph_cov


def _fit_cox_nested(df, risk_col, covars, check_ph=True):
    """Nested Cox analysis with valid unpenalised primary inference.

    The primary fit is unpenalised. Ridge is attempted only when the primary
    model fails to converge; fallback estimates are labelled descriptive and
    receive no likelihood-ratio P value because a standard chi-square LRT is
    not valid for the difference between penalised partial likelihoods.
    """
    out = {
        "HR": np.nan, "Cindex_full": np.nan, "Cindex_covars_only": np.nan,
        "delta_Cindex": np.nan, "LR_stat": np.nan, "LR_p": np.nan,
        "risk_wald_p": np.nan, "fit_type": "not_estimable",
        "penalizer": np.nan, "n_model": 0, "events_model": 0,
        "ph_risk_p": np.nan, "ph_min_p": np.nan, "ph_violations": "",
    }
    d, cov_cols = _prepare_cox_data(df, risk_col, covars)
    out["n_model"] = int(len(d))
    out["events_model"] = int(d["event"].sum()) if len(d) else 0
    if (len(d) < 20 or out["events_model"] < MIN_EVENTS_TOTAL
            or d[risk_col].nunique() < 2 or not _indicator_is_estimable(d, risk_col)):
        return out

    cph_full = cph_cov = None
    used_penalty = None
    try:
        cph_full, cph_cov = _fit_cox_pair(d, risk_col, cov_cols, penalizer=0.0)
        out["fit_type"] = "unpenalised"
        used_penalty = 0.0
    except Exception:
        for penalty in COX_FALLBACK_PENALTIES:
            try:
                cph_full, cph_cov = _fit_cox_pair(
                    d, risk_col, cov_cols, penalizer=penalty)
                out["fit_type"] = "ridge_fallback"
                used_penalty = penalty
                break
            except Exception:
                continue
    if cph_full is None:
        return out

    out["penalizer"] = float(used_penalty)
    out["HR"] = float(np.clip(np.exp(cph_full.params_[risk_col]), *HR_CLIP))
    out["Cindex_full"] = float(cph_full.concordance_index_)
    if risk_col in cph_full.summary.index:
        out["risk_wald_p"] = float(cph_full.summary.loc[risk_col, "p"])

    if cph_cov is not None:
        out["Cindex_covars_only"] = float(cph_cov.concordance_index_)
        out["delta_Cindex"] = out["Cindex_full"] - out["Cindex_covars_only"]
        if out["fit_type"] == "unpenalised":
            lr_stat = 2.0 * (float(cph_full.log_likelihood_)
                             - float(cph_cov.log_likelihood_))
            if lr_stat >= 0:
                out["LR_stat"] = lr_stat
                out["LR_p"] = float(chi2.sf(lr_stat, df=1))
    else:
        out["Cindex_covars_only"] = 0.5
        out["delta_Cindex"] = out["Cindex_full"] - 0.5
        if out["fit_type"] == "unpenalised":
            lrt = cph_full.log_likelihood_ratio_test()
            out["LR_stat"] = float(lrt.test_statistic)
            out["LR_p"] = float(lrt.p_value)

    if check_ph and out["fit_type"] == "unpenalised":
        try:
            ph = proportional_hazard_test(
                cph_full, d[["duration", "event", risk_col] + cov_cols],
                time_transform="rank")
            pvals = ph.summary["p"].astype(float)
            out["ph_risk_p"] = float(pvals.get(risk_col, np.nan))
            out["ph_min_p"] = float(pvals.min()) if len(pvals) else np.nan
            out["ph_violations"] = ";".join(pvals.index[pvals < 0.05].astype(str))
        except Exception:
            pass
    return out


def stable_seed(*parts):
    """Deterministic per-(feature_set, method) seed offset. Python's builtin
    hash() on str/tuple is randomised per PROCESS (PYTHONHASHSEED) since
    Python 3.3, so `hash((set_name, method))` gives a DIFFERENT bootstrap
    seed on every re-run despite the script's data layer being fully cached
    and deterministic -- silently breaking exact numeric reproducibility of
    the reported HR/C-index bootstrap CIs across runs. zlib.crc32 is a fixed,
    unsalted checksum, so the same (set_name, method) always maps to the
    same seed offset regardless of process/interpreter."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return zlib.crc32(key) % 100000


def bootstrap_survival(df, risk_col, covars, n_boot=N_BOOTSTRAP, seed=None):
    """Bootstrap patients -> percentile CIs for the risk HR, the full-model
    C-index, and the nested delta-C-index (risk's added value beyond
    covariates). `seed` should be varied per (feature_set, method) call --
    reusing the same seed across calls makes every ablation arm draw the
    IDENTICAL resampling pattern, which understates how independent the
    arms' bootstrap CIs actually are."""
    base_fit = _fit_cox_nested(df, risk_col, covars, check_ph=False)
    if base_fit["fit_type"] == "not_estimable":
        return {
            "HR_boot_median": np.nan, "HR_CI": (np.nan, np.nan),
            "Cindex_boot_median": np.nan, "Cindex_CI": (np.nan, np.nan),
            "delta_Cindex_boot_median": np.nan,
            "delta_Cindex_CI": (np.nan, np.nan), "n_boot_ok": 0,
            "n_boot_delta_ok": 0, "boot_stable": False,
            "n_unpenalised": 0, "n_ridge_fallback": 0,
            "HR_boundary_fraction": np.nan, "HR_boundary_warning": False}
    rng = np.random.default_rng(RANDOM_STATE if seed is None else seed)
    hrs, cidxs, deltas = [], [], []
    fit_types = []
    idx = np.arange(len(df))
    for _ in range(n_boot):
        take = rng.choice(idx, size=len(idx), replace=True)
        # CRITICAL: .iloc[take] with replace=True produces DUPLICATE pandas
        # index labels (the same patient ID appears multiple times). lifelines'
        # CoxPHFitter.concordance_index_ silently mishandles a non-unique
        # index -- it emits "DataFrame Index is not unique, defaulting to
        # incrementing index instead." and then returns exactly 0.5 for every
        # such fit, while the fitted coefficients (and therefore the HR) stay
        # correct. This was confirmed by comparing cph.concordance_index_
        # against lifelines.utils.concordance_index computed manually on the
        # same fit: the former was pinned at 0.5, the latter matched the
        # expected 0.70-0.77 range. Resetting the index removes the ambiguity.
        d = df.iloc[take].reset_index(drop=True)
        r = _fit_cox_nested(d, risk_col, covars, check_ph=False)
        fit_types.append(r["fit_type"])
        if np.isfinite(r["HR"]):
            hrs.append(r["HR"])
        if np.isfinite(r["Cindex_full"]):
            cidxs.append(r["Cindex_full"])
        if np.isfinite(r["delta_Cindex"]):
            deltas.append(r["delta_Cindex"])

    # A CI is only reported when a clear majority of resamples were estimable;
    # otherwise the HR is flagged non-estimable (e.g. transferred/clustered
    # high-risk arm too small in TCGA) rather than given a misleading interval.
    min_ok = max(50, int(0.6 * n_boot))
    boundary_fraction = (float(np.mean((np.asarray(hrs) <= HR_CLIP[0])
                                       | (np.asarray(hrs) >= HR_CLIP[1])))
                         if hrs else 0.0)
    hr_stable = len(hrs) >= min_ok and boundary_fraction < 0.025
    cindex_stable = len(cidxs) >= min_ok
    delta_stable = len(deltas) >= min_ok

    def ci95(a, ok):
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if ok else (np.nan, np.nan)

    return {
        "HR_boot_median": float(np.median(hrs)) if hr_stable else np.nan,
        "HR_CI": ci95(hrs, hr_stable),
        "Cindex_boot_median": float(np.median(cidxs)) if cindex_stable else np.nan,
        "Cindex_CI": ci95(cidxs, cindex_stable),
        "delta_Cindex_boot_median": float(np.median(deltas)) if delta_stable else np.nan,
        "delta_Cindex_CI": ci95(deltas, delta_stable),
        "n_boot_ok": len(hrs),
        "n_boot_delta_ok": len(deltas),
        "boot_stable": hr_stable,
        "HR_boundary_fraction": boundary_fraction,
        "HR_boundary_warning": bool(boundary_fraction >= 0.025),
        "n_unpenalised": int(sum(x == "unpenalised" for x in fit_types)),
        "n_ridge_fallback": int(sum(x == "ridge_fallback" for x in fit_types)),
    }


def identify_high_risk_denovo_survival(df):
    """SUPERSEDED default -- retained only as a documented cross-check, the
    same way pam_kmedoids() is kept alongside the NMF de-novo clustering.

    The de-novo cluster with the shortest median survival among events.
    Do NOT use this as the primary selection rule: it picks the "high-risk"
    cluster by whichever one already looks worst in the very same TCGA
    outcome data used afterwards to test whether it IS worse -- a form of
    outcome-based double-dipping (selection and test on the same data).
    Empirically, on this project's real TCGA run, 2 of 4 feature-set
    ablations picked a cluster this way that FLIPPED to protective (HR<1)
    once age/sex/stage were adjusted for, i.e. the crude "worst survival"
    cluster was partly just an older/later-stage cluster, not a real
    effect. See identify_high_risk_denovo_by_profile() for the replacement
    used by default."""
    best, best_med = None, np.inf
    for g, sub in df.groupby("group"):
        kmf = KaplanMeierFitter().fit(sub["duration"], sub["event"])
        med = kmf.median_survival_time_
        med = med if np.isfinite(med) else sub["duration"].median()
        if med < best_med:
            best_med, best = med, g
    return best


def identify_high_risk_denovo_by_profile(tcga_sub, assign_d, disc_sub, disc_labels,
                                         min_size=MIN_CLUSTER_SIZE_FOR_PROFILE_MATCH):
    """Identify which de-novo TCGA cluster corresponds to the discovery C4
    subtype using FEATURE-PROFILE similarity, not TCGA survival outcomes.

    For each de-novo cluster, compute its mean binary feature vector
    (centroid) on the SAME feature subset used for this ablation arm, and
    Pearson-correlate it against the discovery C4 centroid on that same
    subset. The cluster with the highest correlation is called "high risk"
    (i.e. "the one whose genomic profile looks like C4"), independent of
    what TCGA survival happens to look like for it.

    This is the classic nearest-centroid-by-correlation approach used for
    intrinsic molecular subtyping (Sorlie et al. 2003, PNAS -- the same
    correlation-to-centroid logic behind PAM50-style subtype calling), and
    it is the direct de-novo-arm analogue of what the label-transfer arm
    already does with a trained classifier: both ask "does this patient's
    genomic profile look like discovery C4", never "does this patient
    already have bad TCGA survival".

    Clusters smaller than `min_size` are excluded as candidates (a
    near-singleton cluster's centroid is too noisy to trust a correlation
    on). Falls back to identify_high_risk_denovo_survival() -- with a
    printed warning -- only if every candidate centroid is degenerate
    (zero variance, so correlation is undefined), which should not happen
    on real data with more than a handful of features.

    Returns (chosen_cluster_label, {cluster_label: correlation}) so the
    caller can log the full similarity table for transparency.
    """
    c4_mask = (disc_labels == 3)
    c4_centroid = disc_sub.loc[c4_mask].mean(axis=0)

    sizes = assign_d.value_counts()
    corrs = {}
    for g in sorted(sizes.index):
        members = assign_d.index[assign_d == g]
        if len(members) < min_size:
            continue
        centroid = tcga_sub.loc[members].mean(axis=0)
        if c4_centroid.std(ddof=0) == 0 or centroid.std(ddof=0) == 0:
            r = np.nan
        else:
            r = float(np.corrcoef(c4_centroid.values, centroid.values)[0, 1])
        corrs[int(g)] = r

    valid = {g: r for g, r in corrs.items() if np.isfinite(r)}
    if not valid:
        print("  [warn] all de-novo cluster centroids are degenerate (zero "
              "variance or all below min_size) -- falling back to the "
              "survival-based high-risk selection for this arm.")
        return None, corrs
    best_g = max(valid, key=valid.get)
    return best_g, corrs


##############################################################################
# PLOTS
##############################################################################
def km_plot(df, title, outpath, mode="transfer", endpoint_label="OS"):
    """Kaplan-Meier by assigned group. mode='transfer' labels groups C1-C4
    with the locked palette; mode='denovo' labels them cluster 1..k with a
    neutral palette (de-novo cluster identity is arbitrary). endpoint_label
    controls the y-axis label only ("OS", unchanged default, or "DFS",
    added 2026-07-16 for the secondary disease-free-survival endpoint)."""
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    groups = sorted(df["group"].unique())
    lr = multivariate_logrank_test(df["duration"], df["group"], df["event"])
    for i, g in enumerate(groups):
        sub = df[df["group"] == g]
        kmf = KaplanMeierFitter()
        if mode == "transfer":
            lbl = display_cluster(g)
            color = CLUSTER_COLORS.get(lbl, "#333333")
        else:
            lbl = f"cluster {int(g) + 1}"
            color = DENOVO_COLORS[i % len(DENOVO_COLORS)]
        kmf.fit(sub["duration"], sub["event"], label=f"{lbl} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax, ci_show=False, color=color, linewidth=2.4)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Overall survival" if endpoint_label == "OS" else "Disease-free survival")
    ax.set_ylim(0, 1.02)
    p = lr.p_value
    ax.text(0.02, 0.05, f"log-rank p = {p:.3g}", transform=ax.transAxes,
            fontsize=15, fontweight="bold",
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax.legend(loc="upper right", frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        plt.savefig(f"{outpath}.{ext}", bbox_inches="tight")
    plt.close()
    return p


def summary_figure(results, outpath, endpoint_label="OS", method="transfer"):
    """Three panels for one assignment method only.

    Transfer HRs are per SD of a continuous C4 score, whereas de-novo HRs
    compare a profile-matched cluster with all other clusters. They are not
    commensurate effect scales and are therefore never overlaid.
    Delta-C-index is the honest headline: the full-model C-index is often
    dominated by stage/age (frequently ~0.70-0.73 from stage alone on this
    cohort), so a high C-index does not by itself mean the genomic risk
    score is contributing anything. Delta-C-index isolates that.
    endpoint_label appends a disambiguating suffix to panel titles for the
    DFS run (added 2026-07-16); the OS run's titles are unchanged."""
    sets = list(FEATURE_SETS.keys())
    method_label = "Label transfer" if method == "transfer" else "De novo NMF"
    title_suffix = f" ({endpoint_label}; {method_label})"
    fig, axes = plt.subplots(1, 3, figsize=(27, 6.8))
    panels = [
        ("HR", "High-risk hazard ratio\n(bootstrap 95% CI)", "HR_boot_median", "HR_CI", 1.0),
        ("Cindex", "Full-model concordance index\n(risk + age/sex/stage)", "Cindex_boot_median", "Cindex_CI", 0.5),
        ("delta", "Delta C-index vs covariates-only\n(added value of the risk score)", "delta_Cindex_boot_median", "delta_Cindex_CI", 0.0),
    ]
    for ax, (metric, mlabel, med_key, ci_key, hline) in zip(axes, panels):
        x = np.arange(len(sets))
        meds, los, his = [], [], []
        for s in sets:
            r = results[s][method]["boot"]
            meds.append(r[med_key]); los.append(r[ci_key][0]); his.append(r[ci_key][1])
        meds = np.array(meds, float); los = np.array(los, float); his = np.array(his, float)
        valid = np.isfinite(meds) & np.isfinite(los) & np.isfinite(his)
        if valid.any():
            yerr = np.vstack([meds[valid] - los[valid], his[valid] - meds[valid]])
            ax.errorbar(x[valid], meds[valid], yerr=yerr, fmt="o", capsize=5,
                        markersize=9, linewidth=2, color=("#D55E00"
                        if method == "transfer" else "#0072B2"))
        ax.set_xticks(x); ax.set_xticklabels(sets, rotation=20, ha="right")
        ax.set_title(mlabel + title_suffix, fontsize=16, fontweight="bold")
        ax.axhline(hline, color="grey", ls="--", lw=1.2)
        ax.set_ylabel({"HR": "HR", "Cindex": "C-index", "delta": "Delta C-index"}[metric])
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        plt.savefig(f"{outpath}.{ext}", bbox_inches="tight")
    plt.close()


def forest_plot(results, outpath, endpoint_label="OS", method="transfer"):
    """Method-specific HR forest plot; effect scales are not mixed."""
    sets = list(FEATURE_SETS.keys())
    rows = []
    for s in sets:
        r = results[s][method]
        boot = r["boot"]
        rows.append({
            "feature_set": s, "method": method,
            "HR": boot["HR_boot_median"],
            "HR_lo": boot["HR_CI"][0], "HR_hi": boot["HR_CI"][1]})
    df = pd.DataFrame(rows)
    ok = df[df["HR"].notna() & df["HR_lo"].notna() & df["HR_hi"].notna()].copy()
    if ok.empty:
        print("  [warn] forest_plot: no estimable bootstrap HRs to plot, skipping.")
        return
    ok["label"] = ok["feature_set"]
    order_map = {s: i for i, s in enumerate(sets)}
    ok["sort_key"] = ok["feature_set"].map(order_map)
    ok = ok.sort_values("sort_key", ascending=False)
    y = np.arange(len(ok))
    fig, ax = plt.subplots(figsize=(11, max(6.0, 0.5 * len(ok) + 2.5)))
    for yi, (_, r) in zip(y, ok.iterrows()):
        color = "#D55E00" if r["HR"] > 1 else "#009E73"
        ax.errorbar(r["HR"], yi,
                    xerr=[[max(r["HR"] - r["HR_lo"], 0)], [max(r["HR_hi"] - r["HR"], 0)]],
                    fmt="o", markersize=9, color=color, ecolor=color, elinewidth=2.2,
                    capsize=4, markeredgecolor="black")
        ax.text(r["HR_hi"] * 1.08, yi, f"HR={r['HR']:.2f} ({r['HR_lo']:.2f}-{r['HR_hi']:.2f})",
                va="center", fontsize=11)
    ax.axvline(1.0, color="black", ls="--", lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(ok["label"])
    ax.set_xscale("log")
    xlabel = ("HR per SD of transferred C4 probability"
              if method == "transfer" else "Profile-matched cluster vs rest HR")
    method_label = "label transfer" if method == "transfer" else "de novo NMF"
    title = f"TCGA modality ablation: {method_label} ({endpoint_label})"
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        plt.savefig(f"{outpath}.{ext}", bbox_inches="tight")
    plt.close()


def run_ablation(tcga_mat, disc, disc_cols, disc_labels, surv, covars,
                 endpoint_label, suffix, available_feature_cols):
    """Runs the full seven-arm modality ablation (label transfer + de novo,
    each with a KM plot, nested Cox model, and bootstrap HR/C-index CIs) for
    one endpoint. Every function called here (_build_surv_df, _fit_cox_nested,
    bootstrap_survival, km_plot, identify_high_risk_*) is already endpoint-
    agnostic -- it only ever sees generic "duration"/"event" columns -- so
    calling this twice with different `surv` DataFrames (OS vs DFS) runs
    byte-identical statistical code on different input data, exactly as
    Module 21's run_pipeline() does for RFS vs OS. Added 2026-07-16 to
    support the DFS secondary endpoint without duplicating this logic.
    Returns (results, summary_rows, patient_assignments)."""
    results = {}
    summary_rows = []
    assignment_frames = []
    for set_name, prefixes in FEATURE_SETS.items():
        print(f"\n### {set_name} ({endpoint_label})")
        requested_cols = subset_columns(disc_cols, prefixes)
        cols = [c for c in requested_cols if c in available_feature_cols]
        print(f"  coverage-matched features: {len(cols)}/{len(requested_cols)}")
        if not cols:
            raise RuntimeError(f"No measured TCGA features for {set_name}")
        disc_sub = disc[cols]
        tcga_sub = tcga_mat[cols]
        results[set_name] = {}

        # ---- (A) label transfer ----
        # HR/C-index use the continuous P(C4) risk score (stable); the KM plot
        # visualises the hard predicted C1-C4 labels.
        pred, c4_score, cv_f1, cv_acc = transfer_labels(disc_sub, disc_labels, tcga_sub)
        assign_t = pd.Series(pred, index=tcga_sub.index)
        score_t = pd.Series(c4_score, index=tcga_sub.index)
        df_t = _build_surv_df(assign_t, surv, risk_col_name="risk",
                              continuous_score=score_t)
        title_t = f"TCGA label transfer — {set_name}"
        if endpoint_label != "OS":
            title_t += f" ({endpoint_label})"
        p_t = km_plot(df_t, title_t,
                      f"{FIGDIR}/KM_transfer_{set_name.replace('+', '_')}{suffix}",
                      mode="transfer", endpoint_label=endpoint_label)
        nest_t = _fit_cox_nested(df_t, "risk", covars)
        # seed varies per (feature_set, method, endpoint) so bootstrap arms
        # are drawn independently rather than reusing the same resampling
        # pattern across arms or across the OS/DFS runs.
        seed_t = RANDOM_STATE + stable_seed(set_name, "transfer", endpoint_label)
        boot_t = bootstrap_survival(df_t, "risk", covars, seed=seed_t)
        n_c4 = int((assign_t == 3).sum())
        results[set_name]["transfer"] = {
            "assign": assign_t, "score": score_t, "df": df_t,
            "logrank_p": p_t, "nest": nest_t, "boot": boot_t,
            "cv_f1": cv_f1, "cv_acc": cv_acc, "n_hard_C4": n_c4,
            "n_features_used": len(cols),
            "n_features_requested": len(requested_cols)}
        print(f"  transfer: disc 5-fold macroF1={cv_f1:.3f} acc={cv_acc:.3f} | "
              f"TCGA hard-C4 n={n_c4} logrank_p={p_t:.3g} | "
              f"C4-score HR/SD={nest_t['HR']:.2f} "
              f"(boot {boot_t['HR_CI'][0]:.2f}-{boot_t['HR_CI'][1]:.2f}) "
              f"Cidx_full={nest_t['Cindex_full']:.3f} "
              f"Cidx_covars_only={nest_t['Cindex_covars_only']:.3f} "
              f"delta={nest_t['delta_Cindex']:+.3f} LR_p={nest_t['LR_p']:.3g} "
              f"fit={nest_t['fit_type']} PH-risk-p={nest_t['ph_risk_p']:.3g} "
              f"[{boot_t['n_boot_ok']}/{N_BOOTSTRAP} boot ok]")

        # ---- (B) de novo ----
        dn, stability = denovo_cluster(tcga_sub)
        assign_d = pd.Series(dn, index=tcga_sub.index)
        sizes_d = pd.Series(dn).value_counts().sort_index().to_dict()
        max_fraction = max(sizes_d.values()) / len(assign_d)
        min_size = min(sizes_d.values())
        degenerate = bool(max_fraction > 0.90 or min_size < MIN_CLUSTER_SIZE_FOR_PROFILE_MATCH)
        print(f"  denovo NMF cluster sizes: {sizes_d}; "
              f"pairwise restart ARI mean={stability['pairwise_ari_mean']:.3f}; "
              f"degenerate={degenerate}")
        # High-risk cluster is identified by FEATURE-PROFILE similarity to the
        # discovery C4 centroid, not by which cluster already looks worst in
        # TCGA survival (see identify_high_risk_denovo_by_profile docstring
        # for why the outcome-based version was replaced).
        hr_lbl, profile_corr = identify_high_risk_denovo_by_profile(
            tcga_sub, assign_d, disc_sub, disc_labels)
        print(f"  denovo cluster-vs-discovery-C4 centroid correlation: {profile_corr}")
        if hr_lbl is None:
            print("  [warn] no eligible profile-matched cluster; Cox contrast "
                  "will be non-estimable (no outcome-based fallback used).")
        df_d = _build_surv_df(
            assign_d, surv, high_risk_label=(hr_lbl if hr_lbl is not None else -1))
        title_d = f"TCGA de novo NMF k=4 — {set_name}"
        if endpoint_label != "OS":
            title_d += f" ({endpoint_label})"
        p_d = km_plot(df_d, title_d,
                      f"{FIGDIR}/KM_denovo_{set_name.replace('+', '_')}{suffix}",
                      mode="denovo", endpoint_label=endpoint_label)
        nest_d = _fit_cox_nested(df_d, "risk", covars)
        seed_d = RANDOM_STATE + stable_seed(set_name, "denovo", endpoint_label)
        boot_d = bootstrap_survival(df_d, "risk", covars, seed=seed_d)
        results[set_name]["denovo"] = {
            "assign": assign_d, "df": df_d, "logrank_p": p_d,
            "nest": nest_d, "boot": boot_d,
            "high_risk_cluster": (int(hr_lbl) if hr_lbl is not None else None),
            "cluster_sizes": sizes_d, "profile_corr": profile_corr,
            "stability": stability, "degenerate_partition": degenerate,
            "n_features_used": len(cols),
            "n_features_requested": len(requested_cols)}
        matched_r = (profile_corr.get(hr_lbl, float("nan"))
                     if hr_lbl is not None else float("nan"))
        matched_display = (hr_lbl + 1) if hr_lbl is not None else "NA"
        print(f"  denovo  : high-risk cluster={matched_display} (profile-matched to C4, "
              f"r={matched_r:.3f}) logrank_p={p_d:.3g} | "
              f"HR={nest_d['HR']:.2f} (boot {boot_d['HR_CI'][0]:.2f}-{boot_d['HR_CI'][1]:.2f}) "
              f"Cidx_full={nest_d['Cindex_full']:.3f} delta={nest_d['delta_Cindex']:+.3f} "
              f"LR_p={nest_d['LR_p']:.3g} fit={nest_d['fit_type']} "
              f"PH-risk-p={nest_d['ph_risk_p']:.3g} "
              f"[{boot_d['n_boot_ok']}/{N_BOOTSTRAP} boot ok]")

        assignment_frames.append(pd.DataFrame({
            "patient_id": tcga_sub.index,
            "endpoint": endpoint_label,
            "feature_set": set_name,
            "transfer_subtype": [display_cluster(x) for x in assign_t.values],
            "transfer_C4_probability": score_t.values,
            "denovo_cluster": assign_d.values + 1,
            "denovo_profile_matched_C4": (
                assign_d.values == hr_lbl if hr_lbl is not None
                else np.full(len(assign_d), False)),
        }))

        for method in ("transfer", "denovo"):
            r = results[set_name][method]
            nest = r["nest"]
            summary_rows.append({
                "feature_set": set_name,
                "method": method,
                "endpoint": endpoint_label,
                "analysis_role": ("primary" if (set_name == "SNV+CNV+SV"
                                                  and method == "transfer")
                                  else "exploratory"),
                "n_features_used": r["n_features_used"],
                "n_features_requested": r["n_features_requested"],
                "logrank_p": r["logrank_p"],
                "highrisk_HR": nest["HR"],
                "HR_boot_median": r["boot"]["HR_boot_median"],
                "HR_CI_low": r["boot"]["HR_CI"][0],
                "HR_CI_high": r["boot"]["HR_CI"][1],
                "Cindex_full": nest["Cindex_full"],
                "Cindex_boot_median": r["boot"]["Cindex_boot_median"],
                "Cindex_CI_low": r["boot"]["Cindex_CI"][0],
                "Cindex_CI_high": r["boot"]["Cindex_CI"][1],
                "Cindex_covars_only": nest["Cindex_covars_only"],
                "delta_Cindex": nest["delta_Cindex"],
                "delta_Cindex_boot_median": r["boot"]["delta_Cindex_boot_median"],
                "delta_Cindex_CI_low": r["boot"]["delta_Cindex_CI"][0],
                "delta_Cindex_CI_high": r["boot"]["delta_Cindex_CI"][1],
                "LR_stat": nest["LR_stat"],
                "LR_p": nest["LR_p"],
                "risk_wald_p": nest["risk_wald_p"],
                "cox_fit_type": nest["fit_type"],
                "cox_penalizer": nest["penalizer"],
                "cox_n": nest["n_model"],
                "cox_events": nest["events_model"],
                "PH_risk_p": nest["ph_risk_p"],
                "PH_min_p": nest["ph_min_p"],
                "PH_violations": nest["ph_violations"],
                "n_boot_ok": r["boot"]["n_boot_ok"],
                "n_boot_unpenalised": r["boot"]["n_unpenalised"],
                "n_boot_ridge_fallback": r["boot"]["n_ridge_fallback"],
                "boot_stable": r["boot"]["boot_stable"],
                "HR_boundary_fraction": r["boot"].get("HR_boundary_fraction", np.nan),
                "HR_boundary_warning": r["boot"].get("HR_boundary_warning", False),
                "discovery_cv_macroF1": r.get("cv_f1", np.nan),
                "discovery_cv_accuracy": r.get("cv_acc", np.nan),
                "n_hard_C4": r.get("n_hard_C4", np.nan),
                "denovo_cluster_sizes": str(r.get("cluster_sizes", "")),
                "denovo_degenerate_partition": r.get("degenerate_partition", np.nan),
                "denovo_restart_ARI_mean": r.get("stability", {}).get(
                    "pairwise_ari_mean", np.nan),
                "denovo_restart_ARI_min": r.get("stability", {}).get(
                    "pairwise_ari_min", np.nan),
                "denovo_c4_profile_corr": (
                    r["profile_corr"].get(r.get("high_risk_cluster"), np.nan)
                    if method == "denovo" else np.nan),
                "highrisk_metric": ("C4_score_per_SD" if method == "transfer"
                                    else "highrisk_cluster_vs_rest_by_profile_match"),
            })
    return results, summary_rows, pd.concat(assignment_frames, ignore_index=True)


def add_fdr_column(df, pcol, qcol, group_col="endpoint"):
    """Benjamini-Hochberg FDR correction (Benjamini and Hochberg, 1995),
    added 2026-07-16 as a permanent output column per Avik's explicit
    request, following up on the manual one-off verification used to write
    the DFS de novo finding into the manuscript. Correction is applied
    WITHIN each group_col value separately (each endpoint's 7-feature-set-
    by-2-method arms are its own family of 14 simultaneously tested
    hypotheses), matching this project's stated convention of correcting
    within each defined family of tests rather than across unrelated
    analyses (see Statistical methods, Methods_final.md). Splitting by
    endpoint here, rather than further by method (transfer vs denovo), is
    deliberate: it reproduces exactly the family structure already used to
    derive and report the DFS SNV+CNV de novo q-value (0.220) in the
    manuscript text, so the script's own output and the manuscript prose
    cannot silently drift apart."""
    df[qcol] = np.nan
    for grp, sub in df.groupby(group_col):
        mask = sub[pcol].notna()
        if mask.sum() == 0:
            continue
        _, qvals, _, _ = multipletests(sub.loc[mask, pcol], alpha=0.05, method="fdr_bh")
        df.loc[sub.index[mask], qcol] = qvals
    return df


##############################################################################
# MAIN
##############################################################################
def main():
    print("=" * 74)
    print("MODULE 22 — TCGA CROSS-PLATFORM PORTABILITY WITH MODALITY ABLATION")
    print("=" * 74)

    # -- discovery feature space + labels --
    disc, disc_cols, snv_genes, cnv_genes, cnv_arms, sv_genes = load_discovery()
    disc_labels = load_discovery_labels()
    common = disc.index.intersection(disc_labels.index)
    disc = disc.loc[common]
    disc_labels = disc_labels.loc[common]
    print(f"Discovery samples with labels: {len(common)}")
    sizes = disc_labels.map(display_cluster).value_counts().sort_index().to_dict()
    print(f"Discovery cluster sizes: {sizes}")

    # -- download TCGA sources --
    print("\n" + "-" * 74 + "\nDOWNLOADING TCGA DATA\n" + "-" * 74)
    mc3 = download_xena(XENA_TCGA, "mc3/COADREAD_mc3.txt", required=True)
    gistic = download_xena(
        XENA_TCGA,
        "TCGA.COADREAD.sampleMap/Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
        required=True)
    seg_coad = download_xena(XENA_TCGA, "TCGA.COAD.sampleMap/SNP6_nocnv_genomicSegment", required=False)
    seg_read = download_xena(XENA_TCGA, "TCGA.READ.sampleMap/SNP6_nocnv_genomicSegment", required=False)
    clinical = download_xena(XENA_TCGA, "TCGA.COADREAD.sampleMap/COADREAD_clinicalMatrix", required=False)
    surv_coad = download_xena(XENA_GDC, "TCGA-COAD.survival.tsv", required=False)
    surv_read = download_xena(XENA_GDC, "TCGA-READ.survival.tsv", required=False)
    if surv_coad is None and surv_read is None:
        # fall back to legacy survival packaged in the clinical matrix path
        surv_legacy = download_xena(XENA_TCGA, "survival/COADREAD_survival.txt", required=True)
        survival_paths = [surv_legacy]
    else:
        survival_paths = [surv_coad, surv_read]

    # -- parse TCGA --
    print("\n" + "-" * 74 + "\nPARSING & BINARISING TCGA\n" + "-" * 74)
    snv_hits, snv_present = parse_mc3_snv(mc3, snv_genes)
    print(f"  SNV: {len(snv_present)} patients genuinely present in MC3; "
          f"{len(snv_hits)} with >=1 functional driver mutation")
    cnv_gene_hits, cnv_genes_cov, cnv_gene_present = parse_gistic_gene_cnv(gistic, cnv_genes)
    print(f"  CNV(gene): {len(cnv_gene_present)} patients present in GISTIC; "
          f"{len(cnv_gene_hits)} with >=1 altered gene; "
          f"{len(cnv_genes_cov)}/{len(cnv_genes)} genes covered")
    arm_hits, arm_cov, cnv_arm_present = parse_segment_arm_cnv([seg_coad, seg_read], cnv_arms)
    print(f"  CNV(arm): {len(cnv_arm_present)} patients present in segment files; "
          f"{len(arm_hits)} with >=1 altered arm; "
          f"{len(arm_cov)}/{len(cnv_arms)} arms computable")
    sv_hits, sv_cov = fetch_cbioportal_sv(sv_genes)

    tcga_mat, coverage = build_tcga_matrix(
        disc_cols, snv_hits, cnv_gene_hits, arm_hits, sv_hits,
        cnv_genes_cov, arm_cov, sv_cov)
    print(f"\nTCGA binary matrix (union, any-modality hit): "
          f"{tcga_mat.shape[0]} patients x {tcga_mat.shape[1]} features")
    cov_rows = []
    for k, (got, tot) in coverage.items():
        print(f"  feature coverage {k:9s}: {got}/{tot}")
        cov_rows.append({"modality": k, "covered": got, "total": tot,
                         "fraction": round(got / tot, 3) if tot else 0})
    pd.DataFrame(cov_rows).to_csv(f"{TABDIR}/tcga_feature_coverage.csv", index=False)

    # Only features genuinely measurable in TCGA are used on BOTH sides of
    # label transfer. In particular, the 24 unavailable discovery SV genes
    # are excluded rather than encoded as wild-type in every TCGA patient.
    available_feature_cols = set()
    available_feature_cols.update(f"SNV_{g}" for g in snv_genes)
    available_feature_cols.update(f"CNV_{g}" for g in cnv_genes_cov)
    available_feature_cols.update(f"CNV_{a}" for a in arm_cov)
    available_feature_cols.update(f"SV_{g}" for g in sv_cov)
    available_feature_cols &= set(disc_cols)
    print(f"  coverage-matched feature universe: "
          f"{len(available_feature_cols)}/{len(disc_cols)} features")

    # -- survival --
    surv = load_clinical_survival(clinical, survival_paths)
    covars = [c for c in ("age", "sex", "stage") if c in surv.columns]
    print(f"\nCovariates available: {covars if covars else '(none)'}")

    # -- common analysis cohort ---------------------------------------------
    # CRITICAL: a patient absent from MC3 has NO SNV data, not a confirmed
    # SNV-negative call -- silently coding "absent" as "0" biases every
    # ablation arm that includes the SNV block, and (empirically, on the
    # real Xena download) affects 230/610 = 38% of patients, who are then
    # spuriously tied to every OTHER all-zero-SNV patient under a Jaccard/
    # binary similarity metric. To make this an ablation of FEATURES (not a
    # shifting, differently-biased patient set per arm), every feature set
    # is evaluated on the SAME fixed cohort: patients with genuine data in
    # ALL THREE base sources (MC3, GISTIC, segment) AND survival follow-up.
    # SV is excluded from this requirement since coverage is near-zero by
    # construction (see tcga_feature_coverage.csv) and is reported, not
    # used to filter patients.
    patient_availability = pd.DataFrame({
        "in_MC3": pd.Series(True, index=sorted(snv_present)),
        "in_GISTIC": pd.Series(True, index=sorted(cnv_gene_present)),
        "in_segment": pd.Series(True, index=sorted(cnv_arm_present)),
    }).reindex(tcga_mat.index).fillna(False)
    patient_availability["has_survival"] = patient_availability.index.isin(surv.index)
    # DFS availability flag, added 2026-07-16 (see build_dfs_survival above).
    surv_dfs = build_dfs_survival(clinical)
    patient_availability["has_dfs"] = patient_availability.index.isin(surv_dfs.index)
    patient_availability.to_csv(f"{TABDIR}/tcga_patient_data_availability.csv")

    common_cohort = sorted(
        snv_present & cnv_gene_present & cnv_arm_present & set(surv.index)
        & set(tcga_mat.index))
    print(f"\nCommon analysis cohort, OS (genuine SNV+CNVgene+CNVarm+survival "
          f"data in ALL sources): {len(common_cohort)} patients")
    print("  This fixed cohort is used for EVERY ablation arm below, so "
          "differences between feature sets reflect features ablated, not a "
          "shifting patient population.")
    if len(common_cohort) < 50:
        print("  [warn] Common cohort is small; ablation comparisons will be "
              "correspondingly underpowered. Check tcga_feature_coverage.csv "
              "and tcga_patient_data_availability.csv for the bottleneck.")

    tcga_mat_os = tcga_mat.loc[common_cohort]
    surv_os = surv.loc[common_cohort]

    # -- ablation, OS (primary endpoint here; unchanged from before) --
    print("\n" + "=" * 74 + "\nABLATION ACROSS FEATURE SETS (OS)\n" + "=" * 74)
    results_os, rows_os, assignments_os = run_ablation(
        tcga_mat_os, disc, disc_cols, disc_labels, surv_os, covars, "OS", "",
        available_feature_cols)

    # -- ablation, DFS (secondary endpoint, added 2026-07-16) ----------------
    # Same cohort-construction discipline as OS above (genuine, non-imputed
    # data in every source -- now including genuine DFS follow-up rather than
    # imputing missing recurrence status as event-free), same seven feature
    # sets, same two assignment methods, same nested-Cox/bootstrap machinery
    # via run_ablation(): only the survival data fed in differs. DFS uses its
    # OWN common cohort (build_dfs_survival() above resolves duration+event
    # for a different, and smaller, set of patients than OS does -- about
    # 20% of patients lack a recorded new-tumor-event status), so this is a
    # genuinely independent secondary check rather than a re-analysis of the
    # identical 365-patient OS cohort under a relabelled endpoint.
    covars_dfs = [c for c in ("age", "sex", "stage") if c in surv_dfs.columns]
    common_cohort_dfs = sorted(
        snv_present & cnv_gene_present & cnv_arm_present & set(surv_dfs.index)
        & set(tcga_mat.index))
    print(f"\nCommon analysis cohort, DFS: {len(common_cohort_dfs)} patients, "
          f"{int(surv_dfs.loc[common_cohort_dfs, 'event'].sum())} events")
    if len(common_cohort_dfs) < 50:
        print("  [warn] DFS common cohort is small; ablation comparisons will "
              "be correspondingly underpowered.")

    tcga_mat_dfs = tcga_mat.loc[common_cohort_dfs]
    surv_dfs = surv_dfs.loc[common_cohort_dfs]

    print("\n" + "=" * 74 + "\nABLATION ACROSS FEATURE SETS (DFS)\n" + "=" * 74)
    results_dfs, rows_dfs, assignments_dfs = run_ablation(
        tcga_mat_dfs, disc, disc_cols, disc_labels, surv_dfs, covars_dfs,
        "DFS", "_DFS", available_feature_cols)

    summary = pd.DataFrame(rows_os + rows_dfs)
    # BH-FDR correction, added 2026-07-16 as a permanent column (see
    # add_fdr_column docstring): corrected within each endpoint's 14 arms.
    summary = add_fdr_column(summary, "logrank_p", "logrank_p_fdr_bh", group_col="endpoint")
    summary = add_fdr_column(summary, "LR_p", "LR_p_fdr_bh", group_col="endpoint")
    summary = add_fdr_column(summary, "risk_wald_p", "risk_wald_p_fdr_bh",
                             group_col="endpoint")
    summary.to_csv(f"{TABDIR}/ablation_survival_summary.csv", index=False)
    assignments_os.to_csv(f"{TABDIR}/tcga_assignments_OS.csv", index=False)
    assignments_dfs.to_csv(f"{TABDIR}/tcga_assignments_DFS.csv", index=False)
    cohort_summary = pd.DataFrame([
        {"endpoint": "OS", "n_genomic_survival": len(common_cohort),
         "events": int(surv_os["event"].sum()),
         "n_complete_age_sex_stage": int(
             surv_os[["age", "sex", "stage"]].dropna().shape[0])},
        {"endpoint": "DFS", "n_genomic_survival": len(common_cohort_dfs),
         "events": int(surv_dfs["event"].sum()),
         "n_complete_age_sex_stage": int(
             surv_dfs[["age", "sex", "stage"]].dropna().shape[0])},
    ])
    cohort_summary.to_csv(f"{TABDIR}/tcga_analysis_cohorts.csv", index=False)

    # Preserve the established filenames for transfer-only figures, and add
    # explicitly named de-novo counterparts. This prevents incomparable HR
    # scales from sharing one axis.
    summary_figure(results_os, f"{FIGDIR}/ablation_summary",
                   endpoint_label="OS", method="transfer")
    summary_figure(results_os, f"{FIGDIR}/ablation_summary_denovo",
                   endpoint_label="OS", method="denovo")
    forest_plot(results_os, f"{FIGDIR}/ablation_forest",
                endpoint_label="OS", method="transfer")
    forest_plot(results_os, f"{FIGDIR}/ablation_forest_denovo",
                endpoint_label="OS", method="denovo")
    summary_figure(results_dfs, f"{FIGDIR}/ablation_summary_DFS",
                   endpoint_label="DFS", method="transfer")
    summary_figure(results_dfs, f"{FIGDIR}/ablation_summary_denovo_DFS",
                   endpoint_label="DFS", method="denovo")
    forest_plot(results_dfs, f"{FIGDIR}/ablation_forest_DFS",
                endpoint_label="DFS", method="transfer")
    forest_plot(results_dfs, f"{FIGDIR}/ablation_forest_denovo_DFS",
                endpoint_label="DFS", method="denovo")

    print("\n" + "=" * 74)
    print("MODULE 22 COMPLETE")
    print(f"Outputs: {OUTDIR}")
    print("=" * 74)
    return results_os, results_dfs, summary, coverage


if __name__ == "__main__":
    main()
