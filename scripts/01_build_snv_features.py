#!/usr/bin/env python3

import os
import re
import gzip
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
from multiprocessing import Pool, cpu_count

##############################################################################
# PATHS
##############################################################################

PROJECT_ROOT = Path(
    os.environ.get("CRC_BASE", Path(__file__).resolve().parents[1])
).resolve()
RAW_ROOT = Path(
    os.environ.get("CRC_RAW_ROOT", PROJECT_ROOT.parent / "genomics_raw_vcf")
).resolve()
VEP_DIR = str(Path(os.environ.get("CRC_VEP_DIR", RAW_ROOT / "vep_output")))
CRC_PANEL_FILE = str(Path(os.environ.get(
    "CRC_PANEL_FILE", PROJECT_ROOT / "colorectal_cancer_all_genes.txt"
)))
OUT_DIR = str(Path(os.environ.get(
    "CRC_M1_OUT", PROJECT_ROOT / "module1_results"
)))

os.makedirs(OUT_DIR, exist_ok=True)

##############################################################################
# PARAMETERS
##############################################################################

MIN_TVAF = 0.05
MIN_TDP = 20
MAX_NVAF = 0.02

MIN_FREQ = 0.02
MAX_FREQ = 0.98

MIN_OCCURRENCE = int(1063 * MIN_FREQ)

##############################################################################
# PASSENGERS
##############################################################################

PASSENGER_GENES = set([
    "TTN","MUC16","MUC4","MUC17","MUC5B","MUC6",
    "OBSCN","FLG","DNAH5","DNAH11","DNAH2",
    "RYR1","RYR2","RYR3","CSMD1","CSMD3",
    "LRP1B","PCLO","XIRP2","FAT4","FAT3"
])

##############################################################################
# LOAD CRC PANEL
##############################################################################

crc_panel = pd.read_csv(
    CRC_PANEL_FILE,
    sep="\t",
    header=None
)

CRC_GENES = set([
    str(x).strip().upper()
    for x in crc_panel.iloc[:,0]
    if pd.notnull(x)
])

##############################################################################
# VEP CONSEQUENCE FILTERS
##############################################################################

FUNCTIONAL_TERMS = set([

    "missense_variant",
    "frameshift_variant",
    "stop_gained",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "protein_altering_variant",
    "inframe_insertion",
    "inframe_deletion",
    "start_lost",
    "stop_lost"
])

LOF_TERMS = set([
    "frameshift_variant",
    "stop_gained",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "start_lost"
])

##############################################################################
# HELPERS
##############################################################################

def open_vcf(path):

    return gzip.open(path, "rt")

def extract_sid(fname):

    m = re.search(
        r"CRC-SW-((?:U|UM)\d+)-T",
        fname
    )

    return m.group(1)

##############################################################################
# PARSER
##############################################################################

def parse_sample(vcf_path):

    sid = extract_sid(vcf_path)

    gene_hits = set()

    mechanism_hits = defaultdict(set)

    total_mut = 0

    csq_fields = None

    with open_vcf(vcf_path) as fh:

        for line in fh:

            if line.startswith("##INFO=<ID=CSQ"):

                m = re.search(
                    r'Format: ([^"]+)"',
                    line
                )

                if m:
                    csq_fields = (
                        m.group(1)
                        .strip()
                        .split("|")
                    )

                continue

            if line.startswith("#"):
                continue

            parts = line.rstrip().split("\t")

            if len(parts) < 8:
                continue

            chrom, pos, _, ref, alt, _, flt, info = parts[:8]

            if flt != "PASS":
                continue

            info_dict = {}

            for item in info.split(";"):

                if "=" in item:

                    k, v = item.split("=", 1)

                    info_dict[k] = v

            try:

                tvaf = float(
                    info_dict.get("TVAF", 0)
                )

                tdp = int(
                    info_dict.get("TDP", 0)
                )

                nvaf = float(
                    info_dict.get("NVAF", 0)
                )

            except:
                continue

            if tvaf < MIN_TVAF:
                continue

            if tdp < MIN_TDP:
                continue

            if nvaf > MAX_NVAF:
                continue

            if "CSQ" not in info_dict:
                continue

            if not csq_fields:
                continue

            for entry in info_dict["CSQ"].split(","):

                vals = entry.split("|")

                if len(vals) < len(csq_fields):
                    continue

                d = dict(zip(csq_fields, vals))

                if d.get("CANONICAL") != "YES":
                    continue

                gene = (
                    d.get("SYMBOL", "")
                    .strip()
                    .upper()
                )

                if not gene:
                    continue

                consequences = set(
                    d.get("Consequence", "")
                    .split("&")
                )

                if not (
                    consequences &
                    FUNCTIONAL_TERMS
                ):
                    continue

                gene_hits.add(gene)

                total_mut += 1

                if consequences & LOF_TERMS:

                    mechanism_hits[
                        f"SNV_{gene}_LOF"
                    ].add(sid)

                else:

                    mechanism_hits[
                        f"SNV_{gene}_MISSENSE"
                    ].add(sid)

                break

    return sid, gene_hits, mechanism_hits, total_mut

##############################################################################
# RUN
##############################################################################

vep_files = sorted(
    glob.glob(f"{VEP_DIR}/*.vep.vcf.gz")
)

print(f"Found VCFs: {len(vep_files)}")

N_CORES = min(16, cpu_count()-1)

with Pool(N_CORES) as pool:

    results = pool.map(
        parse_sample,
        vep_files
    )

##############################################################################
# COLLECT
##############################################################################

sample_gene_hits = {}
sample_burden = {}

mechanism_dict = defaultdict(set)

gene_freq = defaultdict(int)

for sid, genes, mech, burden in results:

    genes = set([
        g for g in genes
        if g not in PASSENGER_GENES
    ])

    sample_gene_hits[sid] = genes

    sample_burden[sid] = burden

    for g in genes:
        gene_freq[g] += 1

    for k, v in mech.items():
        mechanism_dict[k].update(v)

##############################################################################
# DISCOVERY GENES
# RECURRENT CRC-PANEL GENES ONLY
##############################################################################

discovery_genes = set([

    g for g, c in gene_freq.items()

    if (

        ##############################################################
        # MUST EXIST IN CRC MASTER PANEL
        ##############################################################

        g in CRC_GENES

        ##############################################################
        # RECURRENT
        ##############################################################

        and c >= MIN_OCCURRENCE

        ##############################################################
        # NOT NEAR-UNIVERSAL
        ##############################################################

        and c <= int(1063 * MAX_FREQ)
    )
])

print(
    f"Discovery SNV genes retained: "
    f"{len(discovery_genes)}"
)

##############################################################################
# CRC GENES
##############################################################################

crc_genes_present = set([

    g for g in gene_freq

    if g in CRC_GENES
])

##############################################################################
# BUILD MATRICES
##############################################################################

samples = sorted(sample_gene_hits.keys())

##############################################################################
# CRC MATRIX
##############################################################################

crc_features = sorted([
    f"SNV_{g}"
    for g in crc_genes_present
])

crc_mat = pd.DataFrame(
    0,
    index=samples,
    columns=crc_features,
    dtype=np.int8
)

for sid in samples:

    genes = sample_gene_hits[sid]

    for g in genes:

        feat = f"SNV_{g}"

        if feat in crc_mat.columns:

            crc_mat.loc[sid, feat] = 1

##############################################################################
# DISCOVERY MATRIX
##############################################################################

disc_features = sorted([
    f"SNV_{g}"
    for g in discovery_genes
])

disc_mat = pd.DataFrame(
    0,
    index=samples,
    columns=disc_features,
    dtype=np.int8
)

for sid in samples:

    genes = sample_gene_hits[sid]

    for g in genes:

        feat = f"SNV_{g}"

        if feat in disc_mat.columns:

            disc_mat.loc[sid, feat] = 1

##############################################################################
# MECHANISM MATRIX
##############################################################################

mech_features = sorted(
    mechanism_dict.keys()
)

mech_mat = pd.DataFrame(
    0,
    index=samples,
    columns=mech_features,
    dtype=np.int8
)

for feat, sids in mechanism_dict.items():

    mech_mat.loc[
        list(sids),
        feat
    ] = 1

##############################################################################
# BURDEN MATRIX
##############################################################################

burden_mat = pd.DataFrame({

    "sample_id": samples,

    "SNV_TOTAL_BURDEN": [

        sample_burden[s]
        for s in samples
    ]
})

##############################################################################
# SAVE
##############################################################################

crc_mat.to_csv(
    f"{OUT_DIR}/module1_crc_focused_binary_matrix.csv"
)

disc_mat.to_csv(
    f"{OUT_DIR}/module1_discovery_binary_matrix.csv"
)

mech_mat.to_csv(
    f"{OUT_DIR}/module1_snv_mechanism_matrix.csv"
)

burden_mat.to_csv(
    f"{OUT_DIR}/module1_snv_burden_matrix.csv",
    index=False
)

print("\nMODULE 1 COMPLETE")
print(f"CRC matrix shape: {crc_mat.shape}")
print(f"Discovery matrix shape: {disc_mat.shape}")
