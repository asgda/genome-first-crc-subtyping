#!/usr/bin/env python3

##############################################################################
# MODULE 3
# FINAL CRC SV PIPELINE
#
# FINAL BIOLOGICAL DESIGN
#
# 1. Gene-centric SV representation
# 2. Same CRC panel as SNV/CNV
# 3. Uses VCF-native transcript annotations (SID)
# 4. Discovery = recurrent CRC genes only
# 5. CFS genes excluded from clustering
# 6. Architecture matrices retained separately
# 7. Reciprocal BRASS breakends collapsed to one SV junction
#
##############################################################################

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
SV_DIR = str(Path(os.environ.get("CRC_SV_DIR", RAW_ROOT / "sv")))
CRC_PANEL_FILE = str(Path(os.environ.get(
    "CRC_PANEL_FILE", PROJECT_ROOT / "colorectal_cancer_all_genes.txt"
)))
OUT_DIR = str(Path(os.environ.get(
    "CRC_M3_OUT", PROJECT_ROOT / "module3_results"
)))

os.makedirs(OUT_DIR, exist_ok=True)

##############################################################################
# PARAMETERS
##############################################################################

N_CORES = min(16, cpu_count()-1)

##############################################################################
# SVs are sparse
# 1% recurrence threshold is biologically appropriate
##############################################################################

MIN_FREQ = 0.01
MAX_FREQ = 0.98

MIN_OCCURRENCE = int(1063 * MIN_FREQ)

##############################################################################
# PASSENGERS
##############################################################################

PASSENGER_GENES = set([

    "TTN","MUC16","MUC4","MUC17",
    "MUC5B","MUC6","OBSCN",

    "DNAH5","DNAH11","DNAH2",

    "RYR1","RYR2","RYR3",

    "CSMD1","CSMD3",

    "LRP1B","PCLO","XIRP2",

    "FAT3","FAT4"
])

##############################################################################
# COMMON FRAGILE SITE GENES
##############################################################################

CFS_GENES = set([

    "FHIT",
    "WWOX",
    "PARK2",
    "IMMP2L",
    "DMD",
    "GRID2",
    "MACROD2"
])

##############################################################################
# LOAD CRC MASTER PANEL
##############################################################################

print("\nLoading CRC master panel...")

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

CRC_GENES = set([

    x for x in CRC_GENES

    if x not in [
        "",
        "GENE",
        "NAN"
    ]
])

print(f"CRC genes loaded: {len(CRC_GENES)}")

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

    if m:
        return m.group(1)

    return None

##############################################################################
# SV PARSER
##############################################################################

def parse_sv_sample(vcf_path):

    sid = extract_sid(
        os.path.basename(vcf_path)
    )

    print(f"Processing: {sid}")

    ##############################################################
    # STORAGE
    ##############################################################

    sv_gene_hits = set()

    sv_mechanisms = defaultdict(set)

    sv_class_counts = defaultdict(int)

    # BRASS represents one rearrangement as two reciprocal BND records.
    # Burden and SV-class counts must therefore be junction-level, not
    # breakend-record-level.  Gene annotations are still read from every
    # record below because the two mates can annotate different genes.
    event_classes = {}

    event_record_counts = defaultdict(int)

    accepted_breakend_records = 0

    cfs_hits = set()

    chromothripsis_flag = 0

    l1_flag = 0

    total_sv = 0

    ##############################################################
    # OPEN VCF
    ##############################################################

    with open_vcf(vcf_path) as fh:

        for line in fh:

            if line.startswith("#"):
                continue

            parts = line.rstrip().split("\t")

            if len(parts) < 8:
                continue

            chrom = parts[0]

            try:
                pos = int(parts[1])
            except:
                continue

            alt = parts[4]

            filt = parts[6]

            info = parts[7]

            ######################################################
            # FILTER
            ######################################################

            if filt not in ["PASS", "."]:
                continue

            ######################################################
            # INFO PARSE
            ######################################################

            info_dict = {}

            for item in info.split(";"):

                if "=" in item:

                    k, v = item.split("=", 1)

                    info_dict[k] = v

            accepted_breakend_records += 1

            ######################################################
            # SVTYPE
            ######################################################

            svtype = info_dict.get(
                "SVTYPE",
                "UNK"
            )

            ######################################################
            # CHROMOTHRIPSIS
            ######################################################

            if svtype in [
                "BND",
                "INV",
                "TRA"
            ]:
                chromothripsis_flag = 1

            ######################################################
            # LINE1
            ######################################################

            if (
                "LINE1" in info
                or
                "L1" in info
            ):
                l1_flag = 1

            ######################################################
            # SV CLASS COUNTS
            ######################################################

            svclass = info_dict.get(
                "SVCLASS",
                svtype
            )

            record_id = parts[2].strip() if len(parts) > 2 else ""
            mate_ids = [
                x.strip() for x in info_dict.get("MATEID", "").split(",")
                if x.strip() not in {"", "."}
            ]

            if record_id not in {"", "."} and mate_ids:
                # An unordered ID set gives the same key from either mate.
                event_key = ("MATE_PAIR",) + tuple(sorted(set([record_id] + mate_ids)))
            elif record_id not in {"", "."}:
                # Retain an unpaired caller record as one singleton event.
                event_key = ("SINGLETON_ID", record_id)
            else:
                # Last-resort stable key for malformed records without IDs.
                event_key = ("SINGLETON_COORD", chrom, str(pos), alt)

            previous_class = event_classes.get(event_key)
            event_record_counts[event_key] += 1
            if previous_class is None:
                event_classes[event_key] = svclass
                sv_class_counts[svclass] += 1
                total_sv += 1
            elif previous_class != svclass:
                raise ValueError(
                    f"Conflicting SVCLASS values within BRASS mate pair "
                    f"{event_key}: {previous_class!r} versus {svclass!r} "
                    f"in {vcf_path}"
                )

            ######################################################
            # GENE EXTRACTION
            ######################################################

            ######################################################
            # IMPORTANT:
            # SID contains real transcript-aware genes
            ######################################################

            gene_field = info_dict.get(
                "SID",
                ""
            )

            ######################################################
            # NO GENE
            ######################################################

            if gene_field == "":
                continue

            ######################################################
            # MULTI-GENE EVENTS
            ######################################################

            genes = [

                x.strip().upper()

                for x in gene_field.split(",")

                if x.strip() != ""
            ]

            ######################################################
            # PROCESS GENES
            ######################################################

            for gene in genes:

                ##################################################
                # REMOVE WEIRD IDS
                ##################################################

                if not re.match(
                    r"^[A-Z0-9\-\._]+$",
                    gene
                ):
                    continue

                ##################################################
                # CRC PANEL ONLY
                ##################################################

                if gene not in CRC_GENES:
                    continue

                ##################################################
                # PASSENGERS
                ##################################################

                if gene in PASSENGER_GENES:
                    continue

                ##################################################
                # COMMON FRAGILE SITES
                ##################################################

                if gene in CFS_GENES:

                    cfs_hits.add(gene)

                    continue

                ##################################################
                # MAIN FEATURE
                ##################################################

                sv_gene_hits.add(gene)

                ##################################################
                # MECHANISM MATRIX
                ##################################################

                sv_mechanisms[
                    f"SV_{gene}_{svtype}"
                ].add(sid)

    ##############################################################
    # RETURN
    ##############################################################

    unresolved_singletons = [
        key for key, count in event_record_counts.items()
        if key[0] == "MATE_PAIR" and count == 1
    ]

    return {

        "sid":
            sid,

        "genes":
            sv_gene_hits,

        "mechanisms":
            sv_mechanisms,

        "sv_class_counts":
            sv_class_counts,

        "cfs_hits":
            cfs_hits,

        "chromothripsis":
            chromothripsis_flag,

        "l1":
            l1_flag,

        "total_sv":
            total_sv,

        "accepted_breakend_records":
            accepted_breakend_records,

        "unresolved_singletons":
            unresolved_singletons
    }

##############################################################################
# FIND FILES
##############################################################################

sv_files = sorted(
    glob.glob(
        f"{SV_DIR}/*.vcf.gz"
    )
)

print(f"\nTotal SV files: {len(sv_files)}")

##############################################################################
# PARALLEL PROCESSING
##############################################################################

with Pool(N_CORES) as pool:

    results = pool.map(
        parse_sv_sample,
        sv_files
    )

##############################################################################
# STORAGE
##############################################################################

sample_gene_hits = {}

gene_freq = defaultdict(int)

mechanism_dict = defaultdict(set)

sv_burden_rows = []

sv_arch_rows = []

##############################################################################
# PROCESS RESULTS
##############################################################################

for r in results:

    sid = r["sid"]

    genes = set(r["genes"])

    sample_gene_hits[sid] = genes

    ##############################################################
    # GENE FREQUENCY
    ##############################################################

    for g in genes:

        gene_freq[g] += 1

    ##############################################################
    # MECHANISM MATRIX
    ##############################################################

    for k, v in r["mechanisms"].items():

        mechanism_dict[k].update(v)

    ##############################################################
    # BURDEN MATRIX
    ##############################################################

    sv_burden_rows.append({

        "sample_id":
            sid,

        "SV_TOTAL":
            r["total_sv"]
    })

    ##############################################################
    # ARCHITECTURE MATRIX
    ##############################################################

    row = {

        "sample_id":
            sid,

        "SV_chromothripsis":
            r["chromothripsis"],

        "SV_LINE1":
            r["l1"],

        "SV_CFS_burden":
            len(r["cfs_hits"])
    }

    ##############################################################
    # SV CLASS COUNTS
    ##############################################################

    for k, v in r["sv_class_counts"].items():

        row[f"SVCLASS_{k}"] = v

    sv_arch_rows.append(row)

##############################################################################
# EVENT-COUNT AUDIT
##############################################################################

breakend_total = sum(r["accepted_breakend_records"] for r in results)
event_total = sum(r["total_sv"] for r in results)
unresolved = [
    (r["sid"], key) for r in results for key in r["unresolved_singletons"]
]
print(
    f"\nSV counting audit: {breakend_total} accepted BRASS breakend records "
    f"resolved as {event_total - len(unresolved)} reciprocal MATEID-paired "
    f"junctions plus {len(unresolved)} unresolved singleton(s), for "
    f"{event_total} unique SV junctions in total."
)
if unresolved:
    print(
        f"QC flag: {len(unresolved)} retained mate-pair key(s) had only one "
        f"breakend record and were counted once as singleton junctions: {unresolved}"
    )

##############################################################################
# DISCOVERY GENES
##############################################################################

print("\nSelecting recurrent discovery genes...")

discovery_genes = set([

    g for g, c in gene_freq.items()

    if (

        ##########################################################
        # CRC PANEL
        ##########################################################

        g in CRC_GENES

        ##########################################################
        # RECURRENT
        ##########################################################

        and c >= MIN_OCCURRENCE

        ##########################################################
        # NOT UNIVERSAL
        ##########################################################

        and c <= int(1063 * MAX_FREQ)
    )
])

print(
    f"Discovery SV genes retained: "
    f"{len(discovery_genes)}"
)

##############################################################################
# CRC GENES PRESENT
##############################################################################

crc_genes_present = set([

    g for g in gene_freq

    if g in CRC_GENES
])

##############################################################################
# BUILD MATRICES
##############################################################################

samples = sorted(
    sample_gene_hits.keys()
)

##############################################################################
# CRC MATRIX
##############################################################################

print("\nBuilding CRC-focused matrix...")

crc_features = sorted([

    f"SV_{g}"

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

        feat = f"SV_{g}"

        if feat in crc_mat.columns:

            crc_mat.loc[sid, feat] = 1

##############################################################################
# DISCOVERY MATRIX
##############################################################################

print("Building discovery matrix...")

disc_features = sorted([

    f"SV_{g}"

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

        feat = f"SV_{g}"

        if feat in disc_mat.columns:

            disc_mat.loc[sid, feat] = 1

##############################################################################
# MECHANISM MATRIX
##############################################################################

print("Building mechanism matrix...")

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

print("Building burden matrix...")

burden_mat = pd.DataFrame(
    sv_burden_rows
)

##############################################################################
# ARCHITECTURE MATRIX
##############################################################################

print("Building SV architecture matrix...")

arch_mat = pd.DataFrame(
    sv_arch_rows
)

arch_mat = arch_mat.fillna(0)

##############################################################################
# SAVE
##############################################################################

print("\nSaving outputs...")

crc_mat.to_csv(
    f"{OUT_DIR}/module3_crc_focused_binary_matrix.csv"
)

disc_mat.to_csv(
    f"{OUT_DIR}/module3_discovery_binary_matrix.csv"
)

mech_mat.to_csv(
    f"{OUT_DIR}/module3_sv_mechanism_matrix.csv"
)

burden_mat.to_csv(
    f"{OUT_DIR}/module3_sv_burden_matrix.csv",
    index=False
)

arch_mat.to_csv(
    f"{OUT_DIR}/module3_sv_architecture_matrix.csv",
    index=False
)

##############################################################################
# SUMMARY
##############################################################################

print("\n===================================================")
print("MODULE 3 COMPLETE")
print("===================================================")

print("\nGenerated files:\n")

for x in sorted(os.listdir(OUT_DIR)):
    print(x)

print("\nDone.\n")
