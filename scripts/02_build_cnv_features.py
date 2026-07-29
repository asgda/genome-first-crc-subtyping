#!/usr/bin/env python3

#####################################################################
# MODULE 2
# FINAL CRC CNV PIPELINE
#
# FINAL BIOLOGICAL DESIGN
#
# OUTPUTS:
#
# 1. CRC-focused binary clustering matrix
# 2. Discovery binary clustering matrix
# 3. CRC directional interpretation matrix
# 4. Discovery directional interpretation matrix
# 5. Gene-level TCN matrix
# 6. Arm-level CIN matrix
#
# KEY FEATURES:
#
# - Event-centric binary CNV representation
# - Arm-level CIN retained
# - CRC-focused and discovery matrices
# - Passenger filtering
# - 2% recurrence filtering
# - Broad-passenger suppression
# - Parallelized processing
#
#####################################################################

import os
import re
import gzip
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

from multiprocessing import Pool

warnings.filterwarnings("ignore")

#####################################################################
# PATHS
#####################################################################

BASE_DIR = str(Path(
    os.environ.get("CRC_BASE", Path(__file__).resolve().parents[1])
).resolve())
CNV_DIR = os.environ.get(
    "CRC_CNV_DIR", str(Path(BASE_DIR).parent / "genomics_raw_vcf" / "cnv")
)
REF_DIR = os.environ.get(
    "CRC_REFERENCE_DIR", str(Path(BASE_DIR).parent / "reference")
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "module2_results"
)
OUT_DIR = os.environ.get("CRC_M2_OUT", OUT_DIR)

os.makedirs(OUT_DIR, exist_ok=True)

#####################################################################
# CPU SETTINGS
#####################################################################

N_CORES = 16

#####################################################################
# CRC DRIVER GENES
#####################################################################

#####################################################################
# LOAD MASTER CRC GENE PANEL
#####################################################################

print("\nLoading master CRC gene panel...")

CRC_PANEL_FILE = (
    "colorectal_cancer_all_genes.txt"
)

crc_panel = pd.read_csv(
    CRC_PANEL_FILE,
    sep="\t",
    header=None
)

#####################################################################
# CLEAN GENE LIST
#####################################################################

crc_genes = set([

    str(x).strip().upper()

    for x in crc_panel.iloc[:,0].tolist()

    if pd.notnull(x)
])

#####################################################################
# REMOVE HEADER ARTIFACTS
#####################################################################

crc_genes = set([

    x for x in crc_genes

    if x not in [
        "GENE",
        "",
        "NAN"
    ]
])

#####################################################################
# FINAL CRC GENE SET
#####################################################################

CRC_GENES = crc_genes

print(
    f"Loaded CRC panel genes: "
    f"{len(CRC_GENES)}"
)

#####################################################################
# KNOWN PASSENGERS
#####################################################################

PASSENGER_GENES = set([

    "TTN","MUC16","OBSCN",

    "DNAH5","DNAH6","DNAH7",
    "DNAH8","DNAH9","DNAH10",
    "DNAH11","DNAH14","DNAH17",

    "CSMD1","CSMD3",

    "RYR1","RYR2","RYR3",

    "SYNE1","SYNE2",

    "USH2A","FLG",

    "PCLO","XIRP2",

    "FAT3","FAT4",

    "LRP1B","MACROD2"
])

#####################################################################
# LOAD GENE BED
#####################################################################

print("\nLoading gene BED...")

gene_bed = pd.read_csv(
    os.path.join(
        REF_DIR,
        "genes_protein_coding_hg38.bed"
    ),
    sep="\t",
    header=None
)

gene_bed = gene_bed.iloc[:, 0:4]

gene_bed.columns = [
    "chr",
    "start",
    "end",
    "gene"
]

#####################################################################
# KEEP ONLY CRC MASTER PANEL GENES
#####################################################################

gene_bed["gene"] = (
    gene_bed["gene"]
    .astype(str)
    .str.upper()
)

gene_bed = gene_bed[

    gene_bed["gene"].isin(
        CRC_GENES
    )
].copy()

#####################################################################
# REMOVE KNOWN PASSENGERS
#####################################################################

gene_bed = gene_bed[
    ~gene_bed["gene"].isin(
        PASSENGER_GENES
    )
].copy()

print(
    f"Genes retained in BED: "
    f"{gene_bed.shape[0]}"
)

#####################################################################
# LOAD CYTOBANDS
#####################################################################

print("Loading cytobands...")

cyto = pd.read_csv(
    os.path.join(
        REF_DIR,
        "cytoBand.txt.gz"
    ),
    sep="\t",
    compression="gzip",
    header=None
)

cyto.columns = [
    "chr",
    "start",
    "end",
    "band",
    "stain"
]

#####################################################################
# CREATE ARM LABELS
#####################################################################

cyto["arm"] = np.where(

    cyto["band"].str.startswith("p"),

    cyto["chr"] + "p",

    cyto["chr"] + "q"
)

#####################################################################
# SAMPLE PARSER
#####################################################################

def get_sample_id(filename):

    m = re.search(
        r"(U[M]?[0-9]+)",
        filename
    )

    if m:
        return m.group(1)

    return None

#####################################################################
# INFO PARSER
#####################################################################

def parse_info(info):

    vals = {}

    for x in info.split(";"):

        if "=" in x:

            k, v = x.split("=", 1)

            vals[k] = v

    return vals

#####################################################################
# PROCESS SINGLE FILE
#####################################################################

def process_cnv_file(vcf_path):

    sample_id = get_sample_id(
        os.path.basename(vcf_path)
    )

    if sample_id is None:
        return None

    print(f"Processing: {sample_id}")

    #############################################################
    # READ VCF
    #############################################################

    rows = []

    with gzip.open(vcf_path, "rt") as f:

        for line in f:

            if line.startswith("#"):
                continue

            parts = line.strip().split("\t")

            if len(parts) < 8:
                continue

            chrom = parts[0]

            try:
                start = int(parts[1])
            except:
                continue

            info = parse_info(parts[7])

            try:

                end = int(
                    info.get("END")
                )

                tcn = float(
                    info.get(
                        "TCN_EM",
                        np.nan
                    )
                )

                lcn = float(
                    info.get(
                        "LCN_EM",
                        np.nan
                    )
                )

                log2cn = float(
                    info.get(
                        "CNLR_MEDIAN",
                        np.nan
                    )
                )

            except:
                continue

            rows.append({

                "sample_id":
                    sample_id,

                "chr":
                    chrom,

                "start":
                    start,

                "end":
                    end,

                "segment_size":
                    end - start,

                "TCN":
                    tcn,

                "LCN":
                    lcn,

                "log2CN":
                    log2cn
            })

    if len(rows) == 0:
        return None

    seg = pd.DataFrame(rows)

    #############################################################
    # GENE EVENTS
    #############################################################

    gene_rows = []

    for chrom in seg["chr"].unique():

        seg_chr = seg[
            seg["chr"] == chrom
        ]

        gene_chr = gene_bed[
            gene_bed["chr"] == chrom
        ]

        if len(seg_chr) == 0:
            continue

        if len(gene_chr) == 0:
            continue

        #########################################################
        # OVERLAP GENES
        #########################################################

        for _, g in gene_chr.iterrows():

            overlaps = seg_chr[
                (seg_chr["end"] >= g["start"]) &
                (seg_chr["start"] <= g["end"])
            ]

            if len(overlaps) == 0:
                continue

            #####################################################
            # STRONGEST CNV SIGNAL
            #####################################################

            overlaps["absdev"] = np.abs(
                overlaps["TCN"] - 2
            )

            best = overlaps.sort_values(
                "absdev",
                ascending=False
            ).iloc[0]

            gene_rows.append({

                "sample_id":
                    sample_id,

                "feature":
                    f"CNV_{g['gene']}",

                #################################################
                # CLUSTERING VALUE
                #################################################

                "value":
                    int(best["TCN"] != 2),

                #################################################
                # SEGMENT SIZE
                #################################################

                "segment_size":
                    best["segment_size"],

                #################################################
                # INTERPRETATION
                #################################################

                "TCN":
                    best["TCN"],

                "log2CN":
                    best["log2CN"],

                "gain":
                    int(best["TCN"] > 2),

                "loss":
                    int(best["TCN"] < 2),

                "homdel":
                    int(best["TCN"] == 0),

                "loh":
                    int(best["LCN"] == 0)
            })

    #############################################################
    # ARM EVENTS
    #############################################################

    arm_rows = []

    for arm in cyto["arm"].unique():

        arm_df = cyto[
            cyto["arm"] == arm
        ]

        chr_name = arm_df.iloc[0]["chr"]

        arm_start = arm_df["start"].min()

        arm_end = arm_df["end"].max()

        seg_arm = seg[
            (seg["chr"] == chr_name) &
            (seg["end"] >= arm_start) &
            (seg["start"] <= arm_end)
        ]

        if len(seg_arm) == 0:
            continue

        #########################################################
        # FRACTION ALTERED
        #########################################################

        altered_fraction = np.mean(
            seg_arm["TCN"] != 2
        )

        arm_event = int(
            altered_fraction >= 0.5
        )

        #########################################################
        # CIN SCORE
        #########################################################

        cin_score = np.mean(
            np.abs(seg_arm["TCN"] - 2)
        )

        median_tcn = np.median(
            seg_arm["TCN"]
        )

        #########################################################
        # CLEAN ARM NAME
        #########################################################

        clean_arm = arm.replace(
            "chr",
            ""
        )

        arm_rows.append({

            "sample_id":
                sample_id,

            "feature":
                f"CNV_{clean_arm}",

            #####################################################
            # CLUSTERING
            #####################################################

            "value":
                arm_event,

            #####################################################
            # INTERPRETATION
            #####################################################

            "CIN_score":
                cin_score,

            "median_TCN":
                median_tcn,

            "gain":
                int(median_tcn > 2),

            "loss":
                int(median_tcn < 2)
        })

    return {

        "genes":
            pd.DataFrame(gene_rows),

        "arms":
            pd.DataFrame(arm_rows)
    }

#####################################################################
# FIND VCF FILES
#####################################################################

print("\nFinding CNV files...")

vcf_files = [

    os.path.join(CNV_DIR, x)

    for x in os.listdir(CNV_DIR)

    if x.endswith(".vcf.gz")
]

print(f"Total CNV files: {len(vcf_files)}")

#####################################################################
# PARALLEL PROCESSING
#####################################################################

print("\nRunning parallel processing...\n")

with Pool(N_CORES) as pool:

    results = pool.map(
        process_cnv_file,
        vcf_files
    )

#####################################################################
# MERGE RESULTS
#####################################################################

gene_tables = []
arm_tables = []

for r in results:

    if r is None:
        continue

    gene_tables.append(r["genes"])
    arm_tables.append(r["arms"])

gene_df = pd.concat(
    gene_tables,
    ignore_index=True
)

arm_df = pd.concat(
    arm_tables,
    ignore_index=True
)

#####################################################################
# COMBINED EVENTS
#####################################################################

print("\nBuilding combined matrices...")

combined = pd.concat([

    gene_df[
        ["sample_id","feature","value"]
    ],

    arm_df[
        ["sample_id","feature","value"]
    ]
])

#####################################################################
# UNIQUE SAMPLE-FEATURE EVENTS
#####################################################################

combined = combined.groupby(
    ["sample_id","feature"]
)["value"].max().reset_index()

unique_events = combined.drop_duplicates(
    ["sample_id","feature"]
)

#####################################################################
# RECURRENCE COUNTS
#####################################################################

feature_freq = unique_events.groupby(
    "feature"
)["value"].sum()

#####################################################################
# RECURRENCE THRESHOLD
#####################################################################

MIN_OCCURRENCE = int(
    np.ceil(1063 * 0.02)
)

#####################################################################
# ARM FEATURES
#####################################################################

arm_features = set([

    x for x in combined["feature"].unique()

    if re.match(
        r"CNV_(?:[0-9]{1,2}|X|Y)[pq]$",
        x
    )
])

#####################################################################
# GENE FEATURES
#####################################################################

gene_features = set([

    x for x in feature_freq.index

    if not re.match(
        r"CNV_(?:[0-9]{1,2}|X|Y)[pq]$",
        x
    )
])

#####################################################################
# RECURRENT GENES
#####################################################################

recurrent_gene_features = set([

    x for x in gene_features

    if (feature_freq[x] >= MIN_OCCURRENCE
        and
        x.replace("CNV_", "") in CRC_GENES )

])

#####################################################################
# BROAD PASSENGER SUPPRESSION
#####################################################################

gene_seg_summary = gene_df.groupby(
    "feature"
)["segment_size"].median()

#####################################################################
# REMOVE VERY BROAD EVENTS
#####################################################################

MAX_MEDIAN_SEGMENT = 50_000_000

recurrent_gene_features = set([

    x for x in recurrent_gene_features

    if gene_seg_summary.get(x, 0)
    <= MAX_MEDIAN_SEGMENT
])

#####################################################################
# REMOVE EXTREMELY COMMON FEATURES
#####################################################################

MAX_OCCURRENCE = int(
    1063 * 0.98
)

recurrent_gene_features = set([

    x for x in recurrent_gene_features

    if feature_freq[x]
    <= MAX_OCCURRENCE
])

#####################################################################
# REMOVE LOW-INFORMATION GENE TYPES
#####################################################################

blacklist_patterns = [

    "OR",
    "LOC",
    "LINC"
]

recurrent_gene_features = set([

    x for x in recurrent_gene_features

    if not any([
        pat in x
        for pat in blacklist_patterns
    ])
])

#####################################################################
# FINAL FEATURE SETS
#####################################################################

crc_features = set([

    f"CNV_{x}"

    for x in CRC_GENES
])

crc_features = (
    crc_features |
    arm_features
)

discovery_features = (
    recurrent_gene_features |
    arm_features
)

#####################################################################
# CRC CLUSTERING MATRIX
#####################################################################

print("\nBuilding CRC-focused matrix...")

crc_df = combined[
    combined["feature"].isin(
        crc_features
    )
]

crc_matrix = crc_df.pivot_table(

    index="sample_id",

    columns="feature",

    values="value",

    fill_value=0
)

crc_matrix = crc_matrix.astype(int)

crc_matrix.to_csv(

    os.path.join(
        OUT_DIR,
        "module2_crc_focused_binary_matrix.csv"
    )
)

#####################################################################
# DISCOVERY MATRIX
#####################################################################

print("\nBuilding discovery matrix...")

disc_df = combined[
    combined["feature"].isin(
        discovery_features
    )
]

disc_matrix = disc_df.pivot_table(

    index="sample_id",

    columns="feature",

    values="value",

    fill_value=0
)

disc_matrix = disc_matrix.astype(int)

disc_matrix.to_csv(

    os.path.join(
        OUT_DIR,
        "module2_discovery_binary_matrix.csv"
    )
)

#####################################################################
# DIRECTIONAL INTERPRETATION MATRICES
#####################################################################

print("\nBuilding interpretation matrices...")

directional_rows = []

#####################################################################
# GENE DIRECTIONAL EVENTS
#####################################################################

for _, row in gene_df.iterrows():

    base = row["feature"]

    if row["gain"] == 1:

        directional_rows.append([
            row["sample_id"],
            f"{base}_GAIN",
            1
        ])

    if row["loss"] == 1:

        directional_rows.append([
            row["sample_id"],
            f"{base}_LOSS",
            1
        ])

    if row["homdel"] == 1:

        directional_rows.append([
            row["sample_id"],
            f"{base}_HOMDEL",
            1
        ])

    if row["loh"] == 1:

        directional_rows.append([
            row["sample_id"],
            f"{base}_LOH",
            1
        ])

#####################################################################
# ARM DIRECTIONAL EVENTS
#####################################################################

for _, row in arm_df.iterrows():

    base = row["feature"]

    if row["gain"] == 1:

        directional_rows.append([
            row["sample_id"],
            f"{base}_GAIN",
            1
        ])

    if row["loss"] == 1:

        directional_rows.append([
            row["sample_id"],
            f"{base}_LOSS",
            1
        ])

directional_df = pd.DataFrame(

    directional_rows,

    columns=[
        "sample_id",
        "feature",
        "value"
    ]
)

directional_df = directional_df.groupby(
    ["sample_id","feature"]
)["value"].max().reset_index()

#####################################################################
# CRC INTERPRETATION MATRIX
#####################################################################

directional_crc = directional_df[

    directional_df["feature"].str.replace(
        "_GAIN|_LOSS|_HOMDEL|_LOH",
        "",
        regex=True
    ).isin(crc_features)
]

directional_crc_matrix = directional_crc.pivot_table(

    index="sample_id",

    columns="feature",

    values="value",

    fill_value=0
)

directional_crc_matrix = (
    directional_crc_matrix.astype(int)
)

#####################################################################
# RETAIN SAMPLES WITH ZERO DIRECTIONAL EVENTS
#
# pivot_table only ever includes a sample_id if it has at least one
# GAIN/LOSS/HOMDEL/LOH row in directional_rows (built above from
# `if row["gain"] == 1: ...` etc. -- positive events only). A sample
# whose every CRC-panel gene/arm has TCN == 2 therefore contributes
# zero rows and is silently absent from the pivoted matrix, not
# present as an all-zero row. That is a real state (no CNV alteration
# anywhere in this feature set), not missing data, and dropping the
# sample rather than encoding it as all-zero has no defensible
# rationale. Reindex against crc_matrix's sample set (built from every
# gene/arm-segment overlap regardless of value, so it already covers
# every sample with usable CNV segment data) and fill the added rows
# with 0 across all columns.
#####################################################################

directional_crc_matrix = directional_crc_matrix.reindex(
    crc_matrix.index, fill_value=0
)

directional_crc_matrix.to_csv(

    os.path.join(
        OUT_DIR,
        "module2_crc_directional_matrix.csv"
    )
)

#####################################################################
# DISCOVERY INTERPRETATION MATRIX
#####################################################################

directional_disc = directional_df[

    directional_df["feature"].str.replace(
        "_GAIN|_LOSS|_HOMDEL|_LOH",
        "",
        regex=True
    ).isin(discovery_features)
]

directional_disc_matrix = directional_disc.pivot_table(

    index="sample_id",

    columns="feature",

    values="value",

    fill_value=0
)

directional_disc_matrix = (
    directional_disc_matrix.astype(int)
)

#####################################################################
# RETAIN SAMPLES WITH ZERO DIRECTIONAL EVENTS (see the identical note
# above the CRC directional matrix for the full explanation). Reindex
# against disc_matrix's sample set, which already covers every sample
# with usable CNV segment data (built from every gene/arm overlap
# regardless of value, not from positive events only).
#####################################################################

directional_disc_matrix = directional_disc_matrix.reindex(
    disc_matrix.index, fill_value=0
)

directional_disc_matrix.to_csv(

    os.path.join(
        OUT_DIR,
        "module2_discovery_directional_matrix.csv"
    )
)

#####################################################################
# GENE TCN MATRIX
#####################################################################

print("\nBuilding gene TCN matrix...")

tcn_matrix = gene_df.pivot_table(

    index="sample_id",

    columns="feature",

    values="TCN",

    aggfunc="median",

    fill_value=2
)

tcn_matrix.to_csv(

    os.path.join(
        OUT_DIR,
        "module2_gene_tcn_matrix.csv"
    )
)

#####################################################################
# ARM CIN MATRIX
#####################################################################

print("\nBuilding arm CIN matrix...")

cin_matrix = arm_df.pivot_table(

    index="sample_id",

    columns="feature",

    values="CIN_score",

    aggfunc="median",

    fill_value=0
)

cin_matrix.to_csv(

    os.path.join(
        OUT_DIR,
        "module2_arm_cin_matrix.csv"
    )
)

#####################################################################
# SUMMARY
#####################################################################

print("\n====================================================")
print("MODULE 2 COMPLETED")
print("====================================================")

print("\nGenerated files:\n")

for x in sorted(os.listdir(OUT_DIR)):
    print(x)

print("\nDone.\n")
