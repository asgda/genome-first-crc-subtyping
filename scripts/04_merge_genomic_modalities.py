#!/usr/bin/env python3

##############################################################################
# MODULE 4
# MULTIMODAL FUSION
##############################################################################

import os
from pathlib import Path
import pandas as pd

##############################################################################
# PATHS
##############################################################################

BASE = str(Path(
    os.environ.get("CRC_BASE", Path(__file__).resolve().parents[1])
).resolve())

SNV_DIR = f"{BASE}/module1_results"
CNV_DIR = f"{BASE}/module2_results"
SV_DIR  = f"{BASE}/module3_results"

OUT_DIR = f"{BASE}/module4_results"

os.makedirs(OUT_DIR, exist_ok=True)

##############################################################################
# LOAD MATRICES
##############################################################################

print("\nLoading discovery matrices...")

snv_disc = pd.read_csv(
    f"{SNV_DIR}/module1_discovery_binary_matrix.csv",
    index_col=0
)

cnv_disc = pd.read_csv(
    f"{CNV_DIR}/module2_discovery_binary_matrix.csv",
    index_col=0
)

sv_disc = pd.read_csv(
    f"{SV_DIR}/module3_discovery_binary_matrix.csv",
    index_col=0
)

##############################################################################
# PREFIXES
##############################################################################

snv_disc.columns = [
    f"SNV_{x.replace('SNV_', '')}"
    for x in snv_disc.columns
]

cnv_disc.columns = [
    f"CNV_{x.replace('CNV_', '')}"
    for x in cnv_disc.columns
]

sv_disc.columns = [
    f"SV_{x.replace('SV_', '')}"
    for x in sv_disc.columns
]

##############################################################################
# COMMON SAMPLES
##############################################################################

common_samples = sorted(

    set(snv_disc.index)
    &
    set(cnv_disc.index)
    &
    set(sv_disc.index)
)

print(f"\nCommon samples: {len(common_samples)}")

##############################################################################
# SUBSET
##############################################################################

snv_disc = snv_disc.loc[common_samples]
cnv_disc = cnv_disc.loc[common_samples]
sv_disc  = sv_disc.loc[common_samples]

##############################################################################
# CONCAT
##############################################################################

print("\nMerging modalities...")

multi_disc = pd.concat(

    [
        snv_disc,
        cnv_disc,
        sv_disc
    ],

    axis=1
)

##############################################################################
# REMOVE DUPLICATES
##############################################################################

multi_disc = multi_disc.loc[
    :,
    ~multi_disc.columns.duplicated()
]

##############################################################################
# SAVE
##############################################################################

multi_disc.to_csv(
    f"{OUT_DIR}/module4_unified_discovery_matrix.csv"
)

##############################################################################
# SUMMARY
##############################################################################

print("\n================================================")
print("MODULE 4 COMPLETE")
print("================================================")

print(f"\nFinal matrix shape: {multi_disc.shape}")

print("\nModality contribution:")

print(f"SNV : {snv_disc.shape[1]}")
print(f"CNV : {cnv_disc.shape[1]}")
print(f"SV  : {sv_disc.shape[1]}")

print(f"\nTOTAL FEATURES: {multi_disc.shape[1]}")

print("\nDone.\n")
