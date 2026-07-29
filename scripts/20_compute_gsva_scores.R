#!/usr/bin/Rscript

# Compute Hallmark and KEGG GSVA scores for the complete CRC cohort.

gc()
rm(list=ls())

###############################################################
# LIBRARIES
###############################################################

required_packages <- c("GSVA", "msigdbr", "GSEABase", "dplyr")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages)) {
  stop(
    "Missing R packages: ",
    paste(missing_packages, collapse = ", "),
    ". Install the versions documented in README.md before running."
  )
}

library(GSVA)
library(msigdbr)
library(GSEABase)
library(dplyr)

###############################################################
# PATHS
###############################################################

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1]])))
} else {
  getwd()
}
BASE <- Sys.getenv("CRC_BASE", normalizePath(file.path(script_dir, ".."), mustWork = FALSE))

setwd(BASE)

OUTDIR <- paste0(
    BASE,
    "/crc_heterogeneity_data"
)

dir.create(
    OUTDIR,
    showWarnings = FALSE,
    recursive = TRUE
)

###############################################################
# LOAD TRAIN
###############################################################

cat("\nLoading TRAIN...\n")

train <- read.csv(
    "crc_heterogeneity_data/complete_z_scores_Train.csv",
    row.names = 1,
    check.names = FALSE
)

###############################################################
# LOAD TEST
###############################################################

cat("\nLoading TEST...\n")

test <- read.csv(
    "crc_heterogeneity_data/complete_z_scores_Test.csv",
    row.names = 1,
    check.names = FALSE
)

###############################################################
# COMBINE
###############################################################

cat("\nCombining TRAIN + TEST...\n")

expr <- rbind(
    train,
    test
)

###############################################################
# REMOVE DUPLICATES
###############################################################

expr <- expr[
    !duplicated(rownames(expr)),
]

###############################################################
# KEEP ONLY CRC SW SAMPLES
###############################################################

cat("\nKeeping CRC.SW samples only...\n")

keep <- grepl(
    "^CRC\\.SW\\.",
    rownames(expr)
)

expr <- expr[keep, ]

###############################################################
# REMOVE TCGA
###############################################################

expr <- expr[
    !grepl(
        "TCGA",
        rownames(expr)
    ),
]

###############################################################
# DIMENSIONS
###############################################################

cat("\nFinal expression matrix:\n")

print(dim(expr))

###############################################################
# TRANSPOSE
###############################################################

expr <- t(expr)

expr <- as.matrix(expr)

###############################################################
# GENE SETS
###############################################################

cat("\nLoading MSigDB gene sets...\n")

###############################################################
# HALLMARK
###############################################################

hallmark <- msigdbr(
    species = "Homo sapiens",
    collection = "H"
)

hallmark_list <- split(
    hallmark$gene_symbol,
    hallmark$gs_name
)

###############################################################
# KEGG
###############################################################

kegg <- msigdbr(
    species = "Homo sapiens",
    collection = "C2",
    subcollection = "CP:KEGG_LEGACY"
)

kegg_list <- split(
    kegg$gene_symbol,
    kegg$gs_name
)

###############################################################
# GSVA
###############################################################

cat("\nRunning HALLMARK GSVA...\n")

gsvaPar_H <- gsvaParam(
    expr,
    hallmark_list
)

gsva_hallmark <- gsva(
    gsvaPar_H,
    verbose = TRUE
)

###############################################################

cat("\nRunning KEGG GSVA...\n")

gsvaPar_K <- gsvaParam(
    expr,
    kegg_list
)

gsva_kegg <- gsva(
    gsvaPar_K,
    verbose = TRUE
)

###############################################################
# CONVERT
###############################################################

hallmark_df <- as.data.frame(
    t(gsva_hallmark)
)

kegg_df <- as.data.frame(
    t(gsva_kegg)
)

###############################################################
# ADD SAMPLE IDS
###############################################################

hallmark_df$sample_id <- rownames(hallmark_df)

kegg_df$sample_id <- rownames(kegg_df)

###############################################################
# REORDER
###############################################################

hallmark_df <- hallmark_df %>%
    dplyr::select(
        sample_id,
        everything()
    )

kegg_df <- kegg_df %>%
    dplyr::select(
        sample_id,
        everything()
    )

###############################################################
# SAVE
###############################################################

cat("\nSaving outputs...\n")

write.csv(
    hallmark_df,
    paste0(
        OUTDIR,
        "/HALLMARK_GSVA_FULL_CRC.csv"
    ),
    row.names = FALSE
)

write.csv(
    kegg_df,
    paste0(
        OUTDIR,
        "/KEGG_GSVA_FULL_CRC.csv"
    ),
    row.names = FALSE
)

###############################################################
# SUMMARY
###############################################################

cat("\n================================================\n")
cat("GSVA COMPLETE\n")
cat("================================================\n")

cat("\nHallmark matrix:\n")
print(dim(hallmark_df))

cat("\nKEGG matrix:\n")
print(dim(kegg_df))

cat("\nDone.\n")
