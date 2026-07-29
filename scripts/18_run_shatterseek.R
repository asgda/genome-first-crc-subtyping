#!/usr/bin/env Rscript
# Run ShatterSeek on per-sample BRASS/FACETS tables prepared by script 17.
# Candidate and high-confidence classification is performed in script 19.

required_packages <- c("data.table", "ShatterSeek")
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

suppressPackageStartupMessages({
  library(data.table)
  library(ShatterSeek)
})

# -----------------------------------------------------------------------
# CONFIG (env-overridable, same convention as the Python modules)
# -----------------------------------------------------------------------
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1]])))
} else {
  getwd()
}
BASE <- Sys.getenv("CRC_BASE", normalizePath(file.path(script_dir, ".."), mustWork = FALSE))
OUTDIR    <- Sys.getenv("CRC_M10_OUT", file.path(BASE, "module10_chromothripsis_C1C4_ppt"))
INTERDIR  <- file.path(OUTDIR, "intermediate")
RAWDIR    <- file.path(OUTDIR, "raw")
dir.create(RAWDIR, showWarnings = FALSE, recursive = TRUE)

SV_INPUT  <- Sys.getenv("CRC_M10_SV_INPUT",  file.path(INTERDIR, "shatterseek_sv_input.csv"))
CNV_INPUT <- Sys.getenv("CRC_M10_CNV_INPUT", file.path(INTERDIR, "shatterseek_cnv_input.csv"))
GENOME    <- Sys.getenv("CRC_M10_GENOME", "hg38")

# 0 = process every sample present; set e.g. CRC_M10_LIMIT=10 for a smoke test
LIMIT   <- as.integer(Sys.getenv("CRC_M10_LIMIT", "0"))
NCORES  <- as.integer(Sys.getenv("CRC_M10_NCORES", "1"))
MIN_SIZE <- as.numeric(Sys.getenv("CRC_M10_MINSIZE", "1"))  # ShatterSeek default

cat(strrep("=", 70), "\n")
cat("MODULE 10b -- RUN SHATTERSEEK\n")
cat(strrep("=", 70), "\n")
cat("SV input :", SV_INPUT, "\n")
cat("CNV input:", CNV_INPUT, "\n")
cat("Genome   :", GENOME, "\n")
cat("Cores    :", NCORES, "\n")

if (!file.exists(SV_INPUT))  stop("SV input not found: ", SV_INPUT, " -- run 10a first.")
if (!file.exists(CNV_INPUT)) stop("CNV input not found: ", CNV_INPUT, " -- run 10a first.")

sv_all  <- fread(SV_INPUT,  colClasses = list(character = c("chrom1", "chrom2", "strand1", "strand2", "SVtype")))
cnv_all <- fread(CNV_INPUT, colClasses = list(character = "chrom"))

samples <- sort(unique(c(sv_all$sample_id, cnv_all$sample_id)))
if (LIMIT > 0) {
  samples <- head(samples, LIMIT)
  cat("CRC_M10_LIMIT=", LIMIT, " -- SMOKE-TEST MODE, not processing the full cohort\n", sep = "")
}
cat("Samples to process:", length(samples), "\n\n")

# -----------------------------------------------------------------------
# PER-SAMPLE WORKER
# -----------------------------------------------------------------------
run_one_sample <- function(sid) {
  sv_s  <- sv_all[sample_id == sid]
  cnv_s <- cnv_all[sample_id == sid]

  if (nrow(sv_s) == 0 || nrow(cnv_s) == 0) {
    return(list(sample_id = sid, chromSummary = NULL,
                error = "zero SV records or zero CNV segments for this sample"))
  }

  out <- tryCatch({
    SV_data <- SVs(chrom1 = as.character(sv_s$chrom1),
                    pos1   = as.numeric(sv_s$pos1),
                    chrom2 = as.character(sv_s$chrom2),
                    pos2   = as.numeric(sv_s$pos2),
                    SVtype = as.character(sv_s$SVtype),
                    strand1 = as.character(sv_s$strand1),
                    strand2 = as.character(sv_s$strand2))

    CN_data <- CNVsegs(chrom = as.character(cnv_s$chrom),
                        start = as.numeric(cnv_s$start),
                        end   = as.numeric(cnv_s$end),
                        total_cn = as.numeric(cnv_s$total_cn))

    result <- shatterseek(SV.sample = SV_data, seg.sample = CN_data,
                           min.Size = MIN_SIZE, genome = GENOME)
    cs <- result@chromSummary
    cs$sample_id <- sid
    list(sample_id = sid, chromSummary = cs, error = NA_character_)
  }, error = function(e) {
    list(sample_id = sid, chromSummary = NULL, error = conditionMessage(e))
  })
  out
}

# -----------------------------------------------------------------------
# RUN (parallel if NCORES > 1; parallel::mclapply is base R, no extra dep)
# -----------------------------------------------------------------------
t0 <- Sys.time()
if (NCORES > 1 && .Platform$OS.type == "unix") {
  library(parallel)
  results <- mclapply(samples, run_one_sample, mc.cores = NCORES)
} else {
  results <- vector("list", length(samples))
  for (i in seq_along(samples)) {
    results[[i]] <- run_one_sample(samples[i])
    if (i %% 25 == 0 || i == length(samples)) {
      elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
      cat(sprintf("  %d/%d samples done (%.0fs elapsed, %.1fs/sample avg)\n",
                   i, length(samples), elapsed, elapsed / i))
    }
  }
}

# -----------------------------------------------------------------------
# COLLECT + WRITE
# -----------------------------------------------------------------------
ok_tables <- lapply(results, function(r) r$chromSummary)
ok_tables <- ok_tables[!sapply(ok_tables, is.null)]

errors <- data.table(
  sample_id = sapply(results, function(r) r$sample_id),
  error     = sapply(results, function(r) ifelse(is.na(r$error), "", r$error))
)
errors <- errors[error != ""]

n_ok <- length(ok_tables)
n_err <- nrow(errors)
cat("\n", strrep("=", 70), "\n", sep = "")
cat("RESULT: ", n_ok, "/", length(samples), " samples succeeded, ",
    n_err, " failed\n", sep = "")

if (n_ok > 0) {
  combined <- rbindlist(ok_tables, fill = TRUE)
  out_path <- file.path(RAWDIR, "shatterseek_chromsummary_raw.csv")
  fwrite(combined, out_path)
  cat("Combined chromSummary written: ", out_path,
      " (", nrow(combined), " chromosome-level rows across ", n_ok, " samples)\n", sep = "")
  cat("Columns:", paste(names(combined), collapse = ", "), "\n")
} else {
  cat("WARNING: no samples succeeded -- nothing written. Check errors below.\n")
}

if (n_err > 0) {
  err_path <- file.path(RAWDIR, "shatterseek_run_errors.csv")
  fwrite(errors, err_path)
  cat("Errors logged: ", err_path, " (", n_err, " samples)\n", sep = "")
  cat("First few errors:\n")
  print(head(errors, 10))
}

cat("\nNext step: run 10c_module10_chromothripsis_analysis_C1C4_ppt.py on the raw chromSummary.\n")
