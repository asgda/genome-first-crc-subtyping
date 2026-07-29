#!/usr/bin/env Rscript
# Primary OS and supporting Stage I-III RFS analyses for locked C1-C4 subtypes.
# Fits unpenalized, stage-stratified Cox models; reports pairwise contrasts,
# nested likelihood-ratio tests, proportional-hazards diagnostics and
# prespecified clinical sensitivity analyses with within-family FDR control.

set.seed(42)

suppressPackageStartupMessages({
  library(survival)
  library(survminer)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readxl)
  library(broom)
  library(stringr)
  library(purrr)
  library(tibble)
  library(svglite)
  library(multcomp)  # glht()/mcp() -- all pairwise contrasts from one fit
})

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1]])))
} else {
  getwd()
}
BASE <- Sys.getenv("CRC_BASE", normalizePath(file.path(script_dir, ".."), mustWork = FALSE))
SUPP_XLSX <- Sys.getenv(
  "CRC_SUPP_TABLE",
  file.path(BASE, "crc_heterogeneity_data/Supplementary_Table_01.xlsx")
)
CLUSTER_CSV <- Sys.getenv(
  "CRC_CLUSTER_FILE",
  file.path(BASE, "module05_06_loocv_results/labels/NMF_k4_LOOCV.csv")
)
OUTDIR <- Sys.getenv("CRC_SURVIVAL_OUT", file.path(BASE, "survival_combined_R_results"))
FIGDIR <- file.path(OUTDIR, "figures")
TABDIR <- file.path(OUTDIR, "tables")
dir.create(FIGDIR, recursive = TRUE, showWarnings = FALSE)
dir.create(TABDIR, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# STYLE -- Okabe-Ito colour-blind-safe, matching every other module in the
# project (C1 orange, C2 sky blue, C3 bluish green, C4 vermillion). Large
# fonts throughout, per explicit instruction.
# ---------------------------------------------------------------------------
CB4 <- c(C1 = "#E69F00", C2 = "#56B4E9", C3 = "#009E73", C4 = "#D55E00")
SIG_COLORS <- c(significant = "#D55E00", suggestive = "#E69F00", ns = "#56B4E9")
SIG_LEVELS <- c("significant", "suggestive", "ns")

BASE_SIZE <- 22
base_theme <- theme_classic(base_size = BASE_SIZE) +
  theme(
    plot.title    = element_text(face = "bold", size = 24, hjust = 0.5),
    plot.subtitle = element_text(size = 18, hjust = 0.5, colour = "grey30"),
    axis.title    = element_text(size = 22),
    axis.text     = element_text(size = 19),
    legend.text   = element_text(size = 17),
    legend.title  = element_text(size = 18, face = "bold"),
    strip.text    = element_text(size = 19, face = "bold"),
    strip.background = element_rect(fill = "grey92", colour = NA)
  )
theme_set(base_theme)

# ---------------------------------------------------------------------------
# GENERIC HELPERS
# ---------------------------------------------------------------------------
extract_id <- function(x) { m <- str_match(as.character(x), "((?:UM|U)[0-9]+)"); m[, 2] }

sig_label <- function(p) {
  factor(ifelse(is.na(p), NA_character_,
         ifelse(p < 0.05, "significant",
         ifelse(p < 0.10, "suggestive", "ns"))), levels = SIG_LEVELS)
}

fmt_p  <- function(p) ifelse(is.na(p), "--", ifelse(p < 0.001, "<0.001", sprintf("%.3f", p)))
fmt_ci <- function(hr, lo, hi) ifelse(is.na(hr), "--", sprintf("%.2f (%.2f-%.2f)", hr, lo, hi))

save_ggplot <- function(p, path, width = 12, height = 8) {
  ggsave(paste0(path, ".png"), p, width = width, height = height, dpi = 300, bg = "white")
  ggsave(paste0(path, ".pdf"), p, width = width, height = height, bg = "white", device = cairo_pdf)
  ggsave(paste0(path, ".svg"), p, width = width, height = height, bg = "white")
  cat("  ->", basename(path), ".png/pdf/svg\n")
}

save_survplot <- function(p, path, width = 12, height = 10) {
  png(paste0(path, ".png"), width = width, height = height, units = "in", res = 300); print(p); invisible(dev.off())
  pdf(paste0(path, ".pdf"), width = width, height = height);                          print(p); invisible(dev.off())
  svglite::svglite(paste0(path, ".svg"), width = width, height = height);             print(p); invisible(dev.off())
  cat("  ->", basename(path), ".png/pdf/svg\n")
}

#' Save a base-graphics proportional-hazards diagnostic to PNG/PDF/SVG.
#' `expr` is re-evaluated once per device.
save_baseplot <- function(expr, path, width = 14, height = 10) {
  ex <- substitute(expr); env <- parent.frame()
  png(paste0(path, ".png"), width = width, height = height, units = "in", res = 300); eval(ex, env); invisible(dev.off())
  pdf(paste0(path, ".pdf"), width = width, height = height);                          eval(ex, env); invisible(dev.off())
  svglite::svglite(paste0(path, ".svg"), width = width, height = height);             eval(ex, env); invisible(dev.off())
  cat("  ->", basename(path), ".png/pdf/svg\n")
}

# ============================================================================
# DATA -- ONE SOURCE FOR BOTH ENDPOINTS
# ============================================================================
# CHANGED 2026-07-19, and this fixes a genuine defect, not just a tidy-up:
# earlier revisions read OS from clinical_data.tsv and RFS from
# Supplementary_Table_01.xlsx. clinical_data.tsv has no "Pre-Treated" column,
# so the OS model silently omitted pre-treatment while the RFS model adjusted
# for it -- a difference produced by which FILE was read, not by any
# scientific decision. Both endpoints now come from the same xlsx, so any
# remaining difference between the OS and RFS models is deliberate and
# defensible (see the covariate rationale at Section 1.0).
# ============================================================================
cat("Loading locked NMF k=4 cluster labels ...\n")
clus <- read.csv(CLUSTER_CSV, stringsAsFactors = FALSE) %>%
  mutate(sid = extract_id(sample_id)) %>%
  filter(!is.na(sid)) %>%
  distinct(sid, .keep_all = TRUE) %>%
  transmute(sid, cluster = factor(cluster_display, levels = c("C1", "C2", "C3", "C4")))

expected_counts <- c(C1 = 426L, C2 = 274L, C3 = 268L, C4 = 94L)
observed_counts <- as.integer(table(clus$cluster)[names(expected_counts)])
if (!identical(observed_counts, unname(expected_counts))) {
  stop("Locked cluster counts do not match expected NMF k=4 solution: ",
       paste(names(expected_counts), observed_counts, sep = "=", collapse = ", "))
}
cat("  Locked cluster counts OK:", paste(names(expected_counts), expected_counts, sep = "=", collapse = ", "), "\n")

STAGE_LABELS <- c("Stage I", "Stage II", "Stage III", "Stage IV")

supp <- suppressWarnings(read_excel(SUPP_XLSX)) %>%
  mutate(sid = extract_id(`Sample ID`)) %>%
  filter(!is.na(sid)) %>%
  distinct(sid, .keep_all = TRUE) %>%
  transmute(
    sid,
    age        = suppressWarnings(as.numeric(`Age at diagnosis`)),
    sex        = factor(ifelse(Sex == "Male", "Male", "Female"), levels = c("Female", "Male")),
    stage_num  = recode(`Tumour Stage`, "Stage I" = 1, "Stage II" = 2,
                        "Stage III" = 3, "Stage IV" = 4, .default = NA_real_),
    msi        = factor(ifelse(`MSI Status` == "MSI", "MSI-H", "MSS"), levels = c("MSS", "MSI-H")),
    pretreated = factor(ifelse(`Pre-Treated` == "Treated", "Treated", "Untreated"),
                        levels = c("Untreated", "Treated")),
    # "N/A" vital status / recurrence are treated as MISSING, not as
    # alive / not-recurred -- coding them 0 would silently invent censored
    # observations out of records that have no follow-up information at all.
    OS_event   = ifelse(`Vital Status` == "Dead", 1, ifelse(`Vital Status` == "Alive", 0, NA_real_)),
    OS_months  = suppressWarnings(as.numeric(`Overall survival days`)) / 30.44,
    RFS_event  = ifelse(Recurrence == "Yes", 1, ifelse(Recurrence == "No", 0, NA_real_)),
    RFS_months = suppressWarnings(as.numeric(`Recurrence free survival days`)) / 30.44
  ) %>%
  mutate(stage = factor(STAGE_LABELS[stage_num], levels = STAGE_LABELS)) %>%
  inner_join(clus, by = "sid")

# ============================================================================
# SHARED MACHINERY
# ============================================================================

#' Harrell's C-index (concordance) and SE, straight from summary.coxph().
c_index <- function(fit) {
  cc <- summary(fit)$concordance
  list(C = unname(cc[1]), se = unname(cc[2]))
}

tidy_cox_full <- function(fit) {
  broom::tidy(fit, exponentiate = TRUE, conf.int = TRUE) %>%
    transmute(term, HR = estimate, HR_lower = conf.low, HR_upper = conf.high, p = p.value)
}

#' All C(4,2)=6 pairwise cluster contrasts from ONE fitted model, via
#' multcomp::glht(fit, mcp(cluster="Tukey")) (Hothorn, Bretz & Westfall 2008,
#' Biometrical Journal 50:346-363). Raw p-values requested via
#' adjusted("none") so this script's own BH-FDR step applies afterwards;
#' glht's default single-step adjustment is a different, stricter
#' simultaneous method that would double-correct if layered underneath.
pairwise_cox_glht <- function(fit, model_label, n, n_events) {
  g <- tryCatch(multcomp::glht(fit, linfct = multcomp::mcp(cluster = "Tukey")),
                error = function(e) e)
  if (!inherits(g, "glht")) {
    return(tibble(model = model_label, group_a = "ALL", group_b = "ALL",
                  HR = NA_real_, HR_lower = NA_real_, HR_upper = NA_real_, p = NA_real_,
                  n = n, n_events = n_events, status = paste0("failed: ", conditionMessage(g))))
  }
  s  <- summary(g, test = multcomp::adjusted("none"))
  ci <- confint(g, calpha = multcomp::univariate_calpha())
  parts <- str_split_fixed(names(s$test$coefficients), " - ", 2)   # glht names e.g. "C2 - C1"
  tibble(model = model_label, group_a = parts[, 2], group_b = parts[, 1],
         HR = exp(unname(s$test$coefficients)),
         HR_lower = exp(unname(ci$confint[, "lwr"])), HR_upper = exp(unname(ci$confint[, "upr"])),
         p = unname(s$test$pvalues), n = n, n_events = n_events, status = "ok")
}

#' Kaplan-Meier with risk table and large fonts. survfit()/survdiff() calls are
#' built as literal text and eval(parse())-ed because ggsurvplot() inspects
#' sf$call$formula; a formula passed via a variable leaves only an unevaluated
#' symbol there and crashes ggsurvplot ("object of type 'symbol' is not
#' subsettable"). Documented survminer gotcha, not an analysis issue.
km_plot <- function(df, time_col, event_col, title, xlab, unadj_hr = NULL) {
  d <- df %>% filter(!is.na(.data[[time_col]]), !is.na(.data[[event_col]]), !is.na(cluster))
  fml_text <- sprintf("Surv(%s, %s) ~ cluster", time_col, event_col)
  sf <- eval(parse(text = sprintf("survfit(%s, data = d)", fml_text)))
  lr <- eval(parse(text = sprintf("survdiff(%s, data = d)", fml_text)))
  p_global <- 1 - pchisq(lr$chisq, length(lr$n) - 1)

  meds <- summary(sf)$table[, "median"]; names(meds) <- levels(d$cluster)
  ns <- table(d$cluster)
  legend_labs <- vapply(levels(d$cluster), function(cl) {
    med_str <- if (is.finite(meds[cl])) sprintf(", med=%.0f mo", meds[cl]) else ""
    hr_str <- ""
    if (!is.null(unadj_hr)) {
      if (cl == "C1") hr_str <- ", HR=ref" else {
        r <- unadj_hr %>% filter(group_b == cl, group_a == "C1")
        if (nrow(r) && !is.na(r$HR[1])) hr_str <- sprintf(", HR=%.2f", r$HR[1])
      }
    }
    sprintf("%s (n=%d%s%s)", cl, ns[cl], med_str, hr_str)
  }, character(1))

  p <- ggsurvplot(
    sf, data = d, palette = unname(CB4[levels(d$cluster)]),
    legend.labs = legend_labs, legend.title = "", legend = c(0.76, 0.84),
    xlab = xlab, ylab = "Survival probability", title = title,
    subtitle = sprintf("Log-rank p = %.4g%s", p_global,
                       if (!is.null(unadj_hr)) "   |   HR unadjusted, C1 = reference" else ""),
    conf.int = TRUE, conf.int.alpha = 0.15,
    risk.table = TRUE, risk.table.height = 0.25, risk.table.y.text = FALSE,
    risk.table.title = "Number at risk", risk.table.fontsize = 6,
    size = 1.5, censor.shape = "|", censor.size = 4,
    ggtheme = base_theme, font.legend = 17, font.x = 22, font.y = 22,
    font.tickslab = 19, font.title = c(24, "bold"), font.subtitle = 18,
    tables.theme = theme_cleantable(font.main = c(18, "plain"))
  )
  list(plot = p, p_global = p_global, n = nrow(d), n_events = sum(d[[event_col]]))
}

#' Forest plot direct from survminer::ggforest() on the fitted coxph object.
#' ggforest() prints "# Events / Global p-value (Log-Rank) / AIC / Concordance
#' Index" in its own footer, so the C-index appears on the figure natively.
#' NOTE 1: ggforest() does data[, var] expecting base-data.frame drop-to-vector
#'   semantics; tibbles do not drop, which renames table()'s columns and breaks
#'   ggforest's internal rbind ("names do not match previous names"). Hence
#'   as.data.frame() here.
#' NOTE 2: ggforest() cannot render a model containing strata() -- see the
#'   DISPLAY-MODEL rationale in Section 1.2.
#' NOTE 3: ggforest() draws its footer ("# Events / Global p-value / AIC /
#'   Concordance Index") BELOW the panel with clipping switched off, so the
#'   second footer line -- the one carrying the C-index -- is cut off at the
#'   device edge unless extra bottom margin is supplied. Confirmed by
#'   reproduction: enlarging height alone does not help (the footer is placed
#'   proportionally), but adding plot.margin at the bottom does. Since the
#'   C-index is explicitly wanted on these figures, that margin is added here.
ggforest_plot <- function(fit, data, title, fontsize = 1.0, bottom_margin = 110) {
  survminer::ggforest(fit, data = as.data.frame(data),
                      main = str_wrap(title, width = 62), fontsize = fontsize) +
    theme(plot.margin = margin(t = 6, r = 6, b = bottom_margin, l = 6))
}

#' Rename model variables to manuscript-presentable labels for the DISPLAY
#' model only (ggforest prints the raw variable names, and
#' "cluster"/"msi"/"pretreated" are not publication labels). Names are kept
#' syntactically simple -- no spaces or parentheses -- because ggforest strips
#' backticks internally when matching coefficient names to terms, which breaks
#' label lookup for backtick-quoted variables.
display_data <- function(d) {
  d %>% dplyr::rename(Subtype = cluster, Age = age, Sex = sex,
                      MSI = msi, Pretreatment = pretreated, Stage = stage)
}
DISPLAY_RHS <- "Subtype + Age + Sex + MSI + Pretreatment + Stage"

#' Schoenfeld-residual PH test as a tidy table (survival::cox.zph()).
ph_check <- function(fit, label) {
  as.data.frame(cox.zph(fit)$table) %>%
    rownames_to_column("term") %>% mutate(model = label) %>%
    relocate(model, .before = term) %>% mutate(sig = sig_label(p))
}

#' Is a penalized (ridge) Cox model warranted? Compare the unpenalized fit
#' against a ridge refit on the continuous covariate, alongside the
#' events-per-variable ratio. Reported as a table so the answer is evidence,
#' not assertion. Peduzzi et al. 1996 (J Clin Epidemiol 49:1373-1379) put the
#' usual adequacy threshold at EPV >= 10.
penalizer_check <- function(fit, data, endpoint, ridge_terms, strata_term) {
  n_events <- fit$nevent
  n_params <- length(coef(fit))
  epv <- n_events / n_params
  lhs <- deparse1(formula(fit)[[2]])
  other <- setdiff(c("cluster", "age", "sex", "msi", "pretreated"), ridge_terms)
  fml_txt <- sprintf("%s ~ ridge(%s, theta = 1) + %s + %s",
                     lhs, paste(ridge_terms, collapse = ", "),
                     paste(other, collapse = " + "), strata_term)
  fit_r <- tryCatch(coxph(as.formula(fml_txt), data = as.data.frame(data), ties = "efron"),
                    error = function(e) NULL)
  co_u <- coef(fit); c_u <- c_index(fit)$C
  if (is.null(fit_r)) {
    return(tibble(endpoint = endpoint, n_events = n_events, n_parameters = n_params, EPV = epv,
                  max_abs_coef_shift_vs_ridge = NA_real_, C_unpenalized = c_u, C_ridge = NA_real_,
                  penalizer_warranted = ifelse(epv >= 10, "no", "yes"),
                  rationale = "ridge refit unavailable; decision from EPV alone"))
  }
  co_r <- coef(fit_r)[names(co_u)]
  tibble(endpoint = endpoint, n_events = n_events, n_parameters = n_params, EPV = epv,
         max_abs_coef_shift_vs_ridge = max(abs(co_u - co_r), na.rm = TRUE),
         C_unpenalized = c_u, C_ridge = c_index(fit_r)$C,
         penalizer_warranted = ifelse(epv >= 10, "no", "yes"),
         rationale = ifelse(epv >= 10,
           sprintf("EPV %.1f >= 10 (Peduzzi 1996); ridge changes coefficients negligibly", epv),
           sprintf("EPV %.1f < 10; shrinkage advisable", epv)))
}

TERM_LABELS <- c(
  "clusterC2"         = "Subtype C2 vs C1",
  "clusterC3"         = "Subtype C3 vs C1",
  "clusterC4"         = "Subtype C4 vs C1",
  "age"               = "Age (per year)",
  "sexMale"           = "Sex: male vs female",
  "msiMSI-H"          = "MSI-H vs MSS",
  "pretreatedTreated" = "Pre-treated vs untreated",
  "stageStage II"     = "Stage II vs I",
  "stageStage III"    = "Stage III vs I",
  "stageStage IV"     = "Stage IV vs I"
)
label_term <- function(x) ifelse(x %in% names(TERM_LABELS), TERM_LABELS[x], x)

manuscript_model_table <- function(tidy_tab, endpoint, model_label, n, n_events, cidx) {
  tidy_tab %>%
    transmute(Endpoint = endpoint, Model = model_label,
              Variable = label_term(term),
              `HR (95% CI)` = fmt_ci(HR, HR_lower, HR_upper),
              `P value` = fmt_p(p),
              Significance = as.character(sig_label(p)),
              N = n, Events = n_events, `C-index` = sprintf("%.3f", cidx))
}

cat("\n############ SECTION 1: OVERALL SURVIVAL (PRIMARY ENDPOINT) ############\n")

# ---------------------------------------------------------------------------
# 1.0 OS cohort and covariate rationale
#
# Cohort: ALL stages (I-IV). OS is an all-cause-mortality endpoint, so
# restricting it would discard exactly the patients (Stage IV) whose mortality
# it is meant to capture.
#
# Covariates: age, sex, MSI status, pre-treatment; STAGE IS STRATIFIED rather
# than entered as a coefficient (empirically justified in 1.5 -- its Schoenfeld
# test is grossly violated as an ordinary covariate).
#
# ON "WHY ARE THE OS AND RFS COVARIATES THE SAME?" -- they are deliberately
# the same LIST, and that is correct: age, sex, stage and MSI are the
# established clinical prognostic factors for BOTH endpoints in colorectal
# cancer, and pre-treatment is a treatment-allocation confounder for both.
# Using a different adjustment set per endpoint without a reason would make
# the two models non-comparable, which is the opposite of what is wanted when
# RFS is being used to corroborate OS. What legitimately DOES differ, and now
# does so by design rather than by accident, is:
#   (i)   the COHORT -- all stages (n=1062) for OS, Stage I-III (n=948) for
#         RFS, because Stage IV patients never occupy a disease-free state and
#         cannot contribute a recurrence event;
#   (ii)  the STAGE STRATA -- 4 levels for OS, 3 for RFS, following (i);
#   (iii) the EVENT itself -- death from any cause vs cancer recurrence, which
#         is the substantive reason both are reported rather than just one.
# The earlier genuine defect (OS silently missing pre-treatment because
# clinical_data.tsv lacks that column) is fixed at the data-loading step above.
# ---------------------------------------------------------------------------
os_df <- supp %>%
  filter(!is.na(OS_months), !is.na(OS_event), OS_months >= 0, !is.na(stage)) %>%
  droplevels()
cat("  OS cohort (all stages): n =", nrow(os_df), " events =", sum(os_df$OS_event), "\n")

OS_COVS <- c("age", "sex", "msi", "pretreated")

# ---------------------------------------------------------------------------
# 1.1 Adjusted (stage-stratified) Cox -- the PRIMARY INFERENTIAL MODEL.
#     An unadjusted cluster-only fit is also computed, but only to annotate
#     the KM legend and populate the tables; per instruction no unadjusted
#     forest plot is produced.
# ---------------------------------------------------------------------------
os_fit_unadj <- coxph(Surv(OS_months, OS_event) ~ cluster, data = os_df, ties = "efron")
os_fit_adj   <- coxph(Surv(OS_months, OS_event) ~ cluster + age + sex + msi + pretreated +
                        strata(stage), data = os_df, ties = "efron")

os_c_unadj <- c_index(os_fit_unadj); os_c_adj <- c_index(os_fit_adj)
cat(sprintf("  OS adjusted (stage-stratified): C-index = %.3f (SE %.3f)\n", os_c_adj$C, os_c_adj$se))

os_tidy_unadj <- tidy_cox_full(os_fit_unadj)
os_tidy_adj   <- tidy_cox_full(os_fit_adj)
cat(sprintf("  OS adjusted C4 vs C1: HR = %.3f, p = %.4g\n",
            os_tidy_adj$HR[os_tidy_adj$term == "clusterC4"],
            os_tidy_adj$p[os_tidy_adj$term == "clusterC4"]))

os_cox_summary <- bind_rows(
  os_tidy_unadj %>% mutate(model = "unadjusted (subtype only)"),
  os_tidy_adj   %>% mutate(model = "adjusted (age+sex+MSI+pre-treatment, strata=stage)")
) %>% mutate(sig = sig_label(p), n = nrow(os_df), n_events = sum(os_df$OS_event)) %>%
  relocate(model, .before = term)
write.csv(os_cox_summary, file.path(TABDIR, "OS_cox_summary.csv"), row.names = FALSE)

# ---------------------------------------------------------------------------
# 1.2 ggforest -- requires a DISPLAY MODEL with stage as an ordinary covariate.
#
# survminer::ggforest() cannot render a coxph model containing strata()
# (errors "undefined columns selected"; reproduced and confirmed -- a known
# survminer limitation with no argument that works around it). Therefore, a
# clearly labelled display model is fitted with stage as an ordinary
# covariate. This is legitimate for displaying the SUBTYPE effect because
# subtype itself satisfies PH comfortably in both specifications (see 1.5);
# what must NOT be over-interpreted from this model is stage's OWN
# coefficient, which is precisely the term whose PH assumption fails. Both
# models are reported side by side so the near-identical subtype HRs are
# visible.
# ---------------------------------------------------------------------------
os_display_df <- display_data(os_df)
os_fit_display <- coxph(as.formula(sprintf("Surv(OS_months, OS_event) ~ %s", DISPLAY_RHS)),
                        data = os_display_df, ties = "efron")
os_c_display <- c_index(os_fit_display)
os_tidy_display <- tidy_cox_full(os_fit_display)
cat(sprintf("  OS display model (stage as covariate): C-index = %.3f | C4 HR = %.3f (stratified model: %.3f)\n",
            os_c_display$C, os_tidy_display$HR[os_tidy_display$term == "SubtypeC4"],
            os_tidy_adj$HR[os_tidy_adj$term == "clusterC4"]))

save_ggplot(ggforest_plot(os_fit_display, os_display_df,
                          "Overall survival: multivariable Cox (C-index in footer)",
                          fontsize = 1.0),
            file.path(FIGDIR, "OS_forest_adjusted"), width = 14, height = 12)

# ---------------------------------------------------------------------------
# 1.3 Pairwise subtype contrasts (all 6), BH-FDR within each model's family
# ---------------------------------------------------------------------------
os_pw <- bind_rows(
  pairwise_cox_glht(os_fit_unadj, "unadjusted", nrow(os_df), sum(os_df$OS_event)),
  pairwise_cox_glht(os_fit_adj,   "adjusted",   nrow(os_df), sum(os_df$OS_event))
) %>% group_by(model) %>% mutate(padj = p.adjust(p, method = "BH")) %>% ungroup() %>%
  mutate(sig = sig_label(padj))
write.csv(os_pw, file.path(TABDIR, "OS_pairwise_cox.csv"), row.names = FALSE)
cat("  OS pairwise contrasts:", nrow(os_pw), "rows (expect 12)\n")

# ---------------------------------------------------------------------------
# 1.4 Kaplan-Meier
# ---------------------------------------------------------------------------
km_os <- km_plot(os_df, "OS_months", "OS_event",
                 "Overall survival by genomic subtype (all stages)",
                 "Overall survival (months)",
                 unadj_hr = os_pw %>% filter(model == "unadjusted", group_a == "C1"))
save_survplot(km_os$plot, file.path(FIGDIR, "OS_KM"))

# ---------------------------------------------------------------------------
# 1.5 Proportional-hazards diagnostics + INBUILT Schoenfeld plots
# ---------------------------------------------------------------------------
os_ph <- bind_rows(
  ph_check(os_fit_display, "OS display model (stage as ordinary covariate)"),
  ph_check(os_fit_adj,     "OS primary model (stage stratified, as used)")
)
write.csv(os_ph, file.path(TABDIR, "OS_PH_assumption_check.csv"), row.names = FALSE)
os_stage_ph <- min(os_ph$p[str_starts(os_ph$term, "stage") & str_detect(os_ph$model, "display")])
cat(sprintf("  OS PH: stage-as-covariate Schoenfeld p = %.3g -> stratification justified\n", os_stage_ph))
cat(sprintf("  OS PH (primary, stratified): cluster p = %.3g | GLOBAL p = %.3g\n",
            os_ph$p[str_detect(os_ph$model, "primary") & os_ph$term == "cluster"],
            os_ph$p[str_detect(os_ph$model, "primary") & os_ph$term == "GLOBAL"]))

os_zph <- cox.zph(os_fit_adj)
save_baseplot({
  par(mfrow = c(2, 3), cex.axis = 1.5, cex.lab = 1.6, cex.main = 1.7, mar = c(5.5, 5.5, 4, 2))
  plot(os_zph)
}, file.path(FIGDIR, "OS_PH_schoenfeld_plots"), width = 18, height = 11)

# ---------------------------------------------------------------------------
# 1.6 Is a Cox penalizer warranted?
# ---------------------------------------------------------------------------
os_pen <- penalizer_check(os_fit_adj, os_df, "OS", c("age"), "strata(stage)")
cat(sprintf("  OS penalizer: %d events / %d params = EPV %.1f -> warranted: %s\n",
            os_pen$n_events, os_pen$n_parameters, os_pen$EPV, os_pen$penalizer_warranted))

# ---------------------------------------------------------------------------
# 1.7 Likelihood-ratio test (stats::anova on nested coxph fits;
#     Royston & Altman 2013, BMC Med Res Methodol 13:33)
# ---------------------------------------------------------------------------
os_common <- os_df[stats::complete.cases(os_df[, c("OS_months","OS_event","cluster", OS_COVS, "stage")]), ]
os_A <- coxph(Surv(OS_months, OS_event) ~ age + sex + msi + pretreated + strata(stage), data = os_common, ties = "efron")
os_B <- coxph(Surv(OS_months, OS_event) ~ age + sex + msi + pretreated + strata(stage) + cluster, data = os_common, ties = "efron")
os_lrt_raw <- anova(os_A, os_B, test = "Chisq")
os_lrt <- tibble(
  endpoint = "OS",
  model_A = "clinical covariates only (age+sex+MSI+pre-treatment, strata=stage)",
  model_B = "clinical covariates + subtype",
  logLik_A = as.numeric(logLik(os_A)), logLik_B = as.numeric(logLik(os_B)),
  LRT_chisq = os_lrt_raw$Chisq[2], df = os_lrt_raw$Df[2], LRT_p = os_lrt_raw[2, "Pr(>|Chi|)"],
  C_index_A = c_index(os_A)$C, C_index_B = c_index(os_B)$C,
  n = nrow(os_common), n_events = sum(os_common$OS_event)) %>%
  mutate(sig = sig_label(LRT_p))
write.csv(os_lrt, file.path(TABDIR, "OS_LRT_incremental_value.csv"), row.names = FALSE)
cat(sprintf("  OS LRT: chisq = %.2f, df = %d, p = %.4g | C-index %.3f -> %.3f\n",
            os_lrt$LRT_chisq, os_lrt$df, os_lrt$LRT_p, os_lrt$C_index_A, os_lrt$C_index_B))

cat("\n############ SECTION 2: RECURRENCE-FREE SURVIVAL (SUPPORTING) ############\n")
cat("  RFS is reported as a SUPPORTING endpoint: it asks whether the OS signal\n")
cat("  is internally consistent on a second, cancer-specific endpoint in the\n")
cat("  curative-intent (Stage I-III) subset. It is not the primary claim.\n")

rfs_df <- supp %>%
  filter(stage_num %in% c(1, 2, 3), !is.na(RFS_months), !is.na(RFS_event), RFS_months >= 0) %>%
  droplevels()
cat("  RFS cohort (Stage I-III): n =", nrow(rfs_df), " events =", sum(rfs_df$RFS_event), "\n")

rfs_fit_unadj   <- coxph(Surv(RFS_months, RFS_event) ~ cluster, data = rfs_df, ties = "efron")
rfs_fit_adj     <- coxph(Surv(RFS_months, RFS_event) ~ cluster + age + sex + msi + pretreated +
                           strata(stage), data = rfs_df, ties = "efron")
rfs_display_df  <- display_data(rfs_df)
rfs_fit_display <- coxph(as.formula(sprintf("Surv(RFS_months, RFS_event) ~ %s", DISPLAY_RHS)),
                         data = rfs_display_df, ties = "efron")

rfs_c_adj <- c_index(rfs_fit_adj); rfs_c_display <- c_index(rfs_fit_display)
rfs_tidy_unadj   <- tidy_cox_full(rfs_fit_unadj)
rfs_tidy_adj     <- tidy_cox_full(rfs_fit_adj)
rfs_tidy_display <- tidy_cox_full(rfs_fit_display)
cat(sprintf("  RFS adjusted (stage-stratified): C-index = %.3f | C4 vs C1 HR = %.3f, p = %.4g\n",
            rfs_c_adj$C, rfs_tidy_adj$HR[rfs_tidy_adj$term == "clusterC4"],
            rfs_tidy_adj$p[rfs_tidy_adj$term == "clusterC4"]))

rfs_cox_summary <- bind_rows(
  rfs_tidy_unadj %>% mutate(model = "unadjusted (subtype only)"),
  rfs_tidy_adj   %>% mutate(model = "adjusted (age+sex+MSI+pre-treatment, strata=stage)")
) %>% mutate(sig = sig_label(p), n = nrow(rfs_df), n_events = sum(rfs_df$RFS_event)) %>%
  relocate(model, .before = term)
write.csv(rfs_cox_summary, file.path(TABDIR, "RFS_cox_summary.csv"), row.names = FALSE)

save_ggplot(ggforest_plot(rfs_fit_display, rfs_display_df,
                          "Recurrence-free survival: multivariable Cox (C-index in footer)",
                          fontsize = 1.0),
            file.path(FIGDIR, "RFS_forest_adjusted"), width = 14, height = 12)

rfs_pw <- bind_rows(
  pairwise_cox_glht(rfs_fit_unadj, "unadjusted", nrow(rfs_df), sum(rfs_df$RFS_event)),
  pairwise_cox_glht(rfs_fit_adj,   "adjusted",   nrow(rfs_df), sum(rfs_df$RFS_event))
) %>% group_by(model) %>% mutate(padj = p.adjust(p, method = "BH")) %>% ungroup() %>%
  mutate(sig = sig_label(padj))
write.csv(rfs_pw, file.path(TABDIR, "RFS_pairwise_cox.csv"), row.names = FALSE)
cat("  RFS pairwise contrasts:", nrow(rfs_pw), "rows (expect 12)\n")

km_rfs <- km_plot(rfs_df, "RFS_months", "RFS_event",
                  "Recurrence-free survival by genomic subtype (Stage I-III)",
                  "Recurrence-free survival (months)",
                  unadj_hr = rfs_pw %>% filter(model == "unadjusted", group_a == "C1"))
save_survplot(km_rfs$plot, file.path(FIGDIR, "RFS_KM"))

rfs_ph <- bind_rows(
  ph_check(rfs_fit_display, "RFS display model (stage as ordinary covariate)"),
  ph_check(rfs_fit_adj,     "RFS primary model (stage stratified, as used)")
)
write.csv(rfs_ph, file.path(TABDIR, "RFS_PH_assumption_check.csv"), row.names = FALSE)
rfs_stage_ph <- min(rfs_ph$p[str_starts(rfs_ph$term, "stage") & str_detect(rfs_ph$model, "display")])
cat(sprintf("  RFS PH: stage-as-covariate Schoenfeld p = %.3g | stratified GLOBAL p = %.3g\n",
            rfs_stage_ph, rfs_ph$p[str_detect(rfs_ph$model, "primary") & rfs_ph$term == "GLOBAL"]))

rfs_zph <- cox.zph(rfs_fit_adj)
save_baseplot({
  par(mfrow = c(2, 3), cex.axis = 1.5, cex.lab = 1.6, cex.main = 1.7, mar = c(5.5, 5.5, 4, 2))
  plot(rfs_zph)
}, file.path(FIGDIR, "RFS_PH_schoenfeld_plots"), width = 18, height = 11)

rfs_pen <- penalizer_check(rfs_fit_adj, rfs_df, "RFS", c("age"), "strata(stage)")
cat(sprintf("  RFS penalizer: %d events / %d params = EPV %.1f -> warranted: %s\n",
            rfs_pen$n_events, rfs_pen$n_parameters, rfs_pen$EPV, rfs_pen$penalizer_warranted))
write.csv(bind_rows(os_pen, rfs_pen), file.path(TABDIR, "Cox_penalizer_assessment.csv"), row.names = FALSE)

rfs_common <- rfs_df[stats::complete.cases(rfs_df[, c("RFS_months","RFS_event","cluster", OS_COVS, "stage")]), ]
rfs_A <- coxph(Surv(RFS_months, RFS_event) ~ age + sex + msi + pretreated + strata(stage), data = rfs_common, ties = "efron")
rfs_B <- coxph(Surv(RFS_months, RFS_event) ~ age + sex + msi + pretreated + strata(stage) + cluster, data = rfs_common, ties = "efron")
rfs_lrt_raw <- anova(rfs_A, rfs_B, test = "Chisq")
rfs_lrt <- tibble(
  endpoint = "RFS",
  model_A = "clinical covariates only (age+sex+MSI+pre-treatment, strata=stage)",
  model_B = "clinical covariates + subtype",
  logLik_A = as.numeric(logLik(rfs_A)), logLik_B = as.numeric(logLik(rfs_B)),
  LRT_chisq = rfs_lrt_raw$Chisq[2], df = rfs_lrt_raw$Df[2], LRT_p = rfs_lrt_raw[2, "Pr(>|Chi|)"],
  C_index_A = c_index(rfs_A)$C, C_index_B = c_index(rfs_B)$C,
  n = nrow(rfs_common), n_events = sum(rfs_common$RFS_event)) %>%
  mutate(sig = sig_label(LRT_p))
write.csv(rfs_lrt, file.path(TABDIR, "RFS_LRT_incremental_value.csv"), row.names = FALSE)
cat(sprintf("  RFS LRT: chisq = %.2f, df = %d, p = %.4g | C-index %.3f -> %.3f\n",
            rfs_lrt$LRT_chisq, rfs_lrt$df, rfs_lrt$LRT_p, rfs_lrt$C_index_A, rfs_lrt$C_index_B))

cat("\n############ SECTION 4: CLINICAL-SUBGROUP SENSITIVITY (SUPPLEMENTARY) ############\n")
# ---------------------------------------------------------------------------
# IS THIS ANALYSIS WORTH KEEPING? Yes -- but strictly as supplementary, and it
# is trimmed here to one table plus one figure (earlier revisions also
# reproduced full KM/forest/pairwise output for the primary scenario, which
# merely duplicated Sections 1-2).
#
# It answers something none of the other analyses do: whether the subtype
# signal is an artifact of confounding by the three clinical features most
# likely to drive it in colorectal cancer -- neoadjuvant pre-treatment, MSI
# status, and stage. Adjustment (Sections 1-2) handles this on average across
# the cohort; RESTRICTION handles it structurally, by removing the confounder
# instead of modelling it. If C4's elevated risk held only in the pooled cohort
# and vanished in every restricted subset, that would materially undermine the
# prognostic interpretation. Reporting it costs one supplementary table and
# figure and pre-empts an obvious reviewer question.
# ---------------------------------------------------------------------------
scenarios <- list(
  "All patients"           = list(os = os_df, rfs = rfs_df),
  "Pre-treatment excluded" = list(os = os_df  %>% filter(pretreated == "Untreated"),
                                  rfs = rfs_df %>% filter(pretreated == "Untreated")),
  "MSS only"               = list(os = os_df  %>% filter(msi == "MSS"),
                                  rfs = rfs_df %>% filter(msi == "MSS")),
  "Stage II MSS"           = list(os = os_df  %>% filter(stage == "Stage II", msi == "MSS"),
                                  rfs = rfs_df %>% filter(stage == "Stage II", msi == "MSS")),
  "Stage III MSS"          = list(os = os_df  %>% filter(stage == "Stage III", msi == "MSS"),
                                  rfs = rfs_df %>% filter(stage == "Stage III", msi == "MSS"))
)

sens_rows <- list()
for (scen in names(scenarios)) {
  for (ep in c("OS", "RFS")) {
    d <- scenarios[[scen]][[tolower(ep)]] %>% droplevels()
    tcol <- if (ep == "OS") "OS_months" else "RFS_months"
    ecol <- if (ep == "OS") "OS_event"  else "RFS_event"
    nev <- sum(d[[ecol]])
    if (nrow(d) < 40 || nev < 10 || nlevels(d$cluster) < 2) {
      sens_rows[[length(sens_rows) + 1]] <- tibble(scenario = scen, endpoint = ep,
        HR = NA_real_, HR_lower = NA_real_, HR_upper = NA_real_, p = NA_real_,
        n = nrow(d), n_events = nev, status = "skipped_insufficient_events"); next
    }
    rhs <- c("cluster", "age", "sex")
    if (nlevels(d$msi) > 1)        rhs <- c(rhs, "msi")
    if (nlevels(d$pretreated) > 1) rhs <- c(rhs, "pretreated")
    strata_part <- if (nlevels(d$stage) > 1) " + strata(stage)" else ""
    f <- as.formula(sprintf("Surv(%s, %s) ~ %s%s", tcol, ecol, paste(rhs, collapse = " + "), strata_part))
    fit <- tryCatch(coxph(f, data = d, ties = "efron"), error = function(e) NULL)
    if (is.null(fit)) {
      sens_rows[[length(sens_rows) + 1]] <- tibble(scenario = scen, endpoint = ep,
        HR = NA_real_, HR_lower = NA_real_, HR_upper = NA_real_, p = NA_real_,
        n = nrow(d), n_events = nev, status = "model_failed"); next
    }
    r <- tidy_cox_full(fit) %>% filter(term == "clusterC4")
    sens_rows[[length(sens_rows) + 1]] <- tibble(scenario = scen, endpoint = ep,
      HR = r$HR[1], HR_lower = r$HR_lower[1], HR_upper = r$HR_upper[1], p = r$p[1],
      n = nrow(d), n_events = nev, status = "ok")
  }
}
sens <- bind_rows(sens_rows) %>%
  group_by(endpoint) %>% mutate(padj = p.adjust(p, method = "BH")) %>% ungroup() %>%
  mutate(sig = sig_label(padj))
write.csv(sens, file.path(TABDIR, "SENS_scenario_summary_C4_vs_C1.csv"), row.names = FALSE)
cat("  Sensitivity scenarios:", nrow(sens), "rows (5 restrictions x 2 endpoints), adjusted models only\n")

sens_plot <- sens %>% filter(!is.na(HR)) %>%
  mutate(scenario = factor(scenario, levels = rev(names(scenarios))),
         endpoint = factor(endpoint, levels = c("OS", "RFS")))
p_sens <- ggplot(sens_plot, aes(x = HR, y = scenario, colour = sig)) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey40", linewidth = 0.8) +
  geom_errorbarh(aes(xmin = HR_lower, xmax = HR_upper), height = 0.2, linewidth = 1.1) +
  geom_point(size = 4.2) +
  geom_text(aes(x = HR_upper, label = sprintf("  HR=%.2f, p(adj)=%.3g", HR, padj)),
            hjust = 0, size = 5.6, colour = "grey15") +
  facet_wrap(~endpoint, ncol = 1) +
  scale_colour_manual(values = SIG_COLORS, breaks = SIG_LEVELS,
                      labels = c("significant (<0.05)", "suggestive (0.05-0.10)", "ns (>=0.10)"),
                      name = NULL, drop = FALSE) +
  scale_x_log10(expand = expansion(mult = c(0.06, 0.75))) +
  labs(title = "C4 vs C1 across clinical restrictions (adjusted)",
       subtitle = "Supplementary robustness check: restriction rather than adjustment",
       x = "Hazard ratio (log scale)", y = NULL) +
  theme(legend.position = "top", axis.text.y = element_text(size = 18))
save_ggplot(p_sens, file.path(FIGDIR, "SENS_scenario_forest"), width = 13, height = 9)

cat("\n############ SECTION 5: MANUSCRIPT TABLES AND SYNTHESIS ############\n")

manu <- bind_rows(
  manuscript_model_table(os_tidy_adj, "Overall survival", "Multivariable Cox (stage-stratified)",
                         nrow(os_df), sum(os_df$OS_event), os_c_adj$C),
  manuscript_model_table(rfs_tidy_adj, "Recurrence-free survival", "Multivariable Cox (stage-stratified)",
                         nrow(rfs_df), sum(rfs_df$RFS_event), rfs_c_adj$C)
)
write.csv(manu, file.path(TABDIR, "TABLE_multivariable_cox_manuscript.csv"), row.names = FALSE)

manu_pw <- bind_rows(os_pw %>% mutate(Endpoint = "Overall survival"),
                     rfs_pw %>% mutate(Endpoint = "Recurrence-free survival")) %>%
  filter(model == "adjusted") %>%
  transmute(Endpoint, Comparison = paste(group_b, "vs", group_a),
            `HR (95% CI)` = fmt_ci(HR, HR_lower, HR_upper),
            `P value` = fmt_p(p), `P (BH-adjusted)` = fmt_p(padj),
            Significance = as.character(sig), N = n, Events = n_events)
write.csv(manu_pw, file.path(TABDIR, "TABLE_pairwise_contrasts_manuscript.csv"), row.names = FALSE)

manu_lrt <- bind_rows(os_lrt, rfs_lrt) %>%
  transmute(Endpoint = ifelse(endpoint == "OS", "Overall survival", "Recurrence-free survival"),
            `Chi-square` = sprintf("%.2f", LRT_chisq), df,
            `P value` = fmt_p(LRT_p),
            `C-index without subtype` = sprintf("%.3f", C_index_A),
            `C-index with subtype` = sprintf("%.3f", C_index_B),
            N = n, Events = n_events, Significance = as.character(sig))
write.csv(manu_lrt, file.path(TABDIR, "TABLE_LRT_manuscript.csv"), row.names = FALSE)

cohort_tab <- bind_rows(
  os_df  %>% mutate(Cohort = sprintf("OS (all stages, n=%d)", nrow(os_df))),
  rfs_df %>% mutate(Cohort = sprintf("RFS (Stage I-III, n=%d)", nrow(rfs_df)))
) %>%
  group_by(Cohort, Subtype = cluster) %>%
  summarise(N = n(),
            `Age median (IQR)` = sprintf("%.0f (%.0f-%.0f)", median(age, na.rm = TRUE),
                                          quantile(age, .25, na.rm = TRUE), quantile(age, .75, na.rm = TRUE)),
            `Male n (%)` = sprintf("%d (%.0f%%)", sum(sex == "Male"), 100 * mean(sex == "Male")),
            `MSI-H n (%)` = sprintf("%d (%.0f%%)", sum(msi == "MSI-H"), 100 * mean(msi == "MSI-H")),
            `Pre-treated n (%)` = sprintf("%d (%.0f%%)", sum(pretreated == "Treated"),
                                           100 * mean(pretreated == "Treated")),
            .groups = "drop")
write.csv(cohort_tab, file.path(TABDIR, "TABLE_cohort_characteristics.csv"), row.names = FALSE)

os_c4    <- os_tidy_adj    %>% filter(term == "clusterC4")
rfs_c4   <- rfs_tidy_adj   %>% filter(term == "clusterC4")
os_c4_u  <- os_tidy_unadj  %>% filter(term == "clusterC4")
rfs_c4_u <- rfs_tidy_unadj %>% filter(term == "clusterC4")
os_pw_c4  <- os_pw  %>% filter(model == "adjusted", group_a == "C1", group_b == "C4")
rfs_pw_c4 <- rfs_pw %>% filter(model == "adjusted", group_a == "C1", group_b == "C4")

synthesis <- tibble(
  endpoint = c("Overall survival (primary)", "Recurrence-free survival (supporting)"),
  cohort = c(sprintf("All stages, n=%d, %d deaths", nrow(os_df), sum(os_df$OS_event)),
             sprintf("Stage I-III, n=%d, %d recurrences", nrow(rfs_df), sum(rfs_df$RFS_event))),
  HR_unadjusted = c(fmt_ci(os_c4_u$HR, os_c4_u$HR_lower, os_c4_u$HR_upper),
                    fmt_ci(rfs_c4_u$HR, rfs_c4_u$HR_lower, rfs_c4_u$HR_upper)),
  p_unadjusted = c(fmt_p(os_c4_u$p), fmt_p(rfs_c4_u$p)),
  HR_adjusted = c(fmt_ci(os_c4$HR, os_c4$HR_lower, os_c4$HR_upper),
                  fmt_ci(rfs_c4$HR, rfs_c4$HR_lower, rfs_c4$HR_upper)),
  p_adjusted = c(fmt_p(os_c4$p), fmt_p(rfs_c4$p)),
  p_pairwise_BH = c(fmt_p(os_pw_c4$padj), fmt_p(rfs_pw_c4$padj)),
  LRT_p = c(fmt_p(os_lrt$LRT_p), fmt_p(rfs_lrt$LRT_p)),
  C_index_adjusted = c(sprintf("%.3f", os_c_adj$C), sprintf("%.3f", rfs_c_adj$C)))
write.csv(synthesis, file.path(TABDIR, "TABLE_C4_prognostic_synthesis.csv"), row.names = FALSE)


writeLines(capture.output(sessionInfo()), file.path(TABDIR, "session_info.txt"))
cat("\n############ DONE ############\n")
cat("Outputs:", OUTDIR, "\n")
cat("  figures/:", length(list.files(FIGDIR)), "files\n")
cat("  tables/ :", length(list.files(TABDIR)), "files\n")
