# A Genome-First NMF Framework for Colorectal Cancer Subtyping
Authors: Avik Sengupta, Rahul Kumar* (*=correspondence)
Affiliation: Computational Genomics & Transcriptomics Lab, Indian Institute of Technology, Hyderabad, Kandi, Sangareddy, Telangana, India-502284

This repository contains the analysis code supporting a genome-first
classification of colorectal cancer (CRC) from recurrent somatic events.
Non-negative matrix factorisation (NMF) was applied to a binary matrix of
short variants, copy-number alterations and structural variants from 1,062
population-based CRCs. The selected four-component solution defines subtypes
C1-C4 and is subsequently evaluated by survival, genomic, transcriptomic,
immune, pathway, classifier-recovery, modality-dropout and cross-platform
portability analyses.

The repository is intentionally **code only**. Patient-level inputs, derived
matrices, subtype labels and analysis outputs are not redistributed.

## Study design

The workflow has four stages:

1. Construct recurrent binary SNV, CNV and SV feature matrices and merge them.
2. select and lock the four-component NMF solution using outcome-blind
   stability, null and resampling criteria;
3. characterize the resulting C1-C4 subtypes and evaluate their clinical and
   molecular associations; and
4. assess label recoverability, dependence on genomic modalities and
   cross-platform portability to TCGA COAD/READ.

The locked discovery partition contains 426 C1, 274 C2, 268 C3 and 94 C4
tumours. Script 05 regenerates this partition from the genomic input matrix;
downstream scripts do not contain hard-coded patient labels.

## Data availability

The discovery cohort was reported by Nunes *et al.* in *Nature* (2024).
Users must obtain the data under the terms of the source repositories:

- transcriptomic and associated study data: ArrayExpress/BioStudies
  [E-MTAB-12862](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-12862);
- genomic variant calls: European Nucleotide Archive/European Variation
  Archive project [PRJEB61514](https://www.ebi.ac.uk/ena/browser/view/PRJEB61514);
- TCGA COAD/READ portability data: cBioPortal study
  `coadread_tcga_pan_can_atlas_2018`, with the public UCSC Xena and cBioPortal
  sources requested by script 25.

The CRC gene panel used in scripts 01-03 must be reconstructed from the
IntOGen CRC driver set and the prespecified MSigDB CRC gene sets described in
the manuscript, then supplied as `colorectal_cancer_all_genes.txt`.

## Repository structure

```text
.
├── README.md
├── MANUSCRIPT_RESULT_MAP.md
├── SCRIPT_PROVENANCE.tsv
├── requirements.txt
├── R_PACKAGES.md
├── config.example.env
├── LICENSE
├── .gitignore
└── scripts/
    ├── 01_build_snv_features.py
    ├── ...
    └── 25_assess_tcga_portability.py
```

The numbered scripts are ordered by analytical dependency. Historical module
numbers and workstation-specific paths have been removed from the public
filenames. Output directory names are retained internally so that the
manuscript-generating analysis remains reproducible.

## Analysis workflow

| Step | Script | Purpose |
|---:|---|---|
| 01 | `01_build_snv_features.py` | Filter VEP-annotated somatic SNV/indel calls and construct recurrent functional gene-level features and burden tables. |
| 02 | `02_build_cnv_features.py` | Construct gene- and chromosome-arm-level binary CNV features, directional CNV annotations and total-copy-number matrices. |
| 03 | `03_build_sv_features.py` | Collapse reciprocal BRASS breakends to unique junctions and construct gene-level SV, burden and architecture matrices. |
| 04 | `04_merge_genomic_modalities.py` | Intersect sample identifiers and merge 210 SNV, 130 CNV and 31 SV features into the 371-feature discovery matrix. |
| 05 | `05_discover_nmf_subtypes.py` | Evaluate NMF at k=4-6 by consensus stability, prevalence-preserving nulls and resampling; regenerate and write the locked C1-C4 labels. |
| 06 | `06_baseline_characteristics.py` | Generate the clinical, pathological and coarse-genomic baseline table by subtype. |
| 07 | `07_survival_analysis.R` | Perform primary OS and supporting Stage I-III RFS analyses, adjusted Cox modelling, likelihood-ratio tests, diagnostics and sensitivity analyses. |
| 08 | `08_plot_subtype_genomic_landscape.py` | Produce SNV, directional-CNV, chromosome-arm and SV subtype heatmaps with association statistics. |
| 09 | `09_test_driver_feature_enrichment.py` | Test one-versus-rest enrichment/depletion of every discovery feature and summarize subtype-defining genomic events. |
| 10 | `10_compare_genomic_burdens.py` | Estimate C4-versus-rest and C4-versus-C1 genomic-burden effects using Cliff's delta and bootstrap confidence intervals. |
| 11 | `11_characterize_genomic_burden.py` | Compare TMB, unique SV junctions, ecDNA, hypoxia, pathology purity and related genomic variables. |
| 12 | `12_assess_purity_ploidy_wgd.py` | Test whether purity, ploidy or whole-genome doubling explains the C1-C4 CNV-direction and survival results. |
| 13 | `13_assess_site_grade_confounding.py` | Assess primary site and tumour grade in C4-versus-C1 survival models. |
| 14 | `14_analyze_mutational_signatures.py` | Compare SBS, DBS and indel signature burdens, relative contributions and aetiological groups. |
| 15 | `15_analyze_pathway_comutation.py` | Test pathway co-alteration and mutual exclusivity within each subtype. |
| 16 | `16_analyze_pathway_dual_hits.py` | Test convergent CNV-loss/LOH and SV hits in WNT, TGF-beta and PI3K pathway gene sets. |
| 17 | `17_prepare_shatterseek_inputs.py` | Prepare per-tumour BRASS SV and FACETS CNV tables for ShatterSeek. |
| 18 | `18_run_shatterseek.R` | Run ShatterSeek on the prepared tumour-level inputs. |
| 19 | `19_analyze_chromothripsis.py` | Classify candidate/high-confidence calls and test chromothripsis prevalence by subtype. |
| 20 | `20_compute_gsva_scores.R` | Compute Hallmark and KEGG GSVA scores from cohort expression matrices. |
| 21 | `21_characterize_transcriptome_immune_drug.py` | Compare gene expression, GSVA, CIBERSORTx, TIDE, ESTIMATE and predicted drug-response profiles. |
| 22 | `22_compare_cms_crps_icms.py` | Quantify correspondence with CMS, CRPS and iCMS and test complementary prognostic information. |
| 23 | `23_test_supervised_recoverability.py` | Assess held-out label recovery across eight classifier families and summarize important features. |
| 24 | `24_test_modality_dropout.py` | Reconstruct subtypes using all seven non-empty SNV/CNV/SV combinations with cross-fitting and paired bootstrap comparisons. |
| 25 | `25_assess_tcga_portability.py` | Perform coverage-matched transfer and exploratory de novo reconstruction in TCGA COAD/READ. |

Modules 15 and 21 from the development workspace and the minimal-feature
classifier are not part of this repository because their results are not used
in the manuscript.

## Software

The Python analyses used Python 3.10.14. Install the recorded environment with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The R analyses used R 4.6.0. Required package versions are listed in
[`R_PACKAGES.md`](R_PACKAGES.md). Analysis scripts never install packages at
runtime.

## Input layout

The default layout below mirrors the data interfaces used by the analysis.
All paths can be overridden using the variables in `config.example.env`.

```text
genome-first-crc-subtyping/
├── scripts/
├── colorectal_cancer_all_genes.txt
├── clinical_data.tsv
├── crc_heterogeneity_data/
│   ├── Supplementary_Table_01.xlsx
│   ├── SBS_signatures.xlsx
│   ├── complete_z_scores_Train.csv
│   ├── complete_z_scores_Test.csv
│   ├── emtab_zscore.csv
│   ├── TRAIN_TEST_CIBERSORT_Results.csv
│   ├── train_test_tide_output.csv
│   ├── estimate_scores.gct
│   └── DrugPredictions.csv
└── ../
    ├── genomics_raw_vcf/
    │   ├── vep_output/
    │   ├── cnv/
    │   └── sv/
    └── reference/
        ├── cytoBand.txt.gz
        └── genes_protein_coding_hg38.bed
```

`TRAIN_TEST_CIBERSORT_Results.csv`, `train_test_tide_output.csv`,
`estimate_scores.gct` and `DrugPredictions.csv` are upstream method outputs,
not raw measurements. They must be generated using CIBERSORTx, TIDE, ESTIMATE
and oncoPredict/GDSC2, respectively, following the manuscript methods.
Script 21 performs the subtype-level statistical analysis of these outputs.

## Configuration and execution

Copy the example configuration and edit paths only if the default layout is
not used:

```bash
cp config.example.env config.env
source config.env
```

Run the discovery workflow in order:

```bash
python scripts/01_build_snv_features.py
python scripts/02_build_cnv_features.py
python scripts/03_build_sv_features.py
python scripts/04_merge_genomic_modalities.py
python scripts/05_discover_nmf_subtypes.py
```

The expected label file is:

```text
module05_06_loocv_results/labels/NMF_k4_LOOCV.csv
```

Script 05 will stop rather than write labels if the regenerated k=4 counts
do not equal `426, 274, 268, 94`. The historical label file may optionally be
provided through `CRC_HISTORICAL_LABEL_FILE` for a patient-level equality
check, but it is not required to generate the partition.

After label generation, scripts 06-25 can be run in numerical order. Scripts
17-19 form one three-stage ShatterSeek workflow. Script 20 must precede script
21 when GSVA score files are not already available.

## Reproducibility safeguards

- The master random seed is 42 unless a documented environment override is
  supplied.
- NMF subtype selection does not use survival outcomes. OS event status is
  used only to balance cross-fitting folds; an unstratified sensitivity
  analysis is included.
- Full-cohort random NMF starts contribute only to consensus matrices.
  Patient labels are generated from pooled held-out cross-fitted assignments.
- Reciprocal BRASS breakends are counted once as unique SV junctions
  throughout burden analyses.
- Multiple-comparison families use Benjamini-Hochberg correction as specified
  within each script.
- TCGA is treated as a cross-platform portability assessment rather than
  equivalent external WGS validation because only 7 of 31 discovery SV
  features are available.
- Intermediate and final result directories are excluded by `.gitignore`.

## Provenance

`SCRIPT_PROVENANCE.tsv` records the development script from which each public
script was curated and its source SHA-256 checksum. Public copies contain only
portability, documentation and manuscript-scope edits; the original project
files were not modified.

## Citation

If these scripts are used, cite the accompanying manuscript and the source
cohort:

> Nunes L, *et al.* Prognostic genome and transcriptome signatures in
> colorectal cancers. *Nature*. 2024;633:137-146.
