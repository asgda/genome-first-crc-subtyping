# R environment

The analyses were conducted with R 4.6.0. The manuscript-recorded package
versions are:

| Package | Version |
|---|---:|
| survival | 3.8-6 |
| survminer | 0.5.2 |
| multcomp | 1.4-30 |
| ShatterSeek | 1.1 |
| GSVA | 2.0.7 |
| msigdbr | 26.1.0 |
| GSEABase | 1.70.1 |

The R scripts also require `ggplot2`, `dplyr`, `tidyr`, `readxl`, `broom`,
`stringr`, `purrr`, `tibble`, `svglite`, and `data.table`. Exact
versions of all loaded packages are written by the survival workflow through
`sessionInfo()`. Package installation is intentionally not performed inside
analysis scripts.
