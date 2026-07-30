# Blacklisted / passenger genes excluded from the feature panels

**Revision note (2026-07-30):** an earlier version of this file identified the wrong
mechanism (the `blacklist_patterns` substring filter, 7 genes) as the source of the
manuscript's "27 genes" claim. That filter is real but is not what "27" refers to. The
actual match is the `PASSENGER_GENES` list in the CNV module, documented below. This
version corrects that error and documents all four filtering mechanisms found across the
three modality scripts.

## Summary

There are **two distinct kinds of gene exclusion** in the pipeline, and they are *not*
the same list applied three times — each modality script defines its own:

| Mechanism | Module(s) | Gene count | Matches manuscript's "27"? |
|---|---|---|---|
| `PASSENGER_GENES` (hardcoded list of long/hypermutable genes) | SNV | 21 | No |
| `PASSENGER_GENES` (hardcoded list of long/hypermutable genes) | CNV | **27** | **Yes — exact match** |
| `PASSENGER_GENES` (hardcoded list of long/hypermutable genes) | SV | 20 | No |
| `blacklist_patterns` (substring rule: "OR"/"LOC"/"LINC") | CNV only | ≤7 (of the 346-gene panel) | No |

The manuscript's Preprocessing sentence — *"...and removed blacklisted genes (27 genes,
Table xx)"* — is worded as if one shared filter were applied generally across "the
datasets." In the code, it is not: each modality has its own list, and only the CNV
module's list is exactly 27 genes. If a reviewer asks which 27 genes and whether they
were applied identically to SNV/CNV/SV, the accurate answer is: no, only CNV uses this
specific 27-gene list; SNV and SV apply their own, different, shorter lists. Decide
whether to state this precisely in Methods or leave the general phrasing — either is
defensible, but the current text implies more uniformity than the code has.

## Module 2 (CNV) — 27 genes — matches the manuscript

Source: `scripts/02_build_cnv_features.py` (working copy: `02_module2_cnv_matrix.py`),
`PASSENGER_GENES`. Actively applied to the gene BED file before CNV feature
construction: `gene_bed = gene_bed[~gene_bed["gene"].isin(PASSENGER_GENES)]`.

```
TTN, MUC16, OBSCN,
DNAH5, DNAH6, DNAH7, DNAH8, DNAH9, DNAH10, DNAH11, DNAH14, DNAH17,
CSMD1, CSMD3,
RYR1, RYR2, RYR3,
SYNE1, SYNE2,
USH2A, FLG,
PCLO, XIRP2,
FAT3, FAT4,
LRP1B, MACROD2
```

## Module 1 (SNV) — 21 genes — different list

Source: `scripts/01_build_snv_features.py` (working copy: `01_module1_snv_matrix.py`),
`PASSENGER_GENES`.

```
TTN, MUC16, MUC4, MUC17, MUC5B, MUC6, OBSCN, FLG,
DNAH5, DNAH11, DNAH2,
RYR1, RYR2, RYR3,
CSMD1, CSMD3,
LRP1B, PCLO, XIRP2,
FAT4, FAT3
```

## Module 3 (SV) — 20 genes — same as the SNV list minus FLG

Source: `scripts/03_build_sv_features.py` (working copy: `03_module3_sv_matrix.py`),
`PASSENGER_GENES`.

```
TTN, MUC16, MUC4, MUC17, MUC5B, MUC6, OBSCN,
DNAH5, DNAH11, DNAH2,
RYR1, RYR2, RYR3,
CSMD1, CSMD3,
LRP1B, PCLO, XIRP2,
FAT3, FAT4
```

## Secondary filter: CNV-only substring blacklist (not the "27" list)

A second, unrelated mechanism exists only in the CNV module, applied later in the
pipeline (after recurrence filtering) and using a completely different method —
substring matching rather than a hardcoded gene list:

```python
blacklist_patterns = ["OR", "LOC", "LINC"]
recurrent_gene_features = set([
    x for x in recurrent_gene_features
    if not any([pat in x for pat in blacklist_patterns])
])
```

Checked directly against the 346-gene CRC panel: 7 genes match — BCOR, BCORL1, MTOR,
NCOR2, PORCN, ROR1, ROR2 (all via "OR"; none of the 346 panel genes contain "LOC" or
"LINC", so those two patterns have no effect on this specific panel). This is a real,
separate, additional filter, distinct from `PASSENGER_GENES`, and is not what the
manuscript's "27" refers to.

## Recommended manuscript fix

Cite the CNV module's `PASSENGER_GENES` list (27 genes, reproduced above) for the "27
genes" claim — the number already matches exactly, no correction to "27" is needed.
What's worth fixing, if you want full precision, is the implication that one blacklist
was applied uniformly across modalities; if that distinction matters for your target
journal, say explicitly that this is the CNV-specific passenger list, and that SNV/SV use
their own separate (21- and 20-gene) lists.

## Source

- `scripts/01_build_snv_features.py`, `scripts/02_build_cnv_features.py`,
  `scripts/03_build_sv_features.py` — `PASSENGER_GENES` definitions
- `scripts/02_build_cnv_features.py` — `blacklist_patterns` definition (secondary filter)
- Panel file: `colorectal_cancer_all_genes.txt` (346 genes after header-artifact cleanup)
- All counts verified directly against the working-copy scripts and the panel file
  (2026-07-30)
