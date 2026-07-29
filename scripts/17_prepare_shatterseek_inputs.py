#!/usr/bin/env python3
"""Convert BRASS SV and FACETS CNV VCFs to ShatterSeek input tables.

Reciprocal breakends are collapsed to unique junctions and allele-specific CNV
segments are formatted per tumour. No chromothripsis classification is made in
this preprocessing step.
"""

import os
import re
import gzip
import glob
from pathlib import Path
import sys
import time
import pandas as pd

##############################################################################
# CONFIG (env-overridable, project convention)
##############################################################################
BASE = os.environ.get("CRC_BASE", str(Path(__file__).resolve().parents[1]))
SV_DIR = os.environ.get(
    "CRC_M10_SV_DIR", str(Path(BASE).parent / "genomics_raw_vcf" / "sv")
)
CNV_DIR = os.environ.get(
    "CRC_M10_CNV_DIR", str(Path(BASE).parent / "genomics_raw_vcf" / "cnv")
)
OUTDIR  = os.environ.get("CRC_M10_OUT", f"{BASE}/module10_chromothripsis_C1C4_ppt")
INTERDIR = os.path.join(OUTDIR, "intermediate")
os.makedirs(INTERDIR, exist_ok=True)

# 0 = process every file found; set e.g. CRC_M10_LIMIT=10 for a smoke test
LIMIT = int(os.environ.get("CRC_M10_LIMIT", "0"))

VALID_CHROMS = {str(i) for i in range(1, 23)} | {"X"}  # ShatterSeek chromNames

##############################################################################
# SAMPLE ID
##############################################################################
def normalize_sample(fname):
    m = re.search(r"(UM\d+|U\d+)", fname)
    return m.group(1) if m else None

##############################################################################
# VCF INFO PARSER (shared)
##############################################################################
def parse_info(info_str):
    d = {}
    for kv in info_str.split(";"):
        if not kv:
            continue
        if "=" in kv:
            k, v = kv.split("=", 1)
            d[k] = v
        else:
            d[kv] = True
    return d

def strip_chr(c):
    return c[3:] if c.startswith("chr") else c

##############################################################################
# BND ALT BREAKEND NOTATION (VCFv4.2 spec)
##############################################################################
# t[p[  -> local strand '+' (t precedes bracket; piece at p read forward)
# t]p]  -> local strand '+' (t precedes bracket; piece at p read reverse)
# ]p]t  -> local strand '-' (t follows bracket)
# [p[t  -> local strand '-' (t follows bracket)
_RE_T_BEFORE_SQOPEN  = re.compile(r"^[^\[\]]+\[([^:\[\]]+):(\d+)\[$")   # t[p[
_RE_T_BEFORE_SQCLOSE = re.compile(r"^[^\[\]]+\]([^:\[\]]+):(\d+)\]$")  # t]p]
_RE_T_AFTER_SQCLOSE  = re.compile(r"^\]([^:\[\]]+):(\d+)\][^\[\]]+$")  # ]p]t
_RE_T_AFTER_SQOPEN   = re.compile(r"^\[([^:\[\]]+):(\d+)\[[^\[\]]+$")  # [p[t

def parse_bnd_alt(alt):
    """Return (local_strand, mate_chrom, mate_pos) for a single BND ALT
    allele, per the VCFv4.2 breakend spec. Returns None if unparseable
    (e.g. a non-BND ALT slipped through)."""
    m = _RE_T_BEFORE_SQOPEN.match(alt)
    if m:
        return "+", m.group(1), int(m.group(2))
    m = _RE_T_BEFORE_SQCLOSE.match(alt)
    if m:
        return "+", m.group(1), int(m.group(2))
    m = _RE_T_AFTER_SQCLOSE.match(alt)
    if m:
        return "-", m.group(1), int(m.group(2))
    m = _RE_T_AFTER_SQOPEN.match(alt)
    if m:
        return "-", m.group(1), int(m.group(2))
    return None

##############################################################################
# SV VCF -> ShatterSeek SV table
##############################################################################
def parse_sv_vcf(path, sample_id, warnings):
    opener = gzip.open if path.endswith(".gz") else open
    rows = {}
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            f8 = line.rstrip("\n").split("\t", 8)
            chrom, pos, vid, ref, alt, qual, filt, info = f8[:8]
            info_d = parse_info(info)
            if info_d.get("SVTYPE") != "BND":
                continue
            rows[vid] = dict(chrom=chrom, pos=int(pos), alt=alt,
                              mateid=info_d.get("MATEID"),
                              svclass=(info_d.get("SVCLASS") or "").lower())

    out = []
    visited = set()
    for vid, r in rows.items():
        if vid in visited:
            continue
        mate_id = r["mateid"]
        if mate_id is None or mate_id not in rows:
            warnings.append(f"{sample_id}\t{vid}\tno resolvable MATEID, skipped")
            visited.add(vid)
            continue
        mate = rows[mate_id]
        visited.add(vid); visited.add(mate_id)

        p1 = parse_bnd_alt(r["alt"])
        p2 = parse_bnd_alt(mate["alt"])
        if p1 is None or p2 is None:
            warnings.append(f"{sample_id}\t{vid}/{mate_id}\tunparseable BND ALT, skipped")
            continue
        strand1_raw, mchrom1, mpos1 = p1
        strand2_raw, mchrom2, mpos2 = p2
        # Cross-check: each breakend's own bracket notation should point
        # back at the other breakend's actual locus. Chromosome must match
        # exactly. Position is allowed a small tolerance: BRASS's IMPRECISE/
        # CIPOS/CIEND-adjusted BND records routinely encode a mate position
        # a few bp off from the mate record's own POS (confirmed empirically
        # on this cohort: 0-6bp offset, always same chromosome, consistent
        # with confidence-interval rounding, not a data error) -- ShatterSeek
        # only uses these coordinates at kilobase-scale windows downstream,
        # so a <=25bp tolerance here is inconsequential.
        POS_TOL = 25
        if mchrom1 != mate["chrom"] or abs(mpos1 - mate["pos"]) > POS_TOL or \
           mchrom2 != r["chrom"] or abs(mpos2 - r["pos"]) > POS_TOL:
            warnings.append(f"{sample_id}\t{vid}/{mate_id}\tbreakend cross-check mismatch "
                             f"(chrom {mchrom1}/{mate['chrom']}, pos {mpos1}/{mate['pos']}), skipped")
            continue

        chrom1, pos1, strand1 = r["chrom"], r["pos"], strand1_raw
        chrom2, pos2, strand2 = mate["chrom"], mate["pos"], strand2_raw
        # Force pos1 <= pos2 ourselves (see module docstring point 2 --
        # ShatterSeek's own constructor swaps chrom/pos but not strand on
        # pos1>pos2, which would silently decouple strand from position).
        if pos1 > pos2:
            chrom1, chrom2 = chrom2, chrom1
            pos1, pos2 = pos2, pos1
            strand1, strand2 = strand2, strand1

        svclass = r["svclass"] or mate["svclass"]
        if svclass == "deletion":
            svtype = "DEL"
            if (strand1, strand2) != ("+", "-"):
                warnings.append(f"{sample_id}\t{vid}\tDEL with unexpected strand pair "
                                 f"({strand1},{strand2})")
        elif svclass == "tandem-duplication":
            svtype = "DUP"
            if (strand1, strand2) != ("-", "+"):
                warnings.append(f"{sample_id}\t{vid}\tDUP with unexpected strand pair "
                                 f"({strand1},{strand2})")
        elif svclass == "translocation":
            svtype = "TRA"
        elif svclass == "inversion":
            if (strand1, strand2) == ("+", "+"):
                svtype = "h2hINV"
            elif (strand1, strand2) == ("-", "-"):
                svtype = "t2tINV"
            else:
                warnings.append(f"{sample_id}\t{vid}\tSVCLASS=inversion but strand pair "
                                 f"({strand1},{strand2}) is neither h2h nor t2t, skipped")
                continue
        else:
            warnings.append(f"{sample_id}\t{vid}\tunrecognized SVCLASS={svclass!r}, skipped")
            continue

        c1, c2 = strip_chr(chrom1), strip_chr(chrom2)
        if c1 not in VALID_CHROMS or c2 not in VALID_CHROMS:
            continue  # chrY/chrM/alt contigs -- ShatterSeek supports chr1-22+X only

        out.append((sample_id, c1, pos1, c2, pos2, strand1, strand2, svtype))
    return out

##############################################################################
# CNV VCF -> ShatterSeek CNV segment table
##############################################################################
def parse_cnv_vcf(path, sample_id, warnings):
    opener = gzip.open if path.endswith(".gz") else open
    recs = []
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            f8 = line.rstrip("\n").split("\t", 8)
            chrom, pos, vid, ref, alt, qual, filt, info = f8[:8]
            info_d = parse_info(info)
            end = info_d.get("END")
            tcn = info_d.get("TCN_EM")
            if end is None or tcn is None or tcn == ".":
                continue
            try:
                tcn_val = int(round(float(tcn)))
            except ValueError:
                warnings.append(f"{sample_id}\t{vid}\tnon-numeric TCN_EM={tcn!r}, skipped")
                continue
            c = strip_chr(chrom)
            if c not in VALID_CHROMS:
                continue
            recs.append((c, int(pos), int(end), tcn_val))

    if not recs:
        return []

    df = pd.DataFrame(recs, columns=["chrom", "start", "end", "total_cn"])
    df = df.sort_values(["chrom", "start"])

    # Merge adjacent same-CN segments per chromosome (ShatterSeek README
    # requirement -- see module docstring point 4).
    merged = []
    CHROM_SORT = {c: i for i, c in enumerate(
        [str(i) for i in range(1, 23)] + ["X"])}
    for chrom in sorted(df["chrom"].unique(), key=lambda c: CHROM_SORT[c]):
        g = df.loc[df["chrom"] == chrom].sort_values("start").to_dict("records")
        cur = dict(g[0])
        for row in g[1:]:
            if row["total_cn"] == cur["total_cn"] and row["start"] <= cur["end"] + 1:
                cur["end"] = max(cur["end"], row["end"])
            else:
                merged.append(cur)
                cur = dict(row)
        merged.append(cur)

    return [(sample_id, m["chrom"], m["start"], m["end"], m["total_cn"]) for m in merged]

##############################################################################
# MAIN
##############################################################################
def main():
    sv_files = sorted(glob.glob(os.path.join(SV_DIR, "*.SV.vcf.gz")))
    cnv_files = sorted(glob.glob(os.path.join(CNV_DIR, "*.vcf.gz")))
    if not sv_files:
        sys.exit(f"No SV VCFs found under {SV_DIR} -- check CRC_M10_SV_DIR")
    if not cnv_files:
        sys.exit(f"No CNV VCFs found under {CNV_DIR} -- check CRC_M10_CNV_DIR")

    if LIMIT > 0:
        sv_files = sv_files[:LIMIT]
        cnv_files = cnv_files[:LIMIT]

    print("=" * 70)
    print("MODULE 10a -- PREPARE SHATTERSEEK INPUTS")
    print("=" * 70)
    print(f"SV VCFs found : {len(sv_files)} in {SV_DIR}")
    print(f"CNV VCFs found: {len(cnv_files)} in {CNV_DIR}")
    if LIMIT > 0:
        print(f"CRC_M10_LIMIT={LIMIT} -- SMOKE-TEST MODE, not processing the full cohort")

    warnings = []
    sv_rows, cnv_rows = [], []
    t0 = time.time()

    zero_sv_samples, zero_cnv_samples = [], []
    for i, path in enumerate(sv_files, 1):
        sample_id = normalize_sample(os.path.basename(path))
        if sample_id is None:
            warnings.append(f"?\t{path}\tcould not extract sample id from filename, skipped file")
            continue
        rows = parse_sv_vcf(path, sample_id, warnings)
        if not rows:
            zero_sv_samples.append(sample_id)
        sv_rows.extend(rows)
        if i % 100 == 0 or i == len(sv_files):
            print(f"  SV:  {i}/{len(sv_files)} samples parsed "
                  f"({len(sv_rows)} SV records so far, {time.time()-t0:.0f}s)")

    t0 = time.time()
    for i, path in enumerate(cnv_files, 1):
        sample_id = normalize_sample(os.path.basename(path))
        if sample_id is None:
            warnings.append(f"?\t{path}\tcould not extract sample id from filename, skipped file")
            continue
        rows = parse_cnv_vcf(path, sample_id, warnings)
        if not rows:
            zero_cnv_samples.append(sample_id)
        cnv_rows.extend(rows)
        if i % 100 == 0 or i == len(cnv_files):
            print(f"  CNV: {i}/{len(cnv_files)} samples parsed "
                  f"({len(cnv_rows)} segments so far, {time.time()-t0:.0f}s)")

    sv_df = pd.DataFrame(sv_rows, columns=[
        "sample_id", "chrom1", "pos1", "chrom2", "pos2", "strand1", "strand2", "SVtype"])
    cnv_df = pd.DataFrame(cnv_rows, columns=[
        "sample_id", "chrom", "start", "end", "total_cn"])

    sv_out = os.path.join(INTERDIR, "shatterseek_sv_input.csv")
    cnv_out = os.path.join(INTERDIR, "shatterseek_cnv_input.csv")
    sv_df.to_csv(sv_out, index=False)
    cnv_df.to_csv(cnv_out, index=False)

    warn_path = os.path.join(INTERDIR, "prepare_warnings.log")
    with open(warn_path, "w") as f:
        f.write("\n".join(warnings))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"SV records written : {len(sv_df)}  -> {sv_out}")
    print(f"  SVtype distribution:\n{sv_df['SVtype'].value_counts().to_string()}" if len(sv_df) else "  (empty)")
    print(f"CNV segments written: {len(cnv_df)} -> {cnv_out}")
    print(f"Samples with zero SV records : {len(zero_sv_samples)}")
    print(f"Samples with zero CNV segments: {len(zero_cnv_samples)}")
    print(f"Warnings logged: {len(warnings)} -> {warn_path}")
    print("\nNext step: run 10b_module10_run_shatterseek.R on these two CSVs.")

if __name__ == "__main__":
    main()
