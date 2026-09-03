#!/usr/bin/env python3
"""Functional class per candidate, and a shortlist stratified by it (issue #97).

Rob's direction on #97:

  "Keep mobile-element candidates in the total. They are real protein-coding
   genes if supported... However, quantify and report them explicitly as a
   separate functional category."
  "Select manuscript examples by biological class, not simply by score. Do not
   take the top 10 TonB-dependent receptors or the top mobile-element hits."

So this stage does not filter anything. It labels, reports the composition,
and then picks a shortlist by walking classes rather than by walking score --
which is the only way to avoid ten versions of one phenomenon. Ranking by score
alone put 900-1,080 aa Bacteroidota TonB-dependent receptors in every one of
the top ten bacterial slots.

WHERE THE LABELS COME FROM, AND WHAT THAT LIMITS

Only Swiss-Prot and PDB100 carry free-text product names; CATH50 gives a
superfamily code and BFVD a bare accession (21_target_descriptions.py). So a
candidate is classifiable only if its best structural hit is in Swiss-Prot or
PDB. Everything else is `unclassified` -- which is a property of the reference
database, NOT weak evidence, and must not be read as one.

Keyword classification is crude by construction. It is used here for
STRATIFICATION and for reporting composition, never as evidence about any
individual ORF, and the class of anything that reaches a figure should be
checked by hand.

ORDER MATTERS. The patterns are applied in sequence and the first match wins,
because product names carry several senses at once: "prophage integrase" is a
mobile element rather than a DNA-repair enzyme, and "DNA endonuclease I-CreI"
is a homing endonuclease rather than a nuclease. Mobile-element and defence
patterns therefore run before the generic enzyme ones.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# First match wins. Keep mobile_element and defence ahead of the generic
# enzyme classes -- see the docstring.
CLASSES = [
    ("mobile_element", re.compile(
        r"transpos|resolvase|recombinase|integrase|relaxase|excisionase|"
        r"insertion sequence|\bIS\d|intron|prophage|\bphage|capsid|tail fibre|"
        r"tail fiber|portal protein|terminase|\bI-[A-Z][a-z]{2}[IVX]+\b|"
        r"mobile element|conjugal|conjugative|plasmid|retron|"
        r"reverse transcriptase|group II intron", re.I)),
    ("defence", re.compile(
        r"CRISPR|\bcas\d|restriction|methyltransferase subunit|toxin|"
        r"antitoxin|abortive infection|nicking enzyme|argonaute|"
        r"retributive|anti-?phage", re.I)),
    ("transport", re.compile(
        r"TonB|transporter|permease|\bporin\b|channel|efflux|\bABC\b|"
        r"symporter|antiporter|uptake|siderophore|receptor P\d|"
        r"outer membrane|translocase|secretion", re.I)),
    ("replication_repair", re.compile(
        r"\bRec[A-Z]\b|DNA polymerase|DNA ligase|primase|helicase|"
        r"topoisomerase|gyrase|mismatch repair|\bMut[A-Z]\b|excinuclease|"
        r"single-strand|replication|DNA repair|Holliday", re.I)),
    ("translation", re.compile(
        r"ribosomal|ribonuclease P|\btRNA\b|\brRNA\b|elongation factor|"
        r"initiation factor|release factor|aminoacyl|synthetase.*tRNA|"
        r"peptidyl", re.I)),
    ("regulation", re.compile(
        r"transcriptional regulator|transcription regulator|sigma factor|"
        r"histidine kinase|response regulator|two-component|repressor|"
        r"activator|anti-sigma|\bregulator\b", re.I)),
    ("cell_envelope", re.compile(
        r"peptidoglycan|cell wall|murein|flagell|pilus|pili|fimbri|"
        r"S-layer|lipopolysaccharide|teichoic|sortase|autolysin|"
        r"cell division|septum", re.I)),
    ("metabolism", re.compile(
        r"dehydrogenase|reductase|oxidase|oxygenase|kinase|phosphatase|"
        r"synthase|synthetase|transferase|hydrolase|isomerase|mutase|lyase|"
        r"carboxylase|decarboxylase|aldolase|esterase|lipase|protease|"
        r"peptidase|amidase|deaminase|dehydratase|epimerase|racemase|"
        r"thiolase|acetyltransferase|CoA", re.I)),
    ("uncharacterized", re.compile(
        r"uncharacteri|hypothetical|\bDUF\d|putative protein|"
        r"^protein [A-Z]{2}\d|conserved protein", re.I)),
]

PRODUCT_COLS = ["afdb_swissprot_product", "pdb100_product"]


def classify(products):
    """Series of product strings -> Series of class labels."""
    out = pd.Series("unclassified", index=products.index, dtype=object)
    have = products.fillna("").str.len() > 0
    remaining = have.copy()
    for name, pat in CLASSES:
        if not remaining.any():
            break
        hit = remaining & products.fillna("").str.contains(pat)
        out[hit] = name
        remaining &= ~hit
    # Had a product name but matched nothing -> genuinely other, which is
    # different from having no product at all.
    out[remaining] = "other_named"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ranked", required=True,
                    help="ranked*_candidates.tsv.gz from 22_rank_candidates.py")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--per-class", type=int, default=25,
                    help="candidates kept per functional class in the shortlist")
    args = ap.parse_args()

    d = pd.read_csv(args.ranked, sep="\t", low_memory=False)
    print(f"ranked table : {len(d):,} candidates")

    # Best available product text: Swiss-Prot first, PDB as fallback.
    prod = pd.Series("", index=d.index, dtype=object)
    for c in PRODUCT_COLS:
        if c in d.columns:
            prod = prod.mask(prod.str.len() == 0, d[c].fillna(""))
    d["best_product"] = prod
    d["functional_class"] = classify(prod)
    d["mobile_element"] = d.functional_class == "mobile_element"

    # BFVD-only: a full-length viral hit and nothing full-length in the three
    # databases used for estimation. Rob: "may be exactly the kind of
    # phage/prophage biology we want to showcase."
    est = [c for c in ("afdb_swissprot_struct_full", "pdb100_struct_full",
                       "cath50_struct_full") if c in d.columns]
    if "bfvd_struct_full" in d.columns and est:
        any_est = d[est].fillna(False).any(axis=1)
        d["bfvd_only"] = d.bfvd_struct_full.fillna(False) & ~any_est
    else:
        d["bfvd_only"] = False

    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    for label, sub in [("ALL candidates", d),
                       ("any full-length struct, not truncated",
                        d[d.any_full_struct & ~d.truncated_calc.astype(bool)]),
                       ("strict combined rule", d[d.strict_rule])]:
        emit()
        emit("=" * 78)
        emit(f"FUNCTIONAL COMPOSITION -- {label}  (n = {len(sub):,})")
        emit("=" * 78)
        vc = sub.functional_class.value_counts()
        emit(f"  {'class':<22}{'n':>10}{'share':>9}")
        for k, v in vc.items():
            emit(f"  {k:<22}{v:>10,}{100.0 * v / max(len(sub), 1):>8.1f}%")
        emit(f"  {'-- mobile element':<22}{int(sub.mobile_element.sum()):>10,}"
             f"{sub.mobile_element.mean() * 100:>8.1f}%")
        emit(f"  {'-- BFVD-only hit':<22}{int(sub.bfvd_only.sum()):>10,}"
             f"{sub.bfvd_only.mean() * 100:>8.1f}%")

    # Stratified shortlist: walk classes, not score. Within a class, keep the
    # strict-rule candidates ordered as 22_rank_candidates.py ordered them.
    emit()
    emit("=" * 78)
    emit(f"STRATIFIED SHORTLIST -- up to {args.per_class} per class")
    emit("=" * 78)
    pool = d[d.strict_rule] if d.strict_rule.any() else d[d.any_full_struct]
    pool = pool[~pool.truncated_calc.astype(bool)]
    picks = []
    for cls, g in pool.groupby("functional_class"):
        take = g.head(args.per_class)
        picks.append(take)
        emit(f"  {cls:<22}{len(take):>6} of {len(g):>8,} available")
    short = pd.concat(picks, ignore_index=True) if picks else pool.head(0)

    out_tab = f"{args.out_prefix}_classified.tsv.gz"
    d.to_csv(out_tab, sep="\t", index=False, compression="gzip")
    out_short = f"{args.out_prefix}_stratified_shortlist.tsv"
    short.to_csv(out_short, sep="\t", index=False)
    out_rep = f"{args.out_prefix}_functional_report.txt"
    Path(out_rep).write_text("\n".join(lines) + "\n")
    print(f"\nclassified   -> {out_tab}  ({len(d):,} rows)")
    print(f"shortlist    -> {out_short}  ({len(short):,} rows)")
    print(f"report       -> {out_rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
