#!/usr/bin/env python3
"""Render the population summary TSVs as the markdown tables used in the report.

Reads only what 29_population_entropy_summary.pbs wrote. It recomputes nothing:
a presentation script that reimplements the analysis is a second implementation
that will drift from the first.

usage: population_tables.py <summary_dir>
"""
import sys, csv, os

GROUPS = ["all ORFs", "in_genbank=True", "in_genbank=False",
          "CDS-bearing genomes: all", "CDS-bearing genomes: matched",
          "CDS-bearing genomes: unmatched", "no-CDS genomes: all"]
LABEL = {
    "all ORFs": "all ORFs",
    "in_genbank=True": "matched (`in_genbank=True`)",
    "in_genbank=False": "unmatched (`in_genbank=False`)",
    "CDS-bearing genomes: all": "all ORFs",
    "CDS-bearing genomes: matched": "matched",
    "CDS-bearing genomes: unmatched": "unmatched",
    "no-CDS genomes: all": "all ORFs (all unmatched by construction)",
}

def load(path):
    d = {}
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            d[(r["group"], r["metric"])] = r
    return d

def f(r, k, nd=3):
    v = r.get(k, "")
    return f"{float(v):.{nd}f}" if v not in ("", None) else ""

def block(d, groups, title):
    out = [f"**{title}**", "",
           "| group | n | mean 3Di | median 3Di | Q1 | Q3 | IQR | mean protein | "
           "median protein | mean 12-state | mean 3Di–12st MI | mean aa length |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in groups:
        t = d.get((g, "three_di"))
        if t is None:
            continue
        p = d[(g, "protein")]; w = d[(g, "twelve_state")]; m = d[(g, "mi")]
        a = d[(g, "aa_length")]
        out.append(
            f"| {LABEL[g]} | {int(t['n']):,} | {f(t,'mean')} | {f(t,'median')} | "
            f"{f(t,'q1')} | {f(t,'q3')} | {f(t,'iqr')} | {f(p,'mean')} | "
            f"{f(p,'median')} | {f(w,'mean')} | {f(m,'mean')} | {f(a,'mean',1)} |")
    out.append("")
    return out

def main():
    sd = sys.argv[1]
    lines = []
    for dom, name in (("bac", "Bacteria"), ("arc", "Archaea")):
        d = load(os.path.join(sd, f"population_entropy_{dom}.tsv"))
        allo = int(d[("all ORFs", "three_di")]["n"])
        mat = int(d[("in_genbank=True", "three_di")]["n"])
        unm = int(d[("in_genbank=False", "three_di")]["n"])
        cds = int(d[("CDS-bearing genomes: all", "three_di")]["n"])
        cmat = int(d[("CDS-bearing genomes: matched", "three_di")]["n"])
        lines += [f"#### {name}", "",
                  f"| | count | share of all ORFs |", "|---|---:|---:|",
                  f"| all ORFs | {allo:,} | 100% |",
                  f"| matched (`in_genbank=True`) | {mat:,} | {100*mat/allo:.2f}% |",
                  f"| unmatched (`in_genbank=False`) | {unm:,} | {100*unm/allo:.2f}% |",
                  f"| in genomes with ≥1 deposited CDS | {cds:,} | {100*cds/allo:.2f}% |",
                  f"| ... of which matched | {cmat:,} | {100*cmat/cds:.2f}% of that subset |",
                  ""]
        lines += block(d, GROUPS[:3], "All genomes")
        lines += block(d, GROUPS[3:6], "Restricted to genomes with at least one deposited CDS")
        lines += block(d, GROUPS[6:], "Genomes with no deposited CDS (excluded from every figure)")

    lines += ["### Figure samples, as plotted", "",
              "| | bacteria | archaea |", "|---|---:|---:|"]
    fs = {dom: load(os.path.join(sd, f"figure_sample_{dom}.tsv")) for dom in ("bac", "arc")}
    rows = [("sampled rows", "all ORFs"),
            ("dropped: genomes with no annotated CDS", "no-CDS genomes: all"),
            ("**plotted**", "CDS-bearing genomes: all"),
            ("plotted, matched", "CDS-bearing genomes: matched"),
            ("plotted, unmatched", "CDS-bearing genomes: unmatched")]
    for lab, g in rows:
        vals = []
        for dom in ("bac", "arc"):
            r = fs[dom].get((g, "three_di"))
            vals.append(f"{int(r['n']):,}" if r else "0")
        lines.append(f"| {lab} | {vals[0]} | {vals[1]} |")
    for lab, g in (("plotted matched fraction", None),):
        vals = []
        for dom in ("bac", "arc"):
            tot = int(fs[dom][("CDS-bearing genomes: all", "three_di")]["n"])
            mm = int(fs[dom][("CDS-bearing genomes: matched", "three_di")]["n"])
            vals.append(f"{100*mm/tot:.2f}%")
        lines.append(f"| {lab} | {vals[0]} | {vals[1]} |")
    lines.append("")
    print("\n".join(lines))

main()
