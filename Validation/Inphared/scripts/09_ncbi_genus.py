#!/usr/bin/env python3
"""Genus (and family) of each Inphared genome from NCBI Taxonomy, for genus-grouped cross-validation.

The Pharokka/phold GenBank files carry no taxonomy, and the Inphared metadata
for this release is not published, so each accession is looked up in NCBI:
nuccore esummary -> taxid, then taxonomy efetch -> lineage. Only public
accessions are sent. Run on a node with internet access (the login node).

Output TSV: input_id, taxid, genus, family, lineage (genus/family empty when
the lineage has no such rank; many phages are unclassified at genus level).

  09_ncbi_genus.py --ids ids.txt --out genus.tsv
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def post(tool: str, **params) -> bytes:
    data = urllib.parse.urlencode({**params, "tool": "genome_entropy", "email": "robert.edwards@flinders.edu.au"}).encode()
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{EUTILS}/{tool}", data=data, timeout=120) as r:
                out = r.read()
            time.sleep(0.4)  # stay under 3 requests per second without an API key
            return out
        except Exception as e:  # noqa: BLE001 - transient HTTP / network errors
            print(f"{tool} attempt {attempt + 1}: {e}", file=sys.stderr, flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{tool} failed 5 times")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ids", required=True, type=Path, help="one accession per line")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch", type=int, default=300)
    a = ap.parse_args()
    ids = [x.strip() for x in a.ids.read_text().split() if x.strip()]

    acc2tax: dict[str, str] = {}
    for i in range(0, len(ids), a.batch):
        chunk = ids[i:i + a.batch]
        root = ET.fromstring(post("esummary.fcgi", db="nuccore", id=",".join(chunk)))
        for doc in root.iter("DocSum"):
            items = {it.get("Name"): (it.text or "") for it in doc.iter("Item")}
            acc2tax[items.get("Caption", "")] = items.get("TaxId", "")
            acc2tax[items.get("AccessionVersion", "").split(".")[0]] = items.get("TaxId", "")
        print(f"esummary {i + len(chunk):,}/{len(ids):,}", flush=True)

    taxids = sorted({t for t in acc2tax.values() if t})
    lineage: dict[str, tuple[str, str, str]] = {}
    for i in range(0, len(taxids), a.batch):
        root = ET.fromstring(post("efetch.fcgi", db="taxonomy", id=",".join(taxids[i:i + a.batch])))
        for taxon in root.findall("Taxon"):
            tid = taxon.findtext("TaxId")
            ranks = {t.findtext("Rank"): t.findtext("ScientificName") for t in taxon.iter("Taxon")}
            ranks[taxon.findtext("Rank")] = taxon.findtext("ScientificName")
            lineage[tid] = (ranks.get("genus", ""), ranks.get("family", ""), taxon.findtext("Lineage") or "")
        print(f"taxonomy {min(i + a.batch, len(taxids)):,}/{len(taxids):,}", flush=True)

    n_genus = 0
    with open(a.out, "w") as fh:
        fh.write("input_id\ttaxid\tgenus\tfamily\tlineage\n")
        for acc in ids:
            t = acc2tax.get(acc.split(".")[0], "")
            g, f, lin = lineage.get(t, ("", "", ""))
            n_genus += bool(g)
            fh.write(f"{acc}\t{t}\t{g}\t{f}\t{lin}\n")
    print(f"{len(ids):,} accessions; {sum(bool(acc2tax.get(x.split('.')[0])) for x in ids):,} with a taxid; "
          f"{n_genus:,} with a genus")
    return 0


if __name__ == "__main__":
    sys.exit(main())
