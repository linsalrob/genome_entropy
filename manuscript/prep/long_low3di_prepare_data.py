#!/usr/bin/env python3
"""Build Table S15 (24 long, low-3Di ORFs with ESMFold2 and AlphaFold 3 checks) from the Setonix long_low3di work.

Sources (Setonix /scratch/pawsey1018/edwa0468/genome_entropy/work/long_low3di/; also agent-handoffs
projects/genome_entropy/long_low3di/):
  long_low3di_examples.tsv       the 24 ORFs: accessions, annotation, lengths, DNA/aa/3Di/12-state entropies (build_table.py)
  structure_vs_modernprost.tsv   ESMFold2-Fast on the first 1,500 aa + Foldseek 3Di (diag.slurm, job 51640400; struct3di.py)
  af3_vs_modernprost.tsv         ColabFold MSAs (UniRef30 2302 + envDB 202108, local GPU MMseqs2), MSA depth and Neff,
                                 AlphaFold 3 3.1.4 full length + Foldseek 3Di (af3.slurm, job 51694676; af3_summary.py)
  phage_collapse_by_length.tsv   share of annotated phage CDS with 3Di < 1 and < 2 bits by length

Writes data/tabLL01_long_low3di.tsv.gz, data/tabLL02_phage_low3di_by_length.tsv.gz and images/tabLL01_long_low3di.md.

  long_low3di_prepare_data.py
"""
from pathlib import Path

import pandas as pd

MS = Path(__file__).resolve().parent.parent
W = Path("/scratch/pawsey1018/edwa0468/genome_entropy/work/long_low3di")
GZ = {"method": "gzip", "mtime": 0}

CLASS = {**{i: "coiled coil / long helix" for i in ["E02", "E03", "E11", "E12", "E14", "E16", "E20", "E23"]},
         **{i: "disordered" for i in ["E01", "E05", "E09", "E10", "E13", "E18", "E21", "E22", "E24"]},
         **{i: "low-confidence, not disordered" for i in ["E07", "E15", "E17", "E19"]},
         "E08": "ModernProst failure (folded)", "E04": "partial ModernProst failure", "E06": "partial ModernProst failure"}

ex = pd.read_csv(W / "long_low3di_examples.tsv", sep="\t").drop(columns=["aa", "top_3di", "set"])
esm = pd.read_csv(W / "structure_vs_modernprost.tsv", sep="\t")[["id", "length", "mean_plddt", "struct_3di_entropy"]]
esm.columns = ["id", "esmfold_window_aa", "esmfold_mean_plddt", "esmfold_3di_entropy"]
af = pd.read_csv(W / "af3_vs_modernprost.tsv", sep="\t")
af = af[af.id.str.startswith("E")][["id", "msa_depth", "msa_depth_cov50", "neff80", "af3_mean_plddt", "af3_fraction_disordered",
                                    "af3_3di_entropy"]]
d = ex.merge(esm, on="id").merge(af, on="id", how="left")
d["class"] = d.id.map(CLASS)
assert len(d) == 24 and d["class"].notna().all() and af.shape[0] == 14
d.to_csv(MS / "data/tabLL01_long_low3di.tsv.gz", sep="\t", index=False, float_format="%.3f", compression=GZ)
pd.read_csv(W / "phage_collapse_by_length.tsv", sep="\t").to_csv(MS / "data/tabLL02_phage_low3di_by_length.tsv.gz", sep="\t",
                                                                  index=False, float_format="%.3f", compression=GZ)

f = lambda x, n=2: "–" if pd.isna(x) else f"{x:.{n}f}"
rows = ["**Table S15.** Twenty-four long ORFs (> 500 aa) with low 3Di entropy, with structure-prediction checks. ORF, the stop-to-stop "
        "(GTDB, LOGAN) or longest start-to-stop (phage) ORF the entropies were computed on; ann., the annotated CDS or Prodigal gene. "
        "ESMFold2-Fast saw the first 1,500 aa without an MSA; AlphaFold 3 saw the full ORF with a ColabFold MSA (14 proteins). "
        "Neff, effective sequences at 80% identity. Structure 3Di is Foldseek's 3Di of the predicted model.", "",
        "| id | source | accession | annotation | ORF / ann. aa | DNA | aa | 3Di | 12-state | ESMFold pLDDT | ESMFold 3Di | MSA depth | Neff | AF3 pLDDT | AF3 disordered | AF3 3Di | class |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in d.itertuples():
    acc = r.accession.split(" / ")[-1]
    ann = r.annotation.split(";")[0].replace("none (Prodigal gene)", "Prodigal gene") if r.source.startswith("LOGAN") else r.annotation
    if r.source.startswith("LOGAN"):
        ann = r.annotation.split("; ")[1] if "; " in r.annotation else ann
    ann = ann.replace("seq hit ", "best hit: ")
    rows.append(f"| {r.id} | {r.source.split(' (')[0]} | {acc} | {ann} | {r.orf_aa_len} / {r.annotated_aa_len} | {f(r.dna_entropy)} | "
                f"{f(r.aa_entropy)} | {f(r.three_di_entropy)} | {f(r.twelve_state_entropy)} | {f(r.esmfold_mean_plddt, 0)} | "
                f"{f(r.esmfold_3di_entropy)} | {'–' if pd.isna(r.msa_depth) else int(r.msa_depth)} | {f(r.neff80, 0)} | "
                f"{f(r.af3_mean_plddt, 0)} | {f(r.af3_fraction_disordered)} | {f(r.af3_3di_entropy)} | {CLASS[r.id]} |")
(MS / "images/tabLL01_long_low3di.md").write_text("\n".join(rows) + "\n")
print(d[["id", "class", "esmfold_mean_plddt", "neff80", "af3_mean_plddt", "af3_fraction_disordered", "af3_3di_entropy"]].to_string())
