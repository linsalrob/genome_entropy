# Working on this pipeline

Guidance for an agent picking up the GTDB validation run. The pipeline is
**complete and has produced published results** — this is not a first draft to
be validated, and re-running stages costs real allocation. Read `README.md` for
what the scripts are and
[`../scientific_report.md`](../scientific_report.md)
for what they found.

> An earlier version of this file told the reader that nothing had been tested
> and to stop for confirmation at every step. That was correct in August 2026
> and is now actively misleading.

## Before anything else

- **Read `.agent/CONTINUITY.md`.** It is the working state: current numbers,
  superseded results, and the traps that have already cost time.
- The git repository is the **nested** `Projects/genome_entropy/genome_entropy`.
  The outer directory has no `.git`, and `~/Projects/genome_entropy` is a
  symlink to the same place.
- Check the environment **by import, not by version string** —
  `genome_entropy` shipped 0.2.0 twice with different code:

  ```bash
  python3 -c "from genome_entropy.io.genbank import normalise_orf_interval"
  ```

  A stage run without that helper reproduces the coordinate defect it was
  corrected for.

## Cost discipline

Encoding cost ~84,000 SU. Everything else is small by comparison, but not free:
a full Foldseek search over 1.5 M queries is ~950 SU and the GTDB Prodigal
download is ~27 SU for 1 h 41 m on `copyq`.

- Never re-run `04_run_entropy.pbs` over a whole domain to fix something
  downstream. The archives and `entropy_rows/` are the durable products.
- Smoke-test **the exact invocation the job will make**, not a variant. A driver
  once staged to `<tag>.tsv.gz.partial` while the writer chose gzip from the
  `.gz` suffix — the smoke test used the final name, so 760 workers wrote plain
  text and 61.57 SU was lost.
- Submit through the scheduler. Anything that reads a chunk TSV or a coordinate
  table belongs in a job, not on a login node.

## What to be careful about

**Reading-frame comparisons.** A plus-strand feature's frame is `g_start % 3`; a
minus-strand feature's is `g_end % 3`. Anchoring on `g_start` for both is
correct on plus and wrong on minus. It is nearly invisible because the two
anchors agree whenever both lengths divide by three, which holds for 100% of
complete ORFs and 95.6% of CDS parts — it fires on **partial CDS**, where no
anchor is meaningful. `20_orf_context.py::frame_class` is the single authority;
do not recompute it elsewhere.

**Control arms.** A same-frame shadow *is* the annotated protein and must be
excluded from any comparator. Consumers select "clean" by allow-list
(`{"opposite strand", "same strand, frameshift"}`), which is what lets a new
frame class be excluded automatically rather than silently included.

**Counts that look alike.** `candidate_burden_*.tsv`'s `n_cds` is *ORFs matching
a CDS*; `genome_cds_counts_*.tsv`'s `n_cds` is *deposited CDS features*. They
differ by a median 161 per archaeal genome. Differencing against the wrong one
produced a published figure that was 6× too large.

**Verification that can destroy its own output.** Publish first, compute
statistics afterwards, and make the statistics non-fatal.

## Reporting

- Post meaningful results to the relevant GitHub issue (**#92** for the
  validation run, **#97** for the missed-gene analysis) and **mention
  `@linsalrob`** so the notification fires.
- `gh` is not on `PATH`: `/g/data/ob80/re3494/conda/gh/bin/gh`.
- Report numbers with their controls. A candidate count without its matched
  shadow background is not a result.
- When a published number turns out to be wrong, correct it on the issue
  explicitly rather than quietly replacing it.

## What is still open

`scientific_report.md` §11 carries the live list. The two standing items are the
disagreement between the structural mixture estimate (~128,000–149,000 missed
ORFs) and the Prodigal-based one (~244,000), and testing what the 3Di state `D`
corresponds to. Both are characterisation rather than blockers.
