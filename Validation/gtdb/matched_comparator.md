# §6.1 clean-shadow comparator, re-drawn matched (Gadi, 2026-10-14)

Follow-up to [`supplementary_analyses.md`](supplementary_analyses.md) §8 and
§12 item 1. It does **not** edit `scientific_report.md`. Result files are in
[`matched_comparator/`](matched_comparator/); scripts `39`–`42` are in
`scripts/` (see the scripts README).

## Why

The §6.1 clean comparator is the antisense + same-strand-frameshift subset of
`shadow_hi`. `shadow_hi` was matched 1:1 to the candidates on length and 3Di
*before* the frame filter, so the clean subset is no longer matched.
Setonix's diagnosis reproduces exactly on the Gadi tables:

| | n | median aa | median 3Di | in candidates' top 3Di quintile | contig-truncated | 5×5 TVD vs candidates |
|---|---:|---:|---:|---:|---:|---:|
| bac candidates | 523,346 | 134 | 2.828 | 20.0% | 7.6% | 0 |
| bac published clean | 292,527 | 122 | 2.677 | 4.6% | 2.4% | 0.249 |
| **bac matched clean (new)** | 523,346 | 134 | 2.828 | 19.9% | 7.6% | **0.003** |
| arc candidates | 22,447 | 140 | 2.897 | 20.0% | 11.6% | 0 |
| arc published clean | 13,945 | 120 | 2.745 | 9.5% | 9.7% | 0.212 |
| **arc matched clean (new)** | 22,447 | 140 | 2.895 | 19.5% | 11.6% | **0.010** |

## What was done

1. **Full shadow pool** (`39_shadow_pool.py`). Re-ran the `10_missed_genes.py`
   classification over all 760 + 41 chunks (functions imported, not copied).
   Every chunk reproduced the published stats exactly, and the job failed
   on any difference. Every `shadow_hi` ORF was kept and given
   `20_orf_context.py`'s frame class. On a test chunk the frame classes were
   identical to the published context tables for 221/221 shadows.
   - Bacteria: 9,075,154 shadows, of which 3,925,368 are clean (34.4%
     antisense, 8.9% frameshift). The rest are 40.5% frame-undefined and
     16.3% same-frame.
   - Archaea: 285,470 shadows, 133,748 of them clean.
2. **Matching** (`40_match_clean_comparator.py`, seed 0). One clean partner
   per candidate, without replacement. A partner must have the same
   truncation status, 3Di within ±0.025 bits, and length within ±10% (log
   scale). Preference goes to the same genome, then the same chunk, then
   anywhere in the domain. Unmatched candidates were retried at 2× and 4×
   the calipers, then given their nearest partner *with* replacement.
   Candidates were processed scarcest-first (descending 3Di).
   - Bacteria: 36.7% same genome, 49.6% same chunk, 13.5% domain, 0.03%
     widened, **0.16% with replacement** (817).
   - Archaea: 26.5 / 50.0 / 17.9 / 1.2%, **4.3% with replacement** (972).
     These are all in the high-3Di, long tail (median 3Di 3.62, 307 aa),
     where the archaeal clean pool runs out.
   - Achieved |Δ3Di|: median 0.004 bits, p99 0.025. |Δlen|/len: median
     1.5%, p99 9.8% (bacteria).
3. **Search**. 62,078 bacterial and 7,911 archaeal comparator ORFs had
   already been searched as `shadow_hi`, and their hits were reused. The
   other 460,451 + 13,564 went through `14b` → `15` → `17`. The search
   used the same Foldseek 10.941cd33 binary, the same four databases and
   the same parameters (`-e 0.001`, struct `--alignment-type 2
   --sort-by-structure-bits 0`, plus `base:search`). The new opt-in
   `NULL=0` skips the candidate-only shuffled null.
   - Two 60-ORF smoke tests ran first. The second re-searched 60 published
     archaeal shadows and reproduced their published best hits
     byte-for-byte in all 8 tables.
   - `42_merge_best.py` assembled `best_v7/`: published rows for the
     retained queries plus the new rows. It fails on any query present in
     both.
4. **Analysis**. `18_pilot_analysis.py` (→ `analysis_v7_*`),
   `22_rank_candidates.py` (→ `ranked6_*`) and `24_prodigal_overlap.py`
   (→ `prodigal_v7_*`) ran **unchanged** on `wanted_v7.tsv` and
   `context_v7/`. In those files the `shadow_hi` arm is now the matched
   clean comparator.
   - `41_comparator_summary.py` recomputes the same quantities with the
     comparator weighted by draw multiplicity, which is correct when some
     draws are with replacement. `18`/`22`/`24` count each ORF once.
   - Run against the published clean arm, `41` reproduces every published
     v6 / ranked5 / prodigal number, and Setonix's re-weighted shares.

Total: **394 SU** (ceiling 800). The four bacterial search shards cost
292 SU of that. All 30 jobs exited 0. Job IDs are in the continuity notes.

## Result

Real-gene share = (candidate − comparator) / (annotated − comparator), on
full-length (qcov and tcov ≥ 0.8) struct hits, with a 95% CI. "Matched" is the
weighted estimate from `41`.

| | published v6 | Setonix re-weighted | **matched (new)** |
|---|---:|---:|---:|
| **Bacteria** Swiss-Prot | 23.3% (23.0–23.7) | 9.0% (8.4–9.6) | **10.4% (10.0–10.7)** |
| PDB100 | 23.4% (23.1–23.8) | 8.2% (7.4–8.9) | **9.0% (8.6–9.5)** |
| CATH50 | 27.0% (26.5–27.5) | 10.7% (9.7–11.6) | **9.9% (9.3–10.5)** |
| any SP/PDB/CATH | 31.4% | 13.2% | **13.9% (13.4–14.3)** |
| structural estimate | 122k–141k | 43k–56k | **47.3k–54.2k** |
| C3 excess | 62,507 | 25,759 | **25,620** |
| C7 excess | 19,232 | 5,835 | **4,974** |
| Prodigal share | 44.5% (ilo bg) / 44.8% (clean) | 36.9% | **36.4% (36.2–36.6)** |
| Prodigal ÷ structural | ~1.6–1.9× | ~3.4–4.5× | **3.5–4.0×** |
| **Archaea** Swiss-Prot | 26.2% (24.3–28.2) | 10.5% (7.1–13.3) | **15.6% (13.4–17.9)** |
| PDB100 | 28.3% (26.3–30.4) | 14.3% | **16.2% (13.7–18.7)** |
| CATH50 | 36.4% (33.7–39.2) | 21.3% | **22.0% (18.5–25.4)** |
| any SP/PDB/CATH | 37.2% | 19.1% | **23.7% (21.1–26.2)** |
| structural estimate | 5.9k–8.2k | 2.4k–4.8k | **3.5k–4.9k** |
| C3 excess | 3,256 | 2,189 | **1,731** |
| C7 excess | 1,118 | 684 | **407** |
| Prodigal share | 49.1% | 42.8% (41.7–43.9) | **42.2% (41.1–43.2)** |
| Prodigal ÷ structural | ~1.3–1.9× | ~2.0–4.1× | **1.9–2.7×** |
| **Both domains** structural estimate | ~128k–149k | ~45k–60k | **~51k–59k** |
| C3 floor (any full-length) | ~65,800 | ~28,000 | **~27,400** |
| C7 floor (strict) | ~20,350 | ~6,500 | **~5,400** |

- Unweighted (`18`/`22` as run, each comparator ORF counted once).
  - Bacteria: Swiss-Prot 10.6%, PDB 9.3%, CATH 10.0%; C3 26,051, C7
    5,334.
  - Archaea: 19.8 / 21.2 / 26.8%; C3 2,152, C7 748.
- Sensitivity: drop the with-replacement candidates and their draws.
  - Bacteria: essentially unchanged (10.3%).
  - Archaea: Swiss-Prot 14.0%, CATH 21.7%.
- C2 (not truncated) now has zero excess by construction. The published
  −26,768 was an artefact of the truncation imbalance, so C2 is a filter,
  not evidence.
- Prodigal background: the matched clean comparator coincides with a
  Prodigal call 20.6% of the time (bac) and 25.6% (arc). The published clean
  arm was 9.4% / 16.0%. The report's "take the larger background" rule now
  selects the clean comparator in both domains, not `intergenic_lo`
  (9.85% / 15.4%).

## Reading

1. **Setonix's correction is confirmed for bacteria.** A properly matched
   comparator gives a 9–10% real-gene share against the published 23–27%,
   within about 1 point of the re-weighted estimate.
   - Per 3Di quintile, the matched and published clean arms hit at nearly
     the same rate (Swiss-Prot full-length, Q1–Q5: 0.5 / 0.9 / 1.9 / 6.9
     / 27.1% vs 0.4 / 0.7 / 1.7 / 5.7 / 28.3%;
     `quintile_diagnostic_bac.txt`). So the published arm's problem was its
     composition, not its per-cell rates.
2. **Archaea land between crude and re-weighted** (15.6% against 26.2% and
   10.5%). The archaeal clean pool is thin in the high-3Di, long tail where
   real-gene signal is concentrated: 972 candidates needed reuse, and 454
   of those candidates meet C7. The archaeal share therefore depends on
   the estimator, ranging 14.0–19.8% across the weighted, unweighted and
   tail-excluded variants. Quote it as a range.
3. **Headline**: the structural estimate is **~51k–59k** (not ~128k–149k).
   The direct-evidence floors are **~27,400** (any full-length) and
   **~5,400** (strict), not ~65,800 and ~20,350. The structural and
   Prodigal estimates now differ by ~3.5–4× in bacteria, not ~1.7×.

## Caveats

- **Some clean shadows may be real genes.** Real overlapping, antisense or
  misannotated genes would inflate the comparator and *under*state the
  share. The matched clean shadows that reach C7 (18,566 scaled, bac) carry
  real-looking homology. The comparator bounds "protein-like non-gene" only
  as far as antisense/frameshift ORFs are non-genes.
- The 5×5 standardisation and the matched draw agree in bacteria, but both
  rest on the same frame classes. Frame-undefined shadows (40% of the
  bacterial pool) are excluded, not resolved.
- Matching is domain-wide for 13.5% (bac) and 17.9% (arc) of candidates,
  so lineage is matched less tightly than in the original same-genome
  design. Those partners are mostly high-3Di, and their high hit rates
  (~23%) reflect 3Di, not genome.
- Archaeal tail sparsity (above).
- CIs are binomial/delta-method only. They do not cover the two-component
  mixture assumption.

## Files

On `/g/data/ob80/re3494/gtdb_entropy/missed_genes/`:

- `matched_clean/{bac,arc}/`:
  - `pool/` (full shadow pool);
  - `comparator.tsv.gz`, `wanted_v7.tsv`, `to_search*.tsv`;
  - `seqs*/`, `search*/`, `best_v7/`, `context_v7/`.
- `matched_clean/summary/`.
- `full_bac/{analysis_v7_bac*,ranked6_*,prodigal_v7_bac*}`.
- `pilot_arc/{analysis_v7_arc*,ranked6_*,prodigal_v7_arc*}`.

No v6 / ranked5 / prodigal_*4 output was touched.
