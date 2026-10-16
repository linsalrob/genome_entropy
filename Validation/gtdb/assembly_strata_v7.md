# v7 matched-comparator estimate by assembly quality (Gadi, 2026-10-16)

This is report §11 item 12, carrying out decision D15 from agent-handoffs#24. It replaces the pre-matching strata of [`supplementary_analyses.md`](supplementary_analyses.md) §7 as the source for anything §6.1 says about assembly quality.

## Method

- **Script:** `scripts/43_assembly_strata_v7.py`, run by `43_assembly_strata_v7.pbs` (PBS 181378085, 0.33 SU, exit 0).
- **Inputs:** only existing v7 tables, with no new search:
  - `matched_clean/<dom>/{wanted_v7.tsv, comparator.tsv.gz, best_v7/, context_v7/}`;
  - `prodigal_v7_<dom>.coincidence.tsv.gz`;
  - `candidate_burden_<dom>.tsv` (GTDB r232 contig N50 and CheckM2 completeness).
- **Same computation as the published numbers.** Within each stratum the script runs exactly what `41_comparator_summary.py` runs:
  - the comparator weighted by draw multiplicity;
  - the first-order CI from its `share()`;
  - the #97 ladder from `22_rank_candidates.tiers()`;
  - the Prodigal share against the matched comparator.
- **Reproduction check.** An unstratified "all" row must reproduce `matched_comparator/<dom>_v7_matched.tsv`, and the script exits otherwise. It reproduced **51 of 51 metrics in both domains**.
- **Quartile edges.** They are those of §7: `pd.qcut` over the pooled query rows of the v6 query set (`34_supplementary_arms.py`, `t_assembly`, scheme `pooled_query_rows`). Pre- and post-matching tables therefore share bins:
  - bacteria N50 at 17.2 kb, 59.1 kb and 262 kb; completeness at 89.81%, 96.75% and 99.99%;
  - archaea N50 at 13.7 kb, 39.0 kb and 222 kb; completeness at 83.22%, 92.28% and 99.17%.
- **Stratum assignment (D15).** In both schemes annotated CDS, which have no partner, are placed by their own genome.
  - **(i) candidate genome, the primary analysis.** Each comparator draw goes in its candidate's genome quartile, so matched pairs stay together and the length × 3Di × truncation matching holds within every stratum.
  - **(ii) own genome, the sensitivity analysis.** Every ORF, comparator draws included, is placed by its own genome.
- **Outputs:** `matched_comparator/assembly_strata_v7_{bac,arc}.tsv`, a long table with columns `scheme, covariate, stratum, cov_lo, cov_hi, metric, value, lo, hi`. It holds every 41 metric per stratum, plus `n_comparator_with_replacement_draws`.

## Results, primary scheme (i)

Shares are implied real-gene shares, (candidate − comparator) / (annotated − comparator), on full-length (qcov and tcov ≥ 0.8) structural hits, in % with a 95% CI. "Any of three" means Swiss-Prot, PDB100 or CATH50. "Prodigal" is the Prodigal share against the matched comparator.

**Bacteria, by contig N50**

| quartile | n cand | Swiss-Prot | PDB100 | CATH50 | any of three | C3 excess | C7 excess | Prodigal | repl. draws |
|---|---:|---|---|---|---|---:|---:|---|---:|
| Q1 (< 17.2 kb) | 121,119 | 9.8 (8.7–10.8) | 11.3 (10.2–12.5) | 14.8 (13.3–16.3) | 17.4 (16.2–18.6) | 5,203 | 419 | 44.8 (44.3–45.3) | 225 |
| Q2 | 130,297 | 12.1 (11.3–13.0) | 11.5 (10.6–12.4) | 13.0 (11.8–14.2) | 16.8 (15.9–17.8) | 7,087 | 1,691 | 39.2 (38.7–39.6) | 263 |
| Q3 | 132,348 | 10.3 (9.6–11.0) | 7.7 (7.0–8.4) | 7.6 (6.6–8.6) | 12.4 (11.6–13.2) | 6,875 | 1,364 | 32.8 (32.4–33.2) | 129 |
| Q4 (≥ 262 kb) | 139,582 | 9.2 (8.6–9.9) | 6.5 (5.8–7.2) | 5.8 (4.8–6.8) | 10.3 (9.6–11.0) | 6,455 | 1,500 | 31.1 (30.7–31.5) | 200 |

**Bacteria, by CheckM2 completeness**

| quartile | n cand | Swiss-Prot | PDB100 | CATH50 | any of three | C3 excess | C7 excess | Prodigal | repl. draws |
|---|---:|---|---|---|---|---:|---:|---|---:|
| Q1 (< 89.8%) | 121,470 | 14.2 (13.2–15.2) | 14.4 (13.3–15.5) | 16.1 (14.6–17.6) | 19.9 (18.7–21.1) | 6,392 | 1,585 | 43.2 (42.7–43.7) | 326 |
| Q2 | 130,015 | 12.1 (11.3–13.0) | 11.5 (10.5–12.4) | 13.7 (12.5–15.0) | 17.4 (16.4–18.4) | 6,965 | 1,503 | 41.0 (40.5–41.4) | 250 |
| Q3 | 154,676 | 8.6 (8.0–9.2) | 6.7 (6.0–7.3) | 7.3 (6.4–8.2) | 11.3 (10.6–12.0) | 7,241 | 1,123 | 34.3 (33.9–34.7) | 146 |
| Q4 (≥ 99.99%) | 117,185 | 7.8 (7.2–8.4) | 5.1 (4.3–5.8) | 4.0 (3.0–5.1) | 9.0 (8.2–9.7) | 5,022 | 763 | 28.7 (28.3–29.1) | 95 |

**Archaea, by contig N50**

| quartile | n cand | Swiss-Prot | PDB100 | CATH50 | any of three | C3 excess | C7 excess | Prodigal | repl. draws |
|---|---:|---|---|---|---|---:|---:|---|---:|
| Q1 (< 13.7 kb) | 5,863 | 13.5 (8.3–18.6) | 15.6 (9.9–21.3) | 26.7 (18.9–34.6) | 26.3 (20.3–32.4) | 351 | 46 | 47.0 (44.8–49.1) | 294 |
| Q2 | 5,610 | 21.2 (16.5–25.8) | 24.1 (18.9–29.2) | 31.2 (24.0–38.4) | 31.6 (26.4–36.8) | 513 | 196 | 48.5 (46.3–50.7) | 329 |
| Q3 | 5,451 | 14.1 (9.7–18.5) | 14.0 (9.3–18.7) | 17.5 (11.0–24.1) | 20.3 (15.6–25.1) | 417 | 81 | 42.2 (40.0–44.3) | 236 |
| Q4 (≥ 222 kb) | 5,523 | 16.0 (11.7–20.3) | 12.9 (8.4–17.3) | 15.0 (8.8–21.2) | 20.3 (15.5–25.1) | 450 | 84 | 32.6 (30.8–34.4) | 113 |

**Archaea, by CheckM2 completeness**

| quartile | n cand | Swiss-Prot | PDB100 | CATH50 | any of three | C3 excess | C7 excess | Prodigal | repl. draws |
|---|---:|---|---|---|---|---:|---:|---|---:|
| Q1 (< 83.2%) | 4,962 | 12.9 (8.0–17.7) | 13.4 (8.0–18.8) | 27.6 (20.3–34.9) | 24.1 (18.6–29.6) | 355 | 44 | 46.2 (43.8–48.5) | 240 |
| Q2 | 5,782 | 19.6 (14.7–24.5) | 20.3 (15.1–25.6) | 28.3 (20.8–35.9) | 28.6 (23.1–34.1) | 438 | 122 | 49.5 (47.3–51.6) | 324 |
| Q3 | 6,056 | 18.6 (14.2–23.0) | 20.9 (16.1–25.7) | 22.3 (15.7–28.8) | 27.1 (22.2–32.0) | 525 | 178 | 42.0 (40.0–44.1) | 288 |
| Q4 (≥ 99.2%) | 5,647 | 14.6 (9.8–19.4) | 12.3 (7.2–17.3) | 13.4 (6.6–20.2) | 20.2 (14.7–25.7) | 413 | 63 | 33.1 (31.3–35.0) | 120 |

The full sensitivity tables, scheme (ii), are in the TSVs.

## Reading

1. **The excess is present in every stratum.**
   - Every share is above zero in every quartile, in both domains, under both assignments and for every database. The smallest lower CI bound is 3.0% under (i) (bacteria, CATH50, completeness Q4) and 3.4% under (ii) (archaea, Swiss-Prot, completeness Q1).
   - The C3 and C7 excesses are positive in every stratum. The smallest C7 excess is 419 in bacteria and 44 in archaea under (i); in archaea it falls to 11 under (ii).
   - The candidate signal is therefore not confined to any range of assembly quality, and in particular it is not concentrated in the best assemblies.
2. **Bacteria: on the combined readout the share is higher in poorer assemblies.**
   - Any of three: 17.4% → 10.3% from the lowest to the highest N50 quartile, and 19.9% → 9.0% across completeness quartiles, with non-overlapping CIs at the ends.
   - The trend comes mostly from PDB100 and CATH50.
   - Swiss-Prot shows no monotonic trend with N50 (9.2–12.1%), but declines with completeness (14.2% → 7.8%).
3. **Archaea: no monotonic trend.** Q2 is highest on most readouts, and the CIs (about ±5 points) cover a flat profile.
4. **The Prodigal share falls steeply with assembly quality in both domains.**
   - Bacteria: 44.8% → 31.1% by N50 and 43.2% → 28.7% by completeness.
   - Archaea: 47.0% → 32.6% and 46.2% → 33.1%.
   - It is the most precise readout, and it moves the most.
5. **Comparison with the pre-matching strata (§7).**
   - The bacterial Swiss-Prot decline with N50 (28.1 / 26.0 / 20.0 / 19.6%) **does not survive** re-matching: it becomes 9.8 / 12.1 / 10.3 / 9.2%. Most of that apparent trend was the comparator imbalance.
   - The decline with completeness survives at about half the magnitude: 32.3% → 16.8% before, 14.2% → 7.8% after.
   - The archaeal Q2 peak is present in both.
6. **The two assignments agree on the direction of the combined readout and the Prodigal share, not on every database or cell.**
   - The Q1 → Q4 direction of the combined Swiss-Prot/PDB/CATH share and of the Prodigal share is the same under both schemes, in both domains and for both covariates.
   - Two single-database readouts that have no trend under either scheme flip endpoint direction: bacterial Swiss-Prot by N50 (9.8 → 9.2% under (i), 10.5 → 11.3% under (ii)) and archaeal PDB100 by completeness (13.4 → 12.3% under (i), 11.1 → 11.1% under (ii)).
   - Bacteria: shares differ by up to 3.3 points (mean 1.6), and 9 of 32 share cells have non-overlapping CIs between schemes.
   - Archaea: up to 6.9 points (CATH50), with no non-overlapping cells.
   - Individual stratum values depend on how cross-genome draws are assigned, so quote scheme (i). The combined and Prodigal trends do not depend on it.
7. **With-replacement draws are concentrated in poorer archaeal assemblies** (294 / 329 / 236 / 113 across N50 quartiles under (i)). Archaeal Q1–Q3 estimates lean more on reused comparator ORFs than Q4. In bacteria the count is at most 326 per stratum (under 0.3%).

8. **The Prodigal shares are the larger-background shares in every stratum.** The report's rule (as in `41_comparator_summary.py`) takes the larger of two backgrounds: the matched comparator and `intergenic_lo`. In all 32 strata (2 domains × 2 covariates × 2 schemes × 4 quartiles) the matched comparator is the larger. Per-stratum `intergenic_lo` rates are 8.2–13.7% in bacteria and 13.0–18.9% in archaea, against comparator rates of 13.8–27.1% and 15.6–35.0%. So the Prodigal shares above are the report-rule values, and the trend stands. This was checked on Setonix from the `intergenic_lo` arm of `prodigal_{bac,arc}4.coincidence.tsv.gz`, which the v7 re-matching left unchanged: its pooled rate reproduces the published v7 values (9.8508% and 15.4408%). `43_assembly_strata_v7.py` now emits `prodigal_intergenic_lo`, `prodigal_share_intergenic_lo_bg` and `prodigal_share_larger_bg` per stratum. The TSVs were regenerated on Gadi from the v7 Prodigal table (PBS 181387980): the reproduction check covers 54 metrics in both domains, none of the 884 earlier values per domain changed, and the comparator is the larger background in all 32 strata, with the ranges given above.

## What it does and does not establish

- **Established:** the matched excess, and therefore the pooled §6.1 estimate, is not an artefact confined to fragmented or incomplete assemblies. The excess persists in the highest N50 and completeness quartiles (bacteria 10.3% and 9.0% on the combined readout).
- **Not established: why bacterial shares and Prodigal agreement are higher in poorer assemblies.** These data cannot separate two explanations:
  - **assembly-driven false signal,** where fragmentation or incompleteness inflates the candidate rate beyond what the matched comparator captures;
  - **genuine annotation gaps,** where lower-quality genomes carry less complete GenBank annotation, so more real genes are missed and found.

  Separating them would need, for example, the GTDB annotation-provenance stratification in §11 item 9.
