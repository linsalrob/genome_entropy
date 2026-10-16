# Open scientific and editorial decisions

Decisions that belong to Rob, collected while reconciling the manuscript with
the finished analyses ([agent-handoffs#24](https://github.com/linsalrob/agent-handoffs/issues/24)).
Agents do not choose between these silently. Each entry gives the options, the
evidence, and a recommendation. Record the decision under **Decision** and
move the entry to *Decided* at the bottom.

Status as of 2026-10-16 (integrator, Setonix). **All recommendations D1–D14 accepted by Rob on 2026-10-16**; D15 decided separately.

| # | topic | blocks | recommendation |
|---|---|---|---|
| D1 | Headline GTDB missed-gene number | GTDB Results, Abstract | structural ~51–59k as the headline, Prodigal ~200k as the upper bracket |
| D2 | Prodigal vs structure gap (3.5–4×) | GTDB Results, Discussion | state as a limitation; no new compute |
| D3 | Paper scope: one paper or several | everything | one paper, smORFs as supplementary |
| D4 | Main-figure sequence | Results order | the six-figure plan in `figure_plan.md` |
| D5 | Which phage run and ORF set | phage Methods/Results | Setonix MI (v2) + ranking (corrected); Gadi run for the bacteria comparison and determinism |
| D6 | Phage 90-aa floor | phage Results | state as a limitation |
| D7 | OM869708 worked example | Results, figure | keep, with corrected-model numbers |
| D8 | "Neither entropy alone is sufficient" | Abstract, Results | replace: 3Di carries most of the signal |
| D9 | Mixed encoding configurations across datasets | Methods | report per dataset, with the determinism test |
| D10 | Which software version the paper cites | Methods, Data availability | cite each dataset's version; release current `main` for the paper |
| D11 | Untested mechanistic claims (D state, phage 3Di gap, smORF stripes) | Results wording | soften to observations; no new compute |
| D12 | Unsupported text claims (FLOPs, Phynteny, GenBank-vs-LOGAN agreement) | Results, Discussion | remove unless a source is supplied |
| D13 | LOGAN follow-ups (MetaEuk, single-mode Prodigal, clustering) | Discussion | future work |
| D14 | Data deposition (ROADS) | Data availability | Gadi drafts `roads_deposition_candidates.md`; Rob arranges DOIs |
| D15 | How to stratify the v7 comparison by assembly quality | G-17, GTDB Results | **closed:** done in #134; see *Decided* |

---

## D1. Headline GTDB missed-gene number

**Evidence.** The matched clean comparator (genome_entropy #109, v7) gives:
- a real-gene share of 9–10% for bacteria and 16–22% for archaea (14.0–19.8% depending on estimator);
- a structural estimate of **~51,000–59,000** missed genes;
- direct-evidence floors of **~27,400** (any full-length hit) and **~5,400** (strict rule).

Independent gene calling (GTDB Prodigal) implies **~200,000**. The v6 figure of ~128–149k is superseded. Sources: `Validation/gtdb/scientific_report.md` §1, §6.1; `matched_comparator.md`.

**Options.**
- (a) Structural only.
- (b) Structural as the headline, with Prodigal as an upper bracket.
- (c) The floor (~5,400) as the headline.

**Recommendation.** (b). Quote ~51–59k and the ~5,400 strict floor, and give Prodigal's ~200k as the bracket, explaining why the two differ. Report archaea as a range.

**Decision.** 2026-10-16, Rob: recommendation accepted.

## D2. The Prodigal vs structure disagreement

**Evidence.** Prodigal agrees with 47.4% of bacterial candidates, against 20.6% for matched shadows. That implies about 3.5–4× more real genes than the structural assay, and the gap widened when the comparator was matched. The report gives three explanations it cannot separate (§6.1, "Where the two estimates disagree"):
- Prodigal over-calls;
- the structural mixture under-counts;
- both.

The strict set is 91.8% Prodigal-confirmed.

**Options.**
- (a) State it as an unresolved limitation.
- (b) Commission a Prodigal false-positive test on non-coding DNA before submission.

**Recommendation.** (a). #24 asks for synthesis, not new compute, and the bracket in D1 already carries the uncertainty.

**Decision.** 2026-10-16, Rob: recommendation accepted.

## D3. Paper scope

The canonical `genome_entropy.md` currently covers bacterial genomes and phage (the OM869708 example), plus LOGAN. Three other complete drafts exist in agent-handoffs:
- a stand-alone LOGAN draft (`projects/genome_entropy_logan/manuscript/manuscript.md`);
- a GMSC smORF draft (`smorfs/manuscript.md`);
- the GTDB report, which the canonical manuscript does not cover yet.

**Options.**
- (a) **One paper:**
  - GTDB as the reference population;
  - LOGAN as the metagenome scale-up;
  - phage as the annotation-error use case;
  - smORFs as a supplementary limit-of-method result.
- (b) Split LOGAN and/or smORFs into separate papers.

**Recommendation.** (a). It matches the manuscript's current title, and the smORF result (AUC 0.73; 3Di near-uninformative below 30 aa) is best presented as the method's lower length limit.

**Decision.** 2026-10-16, Rob: recommendation accepted.

## D4. Main-figure sequence

See [`figure_plan.md`](figure_plan.md). Six main figures:
1. GTDB landscape;
2. length and what structure adds;
3. six-frame specificity;
4. GTDB missed genes;
5. LOGAN candidates;
6. phage.

All other panels are supplementary.

**Decision.** 2026-10-16, Rob: recommendation accepted.

## D5. Which phage run and ORF set

**Evidence.** There are three phage runs:

| run | genomes | ORFs | package | encoding | hardware | labels |
|---|---|---|---|---|---|---|
| Setonix MI (#108) | 36,025 | 6,754,361 | ≈0.2.0 | 20k | ROCm fp16 | v2 |
| Setonix ranking (#110) | 31,200 | 5,887,690 | v0.1.15 ORFs | 20k | ROCm | relabelled |
| Gadi (`agent-handoffs/.../inphared_gadi/`) | 36,025 | 6,754,361 | 0.3.0 | 10k | V100 | current |

- The ranking set lacks 49 of 361 JSON parts.
- On the same ORFs, the two labelling routes agree on 5,887,451 of 5,887,690.

**Recommendation.**
- Report MI and ranking from the Setonix runs, as merged.
- State in the Methods that the ranking covers 31,200 of 36,025 genomes, and why.
- Use the Gadi run for the phage-vs-bacteria entropy comparison and the determinism test.

The alternative, re-running ranking on the full Gadi set, is new compute.

**Decision.** 2026-10-16, Rob: recommendation accepted.

## D6. Phage 90-aa ORF floor

`get_orfs -l` was in amino acids (#103/#104), so the phage, MI and GTDB runs have an effective 90-aa floor. That makes 28.8% of annotated phage genes invisible. LOGAN used a true 30-aa floor.

**Options.**
- (a) State it as a limitation.
- (b) Regenerate phage ORFs at `-l 29` and re-run.

**Recommendation.** (a). Say it plainly in the Methods for every affected dataset.

**Decision.** 2026-10-16, Rob: recommendation accepted.

## D7. OM869708 worked example

**Checked 2026-10-16** against the corrected phage model (`work/phage_rank/rank_orfs_relabelled_intrinsic`, Setonix):
- **The example stands.** All OM869708 genes are on the plus strand, so the strand bug does not affect it.
- **ORF3** (+, frame 1, 4722–5730, 335 aa, 3Di 2.98 bits) scores **p = 0.889** with the corrected intrinsic model, against 0.947 legacy.
- **The annotated ORF7 (CDS_0007)** scores **p = 0.175**, against 0.199 legacy.
- **Ranking:** ORF3 → ORF7 is replacement **179 of 4,102** in the corrected list, against 121 of 1,797 legacy.

The manuscript's 0.91 and 0.27 match neither model. They likely come from the July notebook model.

**BLASTP re-run and the original record (2026-10-16; #122, `figP01_om869708_*`).** These change the story:
- **OM869708 is *Dipodfec virus* RodF1_38.** The submitters' GenBank record annotates CDS 4753..5730 as a replication initiator protein (UPW42009.1). That is ORF3, from its first methionine, with the same stop codon.
- **The Pharokka + phold re-annotation dropped that gene.** It called a hypothetical protein in another frame instead (CDS_0007, which is ORF7).
- **The manuscript's "best hit to the *Dipodfec virus* RodF1 38 replication initiator" is the genome's own record** (100% identity), so it is not independent support.
- **Independent support exists:** the next hits are other microvirus replication initiators, at 43–55% identity (E down to 2 × 10⁻¹²⁷).
- **ORF7 has no hit in nr at E < 10⁻⁵.**

**Still not found in any repo:**
- the genbank_viewer figure (now replaced by Fig. P01d);
- the "Phynteny" step.

**Recommendation.**
- Keep the example, reframed: genome_entropy recovers a replication gene that the submitters annotated and that the re-annotation pipeline lost.
- Use the corrected-model probabilities, and cite only the non-self homologues as independent evidence.
- Drop "Phynteny" unless Rob confirms it was run.

**Decision.** 2026-10-16, Rob: recommendation accepted.

## D8. "Neither amino-acid nor 3Di entropy alone is sufficient"

The manuscript says this at line ~306 and in the Abstract. Every dataset contradicts it:
- **GTDB:** the 3Di gap is 2.08 bits, against 0.39 for protein entropy.
- **Phage:** 3Di alone carries 0.587 bits (AUC 0.947), and the pair adds only 0.057 bits.
- **LOGAN:** within 100–149 aa, 3Di AUC is 0.896 against 0.805 for aa.

**Recommendation.** Reframe: 3Di entropy carries most of the signal. Amino-acid entropy adds little beyond it, and length explains much of what amino-acid entropy appears to show (PR #101).

**Decision.** 2026-10-16, Rob: recommendation accepted.

## D9. Mixed encoding configurations

The datasets were encoded with different package versions, encoding sizes and hardware:

| dataset | package | encoding | hardware |
|---|---|---|---|
| GTDB | 0.2.0 | 10k | V100 |
| LOGAN | 0.3.0 | 10k | MI250X |
| smORFs | 0.3.0 | 50k | V100 |
| phage | ≈0.2.0 / 0.3.0 | 20k / 10k | ROCm / V100 |

The determinism test (Gadi PBS 181181761) shows:
- repeat runs are bit-identical;
- encoding size moves 0.3–0.7% of ORFs by less than 0.08 bits;
- Gadi and Setonix differ by p99 0.31 bits per ORF, with class means moving about 0.01 bits.

**Recommendation.**
- Report the configuration per dataset in a Methods table.
- State that population conclusions are unaffected but per-ORF values are not comparable bit-for-bit across hardware.
- Don't re-encode.

**Decision.** 2026-10-16, Rob: recommendation accepted.

## D10. Software version to cite

The runs used 0.1.15 (phage ranking ORFs), 0.2.0 (GTDB, phage MI) and 0.3.0 (LOGAN, smORFs, phage Gadi). `main` is now 0.4.0 (#112, `--max-aa`). `citation.cff` still says 0.1.1. A Zenodo DOI exists (10.5281/zenodo.18312207).

**Recommendation.**
- Cite each dataset's version in the Methods table (D9).
- Tag and archive a release of `main` for the paper, and update `citation.cff`.

The release is Rob's call (an outward-facing action).

**Decision.** 2026-10-16, Rob: recommendation accepted.

## D11. Untested mechanistic claims

Three claims rest on observation, not tests:
- **3Di state D as disorder or coil:** untested; the D/V/P composition comes from 3 genomes.
- **Phage 3Di sitting 0.20–0.34 bits below bacteria:** cause untested.
- **smORF L34/S14 "stripe" families:** identified by eye.

**Recommendation.** Word each as an observation, with the test named as future work. No new compute.

**Decision.** 2026-10-16, Rob: recommendation accepted.

## D12. Unsupported claims in the current text

No source was found in either repo, agent-handoffs or Teams for:
- "~5.7 × 10²⁰ floating-point operations" (LOGAN computational scale);
- the Phynteny annotation step (phage Methods);
- "strong agreement between GenBank-annotated proteins and the majority of LOGAN ORFs" (Discussion);
- "[Baktfold paper]" (a placeholder citation).

**Recommendation.** Remove each one unless Rob supplies the source.

**Decision.** 2026-10-16, Rob: recommendation accepted.

## D13. LOGAN follow-ups

The LOGAN draft lists three possible follow-ups:
- MetaEuk/AUGUSTUS on the eukaryotic candidates;
- Prodigal single mode on MAGs;
- clustering the candidates into families.

**Recommendation.** Treat all three as future work in the Discussion.

**Decision.** 2026-10-16, Rob: recommendation accepted.

## D14. Data deposition (ROADS / figshare)

Gadi owns checkpoint E: `roads_deposition_candidates.md`. The Teams inventory flags these packages over 500 MB:
- `smORFs/results/smorf_entropy.tsv.gz` (0.9 GB);
- `tables/gtdb_prodigal_coords_all.tsv.gz` (5.7 GB);
- `LOGAN/qc/pilot/features.tsv.zst` (4.3 GB);
- `entropy_rows/` (153 GB).

LOGAN raw outputs (5.95 + 3.32 TB) and the GTDB raw tars (1.5 TB) are too large to deposit; for those, deposit the manifests and summaries instead.

**Decision.** 2026-10-16, Rob: recommendation accepted. Rob arranges any DOIs; agents do not claim one.

---

## Decided

### D15. Stratifying the v7 comparison by assembly quality

**Context.** #128 withdrew the high-N50 claim. It also recorded that the quartile tables showing an excess in every quartile predate the v7 re-matching, so the role of assembly quality is open (G-17). Re-stratifying v7 needs only aggregation over existing tables, weighted by draw multiplicity: under 1 SU on Gadi, with no new search. Only **36.7% (bacteria) and 26.5% (archaea)** of matched comparator draws come from the candidate's own genome, so the strata can be defined in two ways.

**Options.**
- (i) **By the candidate's genome quartile.** Each candidate and its matched partner stay together, preserving the length × 3Di × truncation matching within every stratum.
- (ii) **By each ORF's own genome quartile.** Each arm is internally consistent about assembly quality, but matched pairs split across strata, so the strata are no longer matched.

**Recommendation.** (i) as the primary analysis, because the estimand is a matched contrast. Report (ii) as a sensitivity check. If they agree, assembly quality is not confounding the estimate. If they differ, that difference is the finding.

**Decision.** 2026-10-16, Rob: (i) as the primary analysis, (ii) as a sensitivity check. Gadi to run it (agent-handoffs#24); G-17 is updated from the result.

**Outcome.** Run by Gadi in #134 (`Validation/gtdb/assembly_strata_v7.md`, merged to `main` 0281875). The excess is positive in every stratum. Bacterial shares are higher in poorer assemblies; archaea show no trend. Schemes (i) and (ii) agree on every direction. Recorded in `results_inventory.md` G-17.


