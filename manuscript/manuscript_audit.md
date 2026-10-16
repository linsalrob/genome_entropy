# Audit of `genome_entropy.md` against the final results

A line-by-line check of the manuscript as it stood at `manuscript` d439428 (2026-10-16), carried out for checkpoint A of [agent-handoffs#24](https://github.com/linsalrob/agent-handoffs/issues/24). Line numbers refer to that commit. The final values are summarised in [`results_inventory.md`](results_inventory.md) and the open choices in [`decisions.md`](decisions.md).

**Main finding.** The LOGAN text was copied from superseded drafts of the stand-alone LOGAN manuscript (`agent-handoffs/projects/genome_entropy_logan/manuscript/manuscript.md`, "LMS"):
- the Methods n-values and Table 2 are from before the final all-shard run (93a1bb9, 2026-10-12);
- the candidate/Foldseek section is from the four-database run, before AFDB50 was added (f4477c2, 2026-10-15).

There is no GTDB section.

Abbreviations: **MS** is this manuscript; **LMS** is the LOGAN draft; **GR** is `Validation/gtdb/scientific_report.md`.

## Line-level audit

Status codes:
- **V**: verified against a final source.
- **S**: stale; the correct value and its source are given.
- **U**: unsupported; no source found.
- **ND**: needs data; the source is probably on Gadi, Teams or a notebook.

### 2.1 Abstract and Introduction

| MS line | Claim | Source | Status |
|---|---|---|---|
| 22-25 | "curated set of bacterial genomes annotated in GenBank" | GTDB r232: 189,715 bac + 10,122 arc (GR:16-19) | S/vague. Name GTDB r232 and the counts; Fig. 1 is not from GTDB (see l.295) |
| 25-28 | Real proteins occupy a distinct joint region; non-coding ORFs show "structural incoherence" | GTDB: separation is 3Di (2.08 bits) and protein is weak (0.39) (GR:334-337). Non-coding ORFs are *low-state* (D/V/P, log₂3 ceiling), not "incoherent" (GR §5) | Partly V; reword |
| 28-31 | "one million ORFs from LOGAN … predicted by Prodigal" | 89.99 B six-frame ORFs, compared with Prodigal (LMS:27, 159) | **S** |
| 33-35 | Joint consideration "robustly separates" | LOGAN within-bin AUC 0.90-0.96; GTDB; phage | V (qualitative) |
| 66 | Entropy framework [1] Akhter 2013 | citation | OK |
| 76 | "nine billion ORFs from the LOGAN dataset" | 89.99 billion (LMS:159) | **S** |

### 2.2 Methods

| MS line | Claim | Source | Status |
|---|---|---|---|
| 84-90 | Same codebase for GenBank and LOGAN | GTDB 0.2.0; LOGAN 0.3.0; phage v0.1.15 / 0.2.0; smORFs 0.3.0 (SUMMARY §C) | S: versions differ; state them |
| 94-98 | in_genbank = "overlapped" a CDS | Rule: same strand and stop, phase, ≥90% overlap, ≥98% aa identity (LOGAN README:126-127; SUMMARY l.30) | S: inaccurate |
| (missing) | min ORF length | Effective 90 aa for GTDB, phage and MI (#103; GR:135-141); 30 aa for LOGAN | Gap: must state |
| 100-102 | "For LOGAN, ORFs were predicted using Prodigal and treated as putative genes" | Six-frame `find_orfs`; Prodigal only labels (LMS:197-212) | **S**; contradicts l.222-226 |
| 104-118, 133-136 | Entropy equations | Equations lost in conversion; "{A. C, G, T}" typo | Structural |
| 127-129 | "pretrained protein structure models" | modernprost-50M throughout; model citation missing | U (citation) |
| 138-141 | Functional proteins are "repetitive" with low-complexity non-coding | Data show *higher* 3Di entropy in real proteins | Wording conflicts with results |
| 150-152 | Python libraries; figures from pipeline output | — | vague |
| 156-161 | INPHARED reannotated with Pharokka, **Phold and Phynteny** | Repo: Pharokka + phold, Pyrodigal-gv 0.3.2 positives (Inphared README:73); Phynteny not recorded anywhere | Phynteny U; versions not recorded |
| 163-170 | "all possible ORFs" with start/stop flags and position | Min 90 aa (28.8% of CDS invisible); 49 of 361 parts missing (Inphared README:71-75) | S: qualify |
| 172-179 | XGBoost classifier with positional features | Final ranking is XGBRF, **intrinsic** features only (position and strand dropped because of the strand shortcut) (Inphared README:14-17, 78) | **S** |
| 181-193 | OM869708 / genbank_viewer / BLASTP | No repo source | **ND** (see 2.5) |
| 197-202 | LOGAN filter counts 38,072,291 / 7,754,394 / 1,122,461 / 863,841 / 761,663 | LMS:125-133 | V |
| 203-214 | 13 biomes; v1.1 seqstats; ≤30% disagreement; 5-300 Mbp; N50 ≥150 bp | LMS:138-145 | V. Omits stratification and the ≤25 runs per BioProject cap (LMS:145-148) |
| 218-222 | 81 not found, 1 discarded, contigs ≥120 nt | LMS:156-157, 189-192 | V |
| 222-235 | MetaProdigal GFF; Prodigal+ rule; 5 classes | LMS:193-220 | V. Prodigal 2.6.3 `-p meta` version omitted; `frame_undefined` class omitted |
| 237-242 | modernprost-50M rev 2d68d506; default enc; MI250X 8 × 4; 215-220 k aa/s; 1 M chunks; SHA-256 | LMS:222-228 | V. Add 0.3.0 and encoding size 10,000 |
| 246-255 | ≤1,000 chunks per shard; 16.5 GB/B; 2.7 M check; 5×10⁻⁵; 100% | LMS:251-255 | V |
| 257-266 | Histograms, exact AUC/AP, nine bins, length-stratified AUC | LMS:257-262 | V |
| 268-270 | Model sample "8.2 million complete ORFs from 61 shards" | **11.7 M ORFs (9.5 M complete) from all 90 shards** (LMS:264-265) | **S** (pre-93a1bb9) |
| 270-276 | HGB, 5-fold grouped by accession, weights; fragments | LMS:266-269 | V |
| 278-280 | Six-frame "7.74 million genes from 77 shards" | **8.82 M from all 90 shards** (LMS:271-272). MS l.474 itself says 8.82 M | **S**; internally inconsistent |
| 280-289 | Best-overlap per alternative frame; Spearman; top-k vs 1/(k+1) | LMS:273-276 | V |

### 2.3 Results

| MS line | Claim | Source | Status |
|---|---|---|---|
| 295-297 | "ORFs from several hundred bacterial genomes" | Fig. 1 (image1) is an old test-case scatter (Jan 2026, ~130 genomes, superseded per SUMMARY §2). GTDB is 189,715 genomes | **S**: replace with GTDB `protein_vs_3di_joint_bac.png` / `_domains.png` |
| 299-304 | Unannotated ORFs bimodal, "abnormally low aa entropy" | GTDB unmatched protein mean 3.59 vs 3.98; low **3Di** (2-3 states) is the signature (GR:288-300, §5) | Partly S |
| 306-309 | "neither aa nor 3Di alone sufficient; strongest from combination" | Contradicted: 3Di alone carries most of the signal. GTDB gap 2.08 vs 0.39 bits; phage 3Di AUC 0.947, pair adds only +0.057 bits (MI README:10-13); LOGAN 3Di within-bin 0.896 vs aa 0.805 (LMS:332-333) | **S/contradicted** |
| 314-315 | Fig. 1 legend | image1 | S (see above) |
| 319-328 | Non-coding ORFs with protein-like composition differ in 3Di | GTDB protein gap small, 3Di large (GR:334-337); MI README | V (qualitative); cite numbers |
| 332-339 | 50,009; 4.90 Tbp; 13 biomes; ≥30 aa; 89.99 B; 5.28 T | LMS:282-288 | V |
| 342-349 | 47.6 / 27.0 / 8.7 / 15.5%; ~59% open; 18.1 / 41.2% | LMS:291-297; S3 | V |
| 351-354 | Complete Prodigal 1.0%, ~881 M; partial 14.5% | LMS:299-301; S3 (881,348,601) | V |
| 360-378 | **Table 2**: 1.02%, 620 M; 14.43%, 5.4 M, 1.34 B, 7.46 B; antisense 13.39 B …; intergenic 8.91% | Final S3 / LMS:310-316: 0.98%, **881 M**; 14.54%, 7.6 M, 1.97 B, 11.12 B; antisense 47.55%, 19.31 B, 8.08 B, 15.40 B; frameshift 26.97%, 11.82 B, 4.27 B, 8.19 B; intergenic 8.71%, 3.90 B, 1.74 B, 2.20 B; marginal 1.18%, 687 M, 192 M, 184 M; same-frame 0.07%, 31 M, 29 M | **S**: whole table from a partial-shard draft; contradicts MS l.351-354 |
| 383-392 | Length tautology; 50-aa strata | LMS:320-328 | V |
| 394-398 | Structure distinguishes within strata; "Thus…strengthened with length" | The supporting numbers were dropped: 3Di 0.896 / aa 0.805 / MI 0.902 at 100-149; 0.960 at 150-199; 0.988 at 200-299 (LMS:332-336) | V, but the "Thus" sentence is orphaned; restore the numbers |
| (missing) | Intergenic-only negatives (LMS:340-346); biome/tranche replication, Fig S1 / Table S21 (LMS:363-392) | — | Gap |
| 400-406 | Shadows lower in 3Di/12-state within strata | LMS:348-354 | V |
| 416-432 | Grouped-CV models; structural gain larger vs shadows | LMS:396-410 | V (no numbers) |
| 434-440 | ~0.90 AUC at 100-149; ~0.96 at 150-199; <75 aa weak | LMS:412-417; image4 (+3Di ≈ 0.90; all four 0.94) | V |
| 452-465 | Fragments: 5′-open AUC 0.791 / AP 0.559; 3′-open 0.854 / 0.751 | LMS:433-438 | V; fig9_fragments is not included |
| 473-477 | 8.82 M genes; 5,722,417 loci | LMS:452-455 | V |
| 479-482 | ρ 0.33-0.45 (length), 0.29-0.36 (aa) | LMS:457-460 | V |
| 484-486 | "neither 3Di nor 12-state entropy was correlated" | 12-state ρ = 0.06-0.13 (weak); 3Di −0.01 to 0.03; MI −0.04 to 0.01 (LMS:462-465) | S: 12-state is weakly correlated |
| 498-499 | 3Di ranks Prodigal frame first in **92.5%** | **92.4%**; 12-state 91.3%, length 80.9%, aa 81.8%, chance 16.7% (LMS:476-478, Table 5 l.487-493) | **S** (pre-final value) |
| 504-508 | Within-locus contrasts; strongest above 75-100 aa | LMS:495-498 | V |
| 518-535 | Discordant population framing | LMS:513-530 | V |
| 537 | Heading "Biome specific analyses" | Content is candidate scoring | Structural mislabel |
| 539-544 | 49.95 M intergenic ≥100 aa; 4.58 M controls; thresholds (Table S12) | LMS:532-539 | V. Table 6 (thresholds: 3.43 M at P0.90) omitted (LMS:544-549) |
| 547-551 | 5.5% soil to 8.7% gut; 7% → 20% with length | LMS:551-554 | V |
| 553-564 | 20,000 candidates + three controls + shuffled; **four databases** | Now **five** databases (+AFDB50 v6, 66.7 M) plus a **frameshift-shadow** arm (LMS:556-566, 592-599) | **S** |
| 565-568 | **19.3%** candidates; 0.03% low-score; 0.13% antisense; 0 shuffled; Prodigal 56.1% | **41.9%**; 0.21% (41/20,000); 0.45% (89/20,000); frameshift 0.36%; shuffled 0.005% (1 of 20,000); Prodigal **72.6%** (LMS:569-573) | **S** |
| 569-572 | "several hundred times" the low-score rate | **~200×** (LMS:574-575) | **S** |
| 574-595 | **Table 7** (four DB) | Five-DB Table 7 (LMS:583-590): Prodigal 72.6 / 70.6 / 75.4 / 92.8; candidates 41.9 / 24.7 / 50.7 / 70.6; shuffled 0.005 / — / 0.04; low-score 0.20 / 2.5 / 0.50 / 26.1; antisense 0.44 / 1.6 / 0.86 / 22.9; + frameshift 0.36 / 0.40 / 0.48 / 8.0 | **S** |
| 597-600 | 16.6% structure-only, 2.7% both | **21.2%** structure-only, **20.7%** both, 4.0% sequence-only (LMS:606-607) | **S**. The "most similarity only derives from structure" framing no longer holds |
| 601 | "[Baktfold paper]" | placeholder citation | U |
| 602-606 | ~16% of 3.07 M → ~0.5 M ORFs | **~38%** of 3.07 M → **~1.2 M** (LMS:613-616) | **S** |
| 610-618 | Superkingdom: euk 29.8% vs 7.4%; bact 29.0 vs 54.0; viral 38.4 vs 35.1 | **bact 64.5%, euk 30.1%, arc 1.9%, viral 1.9%** vs Prodigal 88.4 / 2.9 / 2.1 / 4.7; the 38% viral share was a BFVD artefact (LMS:620-625) | **S** |
| 615-617 | Euk excess: food 60%, plant 62%, built 49% | plant **73%**, food **71%**, built **51%** (LMS:626-627) | **S** |
| 617-620 | Lowest candidate hit rates 13-17%; Prodigal 36-51%; gut, freshwater, marine 24-26%, 12-21% euk | Lowest 35-42% (plant, food, built, animal host, other); gut, marine, sediment, freshwater, wastewater 47-50%, 77-91% bacterial (LMS:627-630) | **S** |
| 620-624 | "Substantial part … eukaryotic; rest prokaryotic/phage" | Now: "Part … eukaryotic; most resembles bacterial proteins" (LMS:630-633); confirmed by Kraken2 (69.8% on euk contigs) (LMS:657-678) | **S**; add Kraken2 and Prodigal-skip results |
| 626-632 | Fig "8" legend, 4 DB | image 223ec… shows 4-DB numbers (19.3% / 56.1%) | **S**: redraw from `fig13_candidate_evidence.png` |
| 634-637 | Fig "10" evidence classes, "all four databases" | image 4ca38… = 4-DB `fig15` | **S** |
| 639-644 | Fig "11" biome / superkingdom | image 2c958… = 4-DB `fig16` | **S** |
| 648-656 | 5.28 T aa; 5.95 TB; 215-220 k aa/s; 1.06 M GPU SU; 264k CPU SU; 176k SU/Tbp | LMS:783-791 | V, but the caveat is missing: ~167k GPU SU were re-encoding after the inode-quota failure, and 176k/Tbp *excludes* it (LMS:788-791). Without the caveat 1.06 M / 4.9 Tbp ≠ 176k |
| 652-653 | "~5.7 × 10²⁰ floating-point operations" | not in LMS, AH or the repo | **U** |
| 667-699 | OM869708: ORF7 p = 0.27, low 3Di; ORF3 p = 0.91; BLASTP hit to Dipodfec virus RodF1 38 Rep | no repo or AH source | **ND**, and at risk (see 2.5) |
| 681 | "(Figure 1)" for the genbank_viewer plot | duplicates Fig. 1 numbering | Structural |

### 2.4 Discussion and Data availability

| MS line | Claim | Status |
|---|---|---|
| 738-741 | "strong agreement between GenBank-annotated proteins and the majority of LOGAN ORFs" | **U**: no GTDB-to-LOGAN comparison exists, and most LOGAN ORFs are shadows. Also conflicts with LMS 3.6 (low-scoring Prodigal genes: 6.9% vs 87.5% structural hits, LMS:725-771) |
| 741-743 | "does not rely on homology" | V |
| 745-749 | Post-processing proposal | opinion. Discussion is ~3 paragraphs and covers no limitations: length bias #101, <75 aa, log₂k ceilings, PU labels, Prodigal 3.6× disagreement, 3Di non-determinism |
| 753-755 | "code at GitHub; data on request" | Incomplete. Zenodo 10.5281/zenodo.18312207 exists; GTDB tables on Gadi; LOGAN cache on Teams (README:87-89, 117-121) |

### 2.5 OM869708 example (MS l.181-193, 665-727)

**Checked 2026-10-16 against the corrected phage model (Setonix `work/phage_rank/`). The example stands; its numbers are stale.**

- **The strand bug cannot affect it.** All 16 ORFs ≥90 aa and all 7 CDS of OM869708 are on the plus strand.
- **ORF3** (+, frame 1, 4723..5730, 335 aa, 3Di 2.98 bits) scores p = **0.889** with the corrected intrinsic model, against 0.947 legacy.
- **The annotated ORF7** (`OM869708_CDS_0007`, 5053–5650, 198 aa, 3Di 1.20 bits) scores p = **0.175**, against 0.199 legacy.
- **ORF3 → ORF7 is still a replacement candidate:** rank 179 of 4,102 corrected, against 121 of 1,797 legacy.
- **The MS values 0.91 and 0.27 match neither model** (probably the July notebook model, #70). Update the text.
- **BLASTP re-run (2026-10-16):** the *Dipodfec virus* RodF1 38 hit is the genome's own submitted Rep (UPW42009.1), which the re-annotation dropped. Independent homologues are at 43–55% identity. ORF7 has no hit. Reframe as in D7.
- **Still unsupported (D12):** the genbank_viewer figure (replaced by Fig. P01d) and "Phynteny".

### 2.6 Figures: images ↔ legends

| MS image | Legend in MS | Actual source (matched by pixel size / content) | Status |
|---|---|---|---|
| `images/media/image1.png` (846×547) | Fig. 1 (l.314) | Old "Protein vs 3Di entropy coloured by in GenBank" scatter (test-case era, not GTDB) | S: replace with GTDB `protein_vs_3di_joint_bac.png` |
| `images/media/image2.png` (3784×1614) | Figure 2 (l.411) | LMS `fig6_distributions_fixed_length.png` (identical size) | V (unchanged since 93a1bb9) |
| `images/media/image4.png` (2976×1168), alt text "Figure 4" | Figure 3 (l.445) | LMS `fig8_models_within_length.png` | V; numbering mismatch |
| `images/media/image6.png` (1645×895), alt "Figure 6" | Figure 6 (l.493) | LMS `fig10_six_frame_inheritance.png` | V; numbering gap (4 and 5 missing) |
| `images/media/image8.png` (3586×830), alt "Figure 8" | Figure 8 (l.513) | LMS `fig12_six_frame_delta.png` | V; `fig11_six_frame_discrimination` (top-k) and Table 5 not included |
| `images/223ec3…png` (560×198) | **Figure 8** again (l.629); text cites "Figure 9" (l.568) | 4-DB `fig13_candidate_evidence` | **S** + duplicate number |
| `images/4ca389…png` (560×215) | Figure 10 (l.637); text cites "Figure 11" (l.600) | 4-DB `fig15_candidate_evidence_classes` | **S** |
| `images/2c9587…png` (560×256) | Figure 11 (l.642); text cites "Figure 12" (l.618) | 4-DB `fig16_candidate_biome_taxonomy` | **S** |
| `images/media/image9.png` (1673×271) | **Fig. 1** again (l.723) | genbank_viewer plot of OM869708 | ND; duplicate number |

The three hash-named images are low resolution (560 px wide) and need re-export. Unused but available:
- LMS `fig9_fragments`, `fig11`, `fig14`, `fig17`, `fig18`, `fig19`, `figS1`;
- all GTDB, Inphared and MI figures.

### 2.7 Structural problems from the Word conversion

- **Empty sections and lost equations:**
  - empty `# Authors` and `# ` headings (l.6-8);
  - the three entropy equations are missing (l.107-109, 115-117, 133-135);
  - "DNA sequence entropy" is plain text, not a heading (l.104);
  - inline list flattened into a sentence (l.198-202, "- librarysource …").
- **Duplicated headings:**
  - l.381: heading text repeated twice;
  - l.516: heading followed by a bold "**3.6 …**" duplicate;
  - l.646: "LOGAN Computational scale**3.7 Computational scale**".
- **Numbering:**
  - figures run 1, 2, 3, 6, 8, 8, 10, 11, 1;
  - in-text calls (Figure 9, 11, 12) do not match the legends;
  - tables are 2 and 7 only, with no Table 1;
  - S-tables S12, S14, S16, S19 and S20 are cited, but no supplement exists in the manuscript branch (they are in `AH/manuscript/tables/`).
- **Heading hierarchy:**
  - LOGAN subsections are `###` under the `##` "six-frame ORF space" heading;
  - "Biome specific analyses" is mislabelled;
  - the OM869708 section sits after LOGAN, while its Methods sit before LOGAN.
- **Order:** the GenBank/GTDB Methods do not describe GTDB at all.

### 2.8 Citations

- Only 2 references, and ref 2's journal is garbled ("bioRxivorg").
- `[Baktfold paper]` is a placeholder (l.601).
- Uncited tools and resources:
  - Prodigal, GeneMark, Glimmer;
  - Foldseek / 3Di (van Kempen), ModernProst, ProstT5;
  - Pharokka, Phold, Phynteny, INPHARED, Pyrodigal-gv;
  - XGBoost, scikit-learn, BLASTP / NCBI nr;
  - PDB100, AFDB / Swiss-Prot, CATH, BFVD, AFDB50;
  - Kraken2 / PlusPFP, GTDB, GMSC;
  - genbank_viewer.

### 2.9 Analyses complete in the repo or AH but absent from the manuscript

1. **GTDB r232, no section at all:**
   - 199,837 genomes and 2.62 B ORFs;
   - 3Di gap 2.08 bits vs 0.39 for protein;
   - log₂3 ceiling;
   - never-annotated genome confounder;
   - missed genes ~51-59k (floors 27.4k / 5.4k);
   - Prodigal 3.6× disagreement;
   - exemplars.
2. Length-bias decomposition (PR #101): protein entropy's length trend is ~all artefact; 3Di 12-15%.
3. GMSC smORFs, a negative/limit result: AUC 0.73; 3Di 0.605.
4. Phage MI (0.587 bits; 82% length-robust) and the corrected phage ranking with Foldseek validation (33.8% / 6.1%, mixture 34%).
5. Newer LOGAN results:
   - Kraken2 contig taxonomy;
   - why Prodigal skipped candidates (coding score −0.6 vs +42);
   - low-scoring Prodigal genes (6.9% vs 87.5%);
   - frameshift control;
   - biome/tranche replication (Fig S1);
   - Table 1 dataset;
   - Table 5 top-k;
   - Table 6 thresholds;
   - fragments figure.
6. Methods caveats:
   - #103 90-aa floor;
   - versions and encoding sizes per dataset;
   - 3Di run-to-run / hardware variation and the determinism test (AH inphared_gadi §5);
   - #112 `--max-aa` / OOM on long ORFs (39 GTDB genomes).
