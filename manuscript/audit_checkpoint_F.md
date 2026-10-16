# Checkpoint F audit: manuscript/genome_entropy.md (branch ms/setonix-front-back, PR #141, HEAD 3767946)

**Status (integrator, 2026-10-16):** findings applied on PR #141 (commit after 104ff43), except A1 (title, Rob's decision) and the ModernProst citation (needs G. Bouras). Line numbers below refer to the audited head 3767946.

Auditor: independent Claude agent, read-only. 2026-10-16. agent-handoffs pulled (ff) before the audit.
Scope: every number in the Abstract, Introduction, Results, Discussion and Tables 1–3; Methods for GTDB, LOGAN, phage, smORF and reproducibility were checked in full (well over 50%).
Sections: A = the auditor's own pass (Abstract, Intro, Discussion, GTDB Methods and Results, figures and decisions); B = LOGAN (MS l.202–242, 393–528 excluding GTDB); C = phage, smORF, reproducibility and Table 1; D = references. Line numbers are manuscript lines.

## Summary of counts

| class | A: front/GTDB/Discussion | B: LOGAN | C: phage/smORF/Table 1 | total |
|---|---:|---:|---:|---:|
| OK | ~125 | ~297 | ~136 | ~558 |
| MISMATCH | 2 (A3, A17) | 3 | 2 | 7 |
| ROUNDING | 2 | 2 | 1 | 5 |
| UNSOURCED | 3 | 2 | 3 | 8 |
| SUPERSEDED | 0 | 0 | 0 | 0 |
| CLAIM stronger than source / contradicts a decision | 9 | 4 | 8 | 21 |

No value from results_inventory.md §6 (superseded list) appears anywhere. Every [@key] resolves (29 cited = 29 bib entries).

## Highest-priority items (fix before submission)

1. **Title vs D8** (A1). "Joint sequence and structural entropy…" contradicts the accepted D8 and the paper's own Discussion (l.632).
2. **Abstract l.20 and Discussion l.635: "spurious ORFs collapse onto two or three states"** (A2). The population fact is entropy < log₂3. The state-count evidence comes from 3 genomes.
3. **Discussion l.644: "200 times as often as matched controls"** (A3). This holds only against low-score intergenic ORFs; against shadows it is ~93–116×.
4. **l.351 and Discussion l.665: the cross-platform maxima 0.92 / 1.05 bits are 50-genome values** (C2, A17). Whole-population maxima are 2.016 / 2.284 bits (IG:201–202). The "at most 0.92 bits" in the Discussion is wrong as a general statement.
5. **l.426: six-frame Δ, "in a random sample of 324,690 genes … across all genes 97.0%"** (B). The 1.99 bits and 97.0% are for the 5,722,417 genes with all five alternatives. Over all 8,797,054 genes the values are 1.82 bits and 94.2%.
6. **l.242: compute overhead.** 6.8% and 26.8% are shares of billed time, not excess over encoding time (7.2% and 36.6%), and "about 136,000 SU" mixes the two bases (~134,200 or ~125,200 SU).
7. **l.250: "17,322 run off a contig end"** should be 18,322. The source typo is at MI README:37; its own table at MI:48–51 gives 18,322.
8. **Discussion l.651 "likely to overestimate" vs D2** (A4). The gap is unresolved.
9. **References**: the Zenodo DOI 18312207 is the v0.1.1 version DOI, so cite the concept DOI 18312206 or archive v0.4.0. Phold is now published (NAR 2026, gkaf1448). ModernProst has no citable paper.

## A. Abstract, Introduction, Discussion, GTDB Methods/Results (auditor's own pass)

Sources: GR = Validation/gtdb/scientific_report.md; MC = matched_comparator.md; AS = assembly_strata_v7.md; SA = supplementary_analyses.md; v7 = Validation/gtdb/matched_comparator/{bac,arc}_v7_matched.tsv; data/ = manuscript/data.

### A1. Non-OK findings

| # | MS line | quoted text | class | issue | correct value / source | suggested fix |
|---|---|---|---|---|---|---|
| A1 | 3 (title) | "Joint sequence and structural entropy distinguish protein-coding genes…" | CLAIM / D8 | Contradicts D8 (accepted): "3Di entropy carries most of the signal; amino-acid entropy adds little". The Discussion (l.632) itself says the "read jointly" idea "needs refining". | decisions.md D8; MS l.627–632 | Retitle, e.g. "Predicted structural entropy distinguishes protein-coding genes from spurious open reading frames…" |
| A2 | 20 (Abstract); 635 (Discussion) | "most spurious ORFs collapse onto two or three predicted structural states" / "Spurious ORFs collapse onto two or three 3Di states" | CLAIM | Population evidence is that 87.6% of *unmatched* ORFs have 3Di entropy < log₂3. Entropy < 1.585 does not imply ≤3 states (a 4+-state string dominated by D also lies below it). The state-count evidence (median 3 distinct states below the line) comes from 52,947 ORFs in **three genomes** of `bac_000`; GR §10 calls the D/V/P composition "a 3-genome sample estimate". "Spurious" also equates "unmatched" with non-coding, which l.92 and l.143 disclaim. | GR:624–655 (3-genome table, "median distinct states 3"), GR:248–251 (87.6%/84.1%), GR:1344–1345 | "most unannotated ORFs have 3Di entropy below log₂3, the ceiling for a string of three states; in a three-genome sample they use a median of three states" |
| A3 | 644 (Discussion) | "full-length structural homologues 200 times as often as matched controls" | MISMATCH / CLAIM | 200× holds only against low-score intergenic ORFs (0.419/0.00205). Against antisense shadows (0.45%) it is ~93×, frameshift (0.36%) ~116×. Results l.514 correctly says "about 200 times the low-score rate". | AH tables/S33_all_arms_evidence.tsv; MS l.514 | "about 90–200 times as often as matched controls" or "200 times as often as low-scoring intergenic ORFs" |
| A4 | 651 (Discussion) | "so this figure is likely to overestimate" | CLAIM / D2 | D2 (accepted) is to state the gap as an unresolved limitation; GR lists over-calling, structural under-counting, or both, "not separable with what is here". Results l.476 says "We cannot resolve this gap". | GR "Where the two estimates disagree" (GR:1101–1125); decisions.md D2 | "so this figure may overestimate" / "it is an upper bracket" |
| A5 | 650 (Discussion) | "It is conservative: structure databases contain full-length homologues for only 30–35% of known genes." | CLAIM | The mixture divides by the annotated-CDS rate, so a low database ceiling alone does not make f conservative; GR argues under-estimation only if missed genes are *worse represented than average annotated genes* ("plausibly"). GR also calls the structural figure "the more conservative" (relative), not conservative in absolute terms. | GR:1113–1122 | "It is probably conservative, because missed genes are likely to be represented in structure databases less well than annotated genes (30–35% of which have full-length hits)." |
| A6 | 476, 650 | "full-length hits for only 30–35% of known genes" | ROUNDING / imprecise | Per-database annotated full-length rates are Swiss-Prot 35.3/35.9%, PDB100 30.1/32.6%, CATH50 23.9/25.3% (bac/arc); "any of the three" is 41.6/43.6%. "30–35%" omits CATH50 and is copied from GR. | v7 `fulllen_*_annotated`; GR:943–955 | "24–36% per database (42–44% in any of Swiss-Prot, PDB100 or CATH50)" |
| A7 | 387 vs 389 | "less than the 0.11 bits that plug-in estimator bias alone produces" (both domains) vs "Only 0.11 and 0.10 bits of that … is bias" | ROUNDING | The i.i.d. bias rise 100→300 aa is 0.105 bits in **both** domains for protein and for 3Di; it is rounded to 0.11 for archaea in l.387 and to 0.10 for archaea in l.389. | data/figG02_mean_curves.tsv.gz (bac protein bias 0.105, 3Di 0.105; arc protein 0.106, 3Di 0.105) | Use "0.105 bits" or round consistently (0.11 and 0.11; 12% and 14% still hold) |
| A8 | 657 (Discussion) | "Correcting the labels raised the model's AUC from 0.740 to 0.976 using intrinsic features alone." | CLAIM (attribution) | 0.740 is the *legacy, strand-using* model scored against corrected labels; 0.976 is a different model retrained on corrected labels with intrinsic features. The rise reflects retraining and the feature change together, not the label correction alone. | Validation/Inphared/README.md Table 1; MS l.544–546 | "Retrained on corrected labels with intrinsic features only, the model reached AUC 0.976, against 0.740 for the earlier model scored on the same labels." |
| A9 | 143 | "The six-frame caller recovers only about 90% of deposited CDS." | CLAIM (generalisation) | Only measured in archaea: median 161 deposited CDS per archaeal genome unmatched (1,502 of 1,670, ~90% recall). No bacterial figure in GR. | GR:831–839, GR:1361–1362 | "recovers about 90% of deposited CDS in archaeal genomes (median per genome)" |
| A10 | 664 (Discussion) | "an effective 90-amino-acid floor from a package default" | CLAIM (cause) | The floor came from a unit error: the default 30 aa was converted to 90 nt and passed to `get_orfs -l`, which reads amino acids (#103). Methods l.78 says this correctly. | GR:135–141; MS l.78 | "from a unit error in the package (#103)" |
| A11 | 482 | "Mobile elements are genuinely under-annotated in GenBank." | UNSOURCED | Stated as fact with no citation or analysis; GR asserts it without evidence. | GR:1155–1159 | Cite a source or soften ("consistent with the known under-annotation of mobile elements [ref]") |
| A12 | 432 | "because a sequence-only encoder emits confident 3Di for any protein-like sequence" | CLAIM (mechanism) | Mechanism inferred, not tested; SA says the saturation "confirms the report's point" but neither measures encoder confidence. | SA:126–131 | "probably because…" |
| A13 | 383 | "These genomes therefore appear to hold many unlabelled genes" | CLAIM (mild) | High 3Di alone is not evidence of a gene (MS l.432 says so); the 23-fold enrichment is in 3Di ≥2.5, not in homology. Acceptable with "appear", but sits awkwardly with l.432. | GR:777–791 | "…hold many gene-like, unlabelled ORFs" |
| A14 | 178 | DB versions "AlphaFold DB/Swiss-Prot (v6, 2025_03) … PDB100 (2025-01-01) … CATH50 (v4.3.0) … BFVD (2023_02_v2)" | UNSOURCED (locally) | Not in any repo file; src is a review comment on #137. `17_pilot_search.pbs` names DBs only. Note the GTDB BFVD (2023_02_v2) and phage BFVD (2025_03_v3) differ; disclosed. | Gadi review on #137 | Record the `.version` files in methods_provenance.md |
| A15 | 196 | "GTDB's own Prodigal v2.6.3 calls" | UNSOURCED (locally) | Version comes from GTDB METHODS.txt via review comment; not in repo. | — | Cite GTDB r232 METHODS or add to methods_provenance.md |
| A16 | 474 vs GR §1 | "3.5–4.0 times the structural estimate in bacteria" | OK (note) | GR §1 and §10 still say "~3.6×"; MS is consistent with MC. No change to MS. | MC Result table | — |

### A2. Internal-consistency checks (all OK unless listed above)

- Abstract 2.62 B ORFs = 2,568,244,984 + 54,858,398 = 2,623,103,382; 199,837 = 189,715 + 10,122 (encoded). OK.
- Abstract 2.08 / 0.39 bits = Results l.370–371 = GR §4.1 item 1 (3.123−1.043; 3.979−3.592=0.387). OK. Discussion "five times" = 2.08/0.39 = 5.3. OK.
- Abstract 92.4% = Results l.426 = Discussion l.636. OK (qualifier: of the 5,722,417 genes with all five alternatives).
- Abstract 0.587 = Results l.536 = Discussion l.628; 0.057 = l.536. OK.
- Abstract 51,000–59,000 = 47,300+3,500 … 54,200+4,900 (50,800–59,100) = l.440 = l.641/650. OK.
- Abstract ~5,400 = 4,974+407 = 5,381. OK. ~200,000 = 190,597+9,468 = 200,065. OK.
- Abstract 41.9% / ≤0.45% = l.514. OK.
- 28.7% phage floor: l.250, l.575, l.664 consistent (D6 says 28.8%, which is the 31,200-genome subset; MS notes this at l.250). OK.
- 17.4% low-scoring Prodigal genes: l.526 = l.645. OK.
- Determinism: l.351 "0.31 bits … up to 0.92" = l.665. OK.
- Figure/table numbering vs figures_manifest.tsv: every cited Fig. 1–6, S1–S18 and Table 2, S1–S14 exists in the manifest. Panel letters checked against notebook captions: Fig. 2a–d (AP for the four negative sets; L02 caption a–d AP, e–h AUC), Fig. 3a/3b/3c–g (L05), Fig. 5a/5b (L06), Fig. S4a (L03), S10a/b/d (L07), S11a/b (L08), S13a/b (L10), S9a/b (G06) — all consistent. Minor: G01 caption says out-of-window rows "are not drawn in panels a–d", but manifest panel d is the population-band panel, not a 2-D density (caption wording only).
- Table S4 ambiguity: Methods l.224 and l.232 comments cite `AH/manuscript/tables/S4_models_overall.tsv` (LOGAN-draft numbering) while the manuscript's Table S4 is `tabL04` thresholds. Text is correct; only the src comments may confuse a reader. Same for S5–S8 in comments (LMS numbering ≠ MS numbering).

### A3. GTDB values verified OK (line: value → source)

137: 189,801; 10,122; 47; 39; 189,715; 2,568,244,984; 54,858,398 → GR §2. 139: 92,840 (48.9%), 4,630 (45.7%), 96,875, 5,492 → GR §2, §6 table. 141: 90 aa → GR:135. 147: 0.2.0/90c8c15/schema 2.2.0 → MP:37; V100-SXM2-32GB, 4 per GPU, 10,000 → GR §3; 39 (0.021%), 48,000–766,000 aa → GR §7 (48,298; 766,188). 149: 10⁻⁴-bit bins → GR §4.1; every 300th/30th; 4,893,192 and 1,138,714 → GR:580. 152: 400,000 draws, 671 lengths, 351 genomes → GR:373–379. 153: 1-aa bins to 999 → GR:418. 165: 3,925,368 / 133,748 → MC item 1. 167: seed 0, ±0.025, ±10%, 36.7/0.16, 26.5/4.3 → MC item 2; TVD 0.003/0.010 → MC table. 174: 151,927 → v7 n_annotated. 178: Foldseek 10.941cd33, E ≤10⁻³, 80% coverage, quintile edges 3.22/3.32, 88–99% → MC item 3, SA:97, 130. 180: ~11% BFVD → GR:960. 196: 618,638,921; 0 bp → GR:1033–1040. 198: 54 metrics → AS Method.
367: 12.03% / 17.26% → GR §2. 370–375: 3.123/1.043, 1.99, 0.39/0.37, 0.960/0.321, 1.930/1.909, 376/178 → GR §4.1. 377–379: 87.6/3.3, 84.1/3.7 → data/figG01_population_bands (0.876025, 0.032569, 0.840920, 0.036502). 381: 99.4% D/V/P in three genomes → GR:649. 383: 19.1/28.3, 0.83/1.25, 23-fold → GR:785–788; bands (1−0.991718, 1−0.98753). 387–389: 0.08/0.09, 0.88/0.76, 12%/14% → figG02 (0.083, 0.086; 0.876, 0.761; 12.0%, 13.8%) [rounding of 0.105 see A7]. 391: z SD 1.00; i.i.d. 3.9–6.6 → figGS02_summary (3.92–6.56).
430: 9,598,500; 94.5%; 523,346; 22,447 → GR §6; figG04_funnel; ρ 0.55 (0.553), 0.10 (0.0997) → figGS03_burden_rho. 434: ≤0.04% (224/523,346) → GR:860; 10.3/7.4/35.3 → GR:943. 436: all six shares and CIs → MC Result, v7. 438: 10.3–10.6; 14.0–19.8 → MC. 440: 47,300–54,200; 3,500–4,900 → v7 structural_estimate (47,291–54,174; 3,512–4,928). 445–448: 25,620+1,731; 4,974+407; 18,566; 13,996 → GR:991–1027. Table 3: all 42 cells → v7 ladder_C*. 466–470: 47.4/55.4, 20.6/25.6, 94.2/96.1 → v7 prodigal_*. 474: 36.4 (36.2–36.6), 42.2 (41.1–43.2), 190,597 + 9,468, 3.5–4.0, 1.9–2.7 → GR:1074–1077, MC. 478: 91.8% (21,617/23,540), 95.9% (1,509/1,574) → GR:1127–1130; 56.7/68.4 → SA:607. 482–484: 33.8/41.3, 13.6/11.8, 23,393, 8,065 → GR:1139–1160. 486–494: C7 + Prodigal-called examples → figG06 caption cell 4; Cas5 226 aa, 5 bp, 96 bp; SusC 917 aa, 3,029 bp → GR:1240–1250. 498–506: 2.97%, 17.4→10.3, 19.9→9.0, 9.2–12.1, ±5, 44.8→31.1, 47.0→32.6, 3.3 points → AS tables and Reading 1–6.

Additional finding from the auditor (confirmed after C2):

| A17 | 665 (Discussion) | "per-ORF values differ between hardware platforms by more than 0.3 bits for 1% of ORFs (at most 0.92 bits)" | MISMATCH | 0.92 is the maximum in the 50-genome determinism test. The whole-population Gadi vs Setonix maximum is 2.016 bits for 3Di (p99 0.317). | AH inphared_gadi/README.md:201 | "(up to 2.0 bits)" |


## B. LOGAN (sub-audit; full detail)
### Numeric audit: LOGAN Methods (l.202-242) and LOGAN Results (l.393-426, 510-528)

Manuscript: wt/ms-fix/manuscript/genome_entropy.md (2026-10-16).
LMS = agent-handoffs projects/genome_entropy_logan/manuscript/manuscript.md at 76aec1a (includes 9886e67). LMS line numbers below are current; the cited ones still match.
Scope note: lines 428-508 (GTDB "Missed genes in GTDB") fall inside 393-528 but are not LOGAN sections and were not audited.

## Counts per class

| class | n |
|---|---:|
| OK | ~297 numbers |
| MISMATCH | 3 (l.426 scope; l.242 overhead %; l.242 136,000 derivation) |
| ROUNDING | 2 (l.520 9.6%; l.526 99.9%) |
| UNSOURCED | 2 (l.206 7,653,092; l.212 byte-identical rerun) |
| SUPERSEDED | 0 |
| Wording stronger than source | 4 (l.516, l.528, l.238, l.234) |

## Non-OK findings

| line | quoted text | issue | correct value + source | suggested fix |
|---|---|---|---|---|
| 206 | "LOGAN v1.2 contigs available (7,653,092)" | UNSOURCED. Not in LMS:125-133 (LMS lists METAGENOMIC → WGS directly) and found nowhere in the worktree, Validation/logan or agent-handoffs | none found | Get the number from the LOGAN agent (01_select_accessions output) and cite it, or drop the step |
| 212 | "byte-identical ... and on the rerun of the 79,774 query contigs" | UNSOURCED locally. LMS:680-684 says only that the `-s` rerun "reproduces the original calls" (every Prodigal-arm gene called again with the same stop). It does not say byte-identical. The cited "#137 review" was not checked | LMS:680-684 | Keep only if the #137 comment says byte-identical; otherwise drop the clause |
| 238 | "which reproduced every original call" | Stronger than the source. LMS:683-684 checked only that every Prodigal-arm gene was called again with the same stop and that no candidate overlaps a called gene | LMS:683-684 | "which reproduced the original call for every Prodigal-arm gene" |
| 234 | "four arms of 20,000 matched to them by nearest length within truncation state: ... complete Prodigal genes" | The Prodigal arm is all complete (tabL06: prodigal complete 20,000), so it is not matched within truncation state. The script docstring gives same-truncation matching only for intergenic_lo | tabL06; 21_candidates_select.py:26-32 | "...by nearest length (within truncation state for the three negative arms)" |
| 242 | "Billed GPU time exceeded encoding time by 6.8% in the first tranche and by 26.8% in the second" | MISMATCH (wrong denominator). 6.8% and 26.8% are 1 − gpu_efficiency, i.e. non-encoding time as a share of *billed* time. Relative to encoding time the excess is 7.2% and 36.6% | S10: 3383.29 vs 3154.82 GCD-h; S11: 4876.36 vs 3569.28 GCD-h (gpu_efficiency 0.932, 0.732) | "6.8% and 26.8% of billed GPU time was not spent encoding" (or "exceeded encoding time by 7.2% and 37%") |
| 242 | "about 136,000 SU after allowing for the first tranche's overhead rate" | MISMATCH (derivation). Second-tranche excess = 1,307.1 GCD-h × 128 = 167,306 SU. Subtracting the first-tranche rate consistently gives 134,200 SU (7.24% of encoding hours) or 125,200 SU (6.75% of billed hours). 136,000 comes from applying the billed-share 6.8% to encoding hours, which mixes the two denominators. results_inventory L-18 still says ~167k | S10/S11 gcd_hours_*, su_gpu_encoding (128 SU per GCD-h) | "about 134,000 SU" (encoding-relative). LMS:793 has the same 136,000, so tell the LOGAN agent |
| 426 | "In a random sample of 324,690 genes, the selected frame exceeded the median ... by a median 1.99 bits ...; across all genes it did so in 97.0%" | MISMATCH (scope). In S8, 1.99 and 0.970 belong to the 5,722,417 genes with all five alternatives, not to the 324,690-gene histogram sample (whose median bin is 1.96-1.99, Δ>0 = 0.9698). They are not "all genes" either: over all 8,797,054 genes with ≥1 alternative, median Δ = 1.82 bits and Δ>0 = 94.2% | S8 rows three_di_entropy; data/figL05_six_frame_delta.tsv.gz (genes=324,690) | "Among the 5,722,417 genes with all five alternatives, the selected frame exceeded the median of its alternatives by a median 1.99 bits of 3Di entropy, and did so in 97.0% (Fig. 3c-g, drawn from a 324,690-gene sample)" |
| 516 | "not the precision of intergenic candidates; the hit rates above measure that directly" | Stronger than the source. A full-length hit rate is not a precision: Prodigal genes themselves reach only 72.6%. LMS:596-599 makes no such claim | LMS:596-599 | "the hit rates above are the direct evidence for them" |
| 520 | "Candidates sat on eukaryotic contigs more often than Prodigal genes (31.6% against 9.6%)" | ROUNDING. In S22, Prodigal plants+animals+fungi+other eukaryotes = 0.05865+0.01945+0.01215+0.0064 = 0.09665, i.e. 1,933/20,000 = 9.67%, which rounds to 9.7%. The candidate value 31.6% includes "other eukaryotes" the same way. LMS:662 also says 9.6 | S22 row prodigal | "31.6% against 9.7%" |
| 526 | "99.9% of genes at 30–49 aa" | ROUNDING. 0.99955 (13,253 of 13,259) is 99.95%, which rounds to 100.0%, not 99.9%. The same truncation is in LMS:729 | S29 row 30–50 below_t80 | ">99.9%" or "99.95%" |
| 528 | "22.4% had sequence hits only, mostly to near-identical UniProt entries in AFDB50" | Stronger than the source. LMS:760-762 says 89% are to AFDB50 but only 48% are at ≥90% identity, so "mostly near-identical" is not supported | LMS:760-762 | "mostly to UniProt entries in AFDB50 (89%), often near-identical (48% at ≥90% identity)" |

### Notes (not errors)
- l.206: "81 had no contig object ... one failed" matches LMS Table 1 (l.156-157), and 91 + 24,975 + 24,943 = 50,009 checks out (S9, S10, S11). However S11 says `accessions_unavailable_in_logan: 0` while 57 runs were lost in phaseb2. S10+S11 alone give only 25 "unavailable", so the comment's citation of S10/S11 for 81 is weak. LMS Table 1 is the source.
- l.412: "fold SD ≤0.002" holds for every reported vs-all model. The unreported length+3di+12st+mi model has 0.00214, and the antisense models reach 0.0027.
- l.242: CPU SU 264,000 = cpu_stage only (264,053). Adding the orchestrator gives 264,681.
- l.514: 0.21% and 0.45% are exact half-values (0.205%, 0.445%) rounded up. Acceptable.

## OK items (line: value)

- 206: 38,072,291; 7,754,394; 1,122,461; 863,841; 761,663; 13 biomes; 5–300 Mbp; N50 ≥150 bp; four size bins; 25/BioProject; 2×25,000; seed 102; 50,091 (91 pilot + 50,000); 81; 1; 50,009
- 210: ≥120 nt; 0.1%; 0.3.0; code 11; #103; -l 29; ≥30 aa; 90-aa floor (GTDB/phage)
- 212: Prodigal 2.6.3; 91 pilot (S9 91/91); every 100th (README); ≥90%; ≥98%; 1–29 bp; frame-undefined = 0 (tabL01)
- 216: 2d68d506; 10,000; 0.3.0; 750e2b92; 89,994 manifests (41,854+48,054+pilot); 8 GCD × 4; 1 M ORFs/chunk
- 218: 2.7×10⁻¹⁵; 2.7 M; 5×10⁻⁵; 100%
- 222: 0.01-bit bins; ties ½; nine length bins (tabL03)
- 224: 11.7 M; 9.5 M (1,760,915+7,751,712 = 9,512,627); 90 shards; 300 iter, lr 0.1, 63 leaves, early stopping, 5-fold GroupKFold (13_models.py:78-80)
- 228: 1%; 8.82 M; five alternatives; 1/(k+1)
- 232: 49.95 M (S14 sum 49,952,797); ≥100 aa; 4.58 M; thresholds 0.633/0.784/0.874/0.963 (tabL04)
- 234: 20,000 per arm; ≥0.90 / <0.80; Foldseek 10; --alignment-type 2; base:search; E ≤10⁻³; v6 2025_03; 66.7 M; coverage ≥0.8
- 238: 79,774; 80,000; 24,824; Kraken2 2.17.2; PlusPFP 2023-06-05; 1-in-1,000; 880,405; 10,269; ≤1,500/bin; 8,769
- 242: 5.28 T aa (2.445+2.831+pilot); 5.95 TB; 215–220 k aa/s (215,253, 220,293); 1.06 M GPU SU (1,057,234); 8,260 GCD-h; 264,000 CPU SU; 6,724 GCD-h; 861,000 SU (860,685); 176,000 SU/Tbp (175,994); 191,000 SU/Tbp (190,732)
- 395: 50,009; 13; 4.895 Tbp; 89,991,549,208 (tabL01 sum); 5.28 T; 47.55%; 26.97%; 8.71%; 0.98%; 881,348,601; 14.54%; ~59% (18.09+41.20 = 59.29); 18.1%; 41.2%
- 397-408 Table 2: 89.99 B; every share (0.979, 14.544, 47.548, 26.974, 8.707, 1.181, 0.067 → as printed, sum 100.00); every billion cell matches tabL01 to the printed precision, including the totals 36.64 / 16.28 / 37.08
- 412: 0.964; 0.580; 0.783; 36.6 B (36,628,754,907); 0.582; 0.860; 0.727; 0.880; ≤0.002; +0.020 (0.8797−0.8597)
- 414: 0.632; 0.877; 0.899; 0.793; 0.918; 0.942; 0.962 (S4 0.9616); 0.972
- 416: 0.612; 0.805; 0.896; 0.902; 0.960; 0.833; 0.988; 0.597; 0.852; 0.863; 0.612/0.898/0.940 (S5); 0.52; 0.66
- 418: 0.714→0.791; 0.331→0.559; 0.697→0.854; 0.471→0.751 (S6)
- 420: 0.819 plant; 0.920 wastewater; 9 of 13 in 0.89–0.92 (other 0.891 … wastewater 0.920); 0.906/0.893/0.898
- 424: 8.82 M; ρ length 0.33–0.45; aa 0.29–0.36; 12-state 0.06–0.13; 3Di −0.01 to 0.02 (−0.0137 to 0.0249); MI −0.04 to 0.01 (S7)
- 426: 5,722,417; 92.4; 91.3; 85.2; 81.8; 80.9; 16.7; top-3 98.8 vs 97.0
- 512: 49.95 M; 3,434,482; 47,754; 5.5% soil (5.51); 8.7% gut (8.66)
- 514: 41.9 (0.41875); 72.6; 0.21 (41); 0.45 (89); 0.36; 1/20,000; 58% (0.41875/0.7259 = 0.577); ~200× (204); 23–26% (0.2291, 0.2608)
- 516: 41.5; 1,123; 47.5; ≤0.5% (max 0.47%, antisense complete) (tabL06)
- 518: 41.5 AFDB50; 7–12 (7.4, 9.3, 11.1, 11.8); 20.7/21.2/4.0; 62.7/9.9/7.9 (S17); 38.1→70.6; 57.7→96.9 (S18); ~38% and ~1.2 M (weighted 1,170,088 of 3,065,354 = 38.2%); 3.07 M (2,759,781+305,573)
- 520: 64.5/30.1/1.9/1.9; 88.4/2.9/2.1/4.7 (S19); 73/71/51 (S24); 84.4% (LMS:660); 31.6% (S22); 2,525; 69.8/20.9/9.3 (tabL09); 38.2% intergenic_lo (S22)
- 522: 5,402; 49.6/42.2/5.0/3.3 (tabL08); −0.6 (−0.58); +42.1; 42.9/89.6 ATG; 61.9/19.2 no RBS (S26)
- 526: 880,405; 17.4%; 88/55/23/2 (S29); 3Di 0.9–1.5 vs 2.9–3.5 bits and aa Δ <0.3 bits from 50 aa (tabL07: max 0.27)
- 528: 6.8 (0.0685); 87.5; 28.3; 79.8; 10,269; 8,769; 70.8; 22.4 (S31); 2.7 → 95.1 (S32)

## C. Phage, smORF, reproducibility and Table 1 (sub-audit; full detail)
### Numeric audit: phage + smORF sections of genome_entropy.md

Audited 2026-10-16. Manuscript: `wt/ms-fix/manuscript/genome_entropy.md`, lines 60–72 (Table 1), 244–351 (Methods), 530–623 (Results). Read-only: nothing was edited.

Abbreviations: MI = `Validation/mutual_information/README.md`; IR = `Validation/Inphared/README.md`; SR = `Validation/smorfs/README.md`; IG = `AH/inphared_gadi/README.md`; ASM = `AH/smorfs/manuscript.md`; `data/` = `manuscript/data/`.

## Counts per class (about 150 numeric items checked)

| class | n |
|---|---:|
| OK | ~136 |
| MISMATCH | 2 |
| ROUNDING | 1 |
| UNSOURCED / dangling citation | 3 |
| SUPERSEDED | 0 |
| Claim stronger than its source (wording) | 8 |

No superseded value from results_inventory §6 appears: 0.383 bits, 52,057 / 1,797, OM869708 p 0.91 / 0.27, and "0.20–0.34 in every bin" are all absent.

## Non-OK findings

| # | MS line | quoted text | issue | correct value + source | suggested fix |
|---|---|---|---|---|---|
| 1 | 250 | "That gave 6,754,361 ORFs, of which 17,322 run off a contig end." | **MISMATCH** (inherited from a typo in the source). MI:37 says 17,322, but MI's own §2.1 table (MI:48–51) and `data/figPS01_label_audit.tsv.gz` give 4,875 + 4,694 + 8,753 = **18,322** contig-edge ORFs. The audit table sums to 6,754,361 in total, so it is internally consistent. | 18,322 (MI:48–51; figPS01_label_audit, `contig_edge == True`) | Change to 18,322 after confirming with the MI owner, and fix MI:37. |
| 2 | 351 | "For 99% of ORFs they agree to within 0.31 bits (3Di) and 0.35 bits (12-state), but single ORFs differ by up to 0.92 and 1.05 bits." | **MISMATCH (scope).** These are the 50-genome (6,284-ORF) test values, but the sentence is a general statement about platforms. Across the whole population, p99 is 0.317 / 0.354 and the maximum is **2.016 / 2.284 bits**, so the 0.92 / 1.05 maxima understate the worst case by more than half. | IG:201–202 (whole population: p99 0.317 / 0.354; max 2.016 / 2.284; median 0.0086); IG TL;DR line 34 ("maximum 2.0 bits") | Use the whole-population values ("within 0.32 and 0.35 bits; single ORFs differ by up to 2.0 and 2.3 bits"), or label these as the 50-genome values. |
| 3 | 586 (table, 300–599 aa) | "0.20" | **ROUNDING.** The recomputed n-weighted value, which the line 589 comment itself gives, is 0.1943, so it rounds to **0.19**. The 0.20 comes from IG's rounded class means (3.302 − 3.107 = 0.195). The other three bins are fine: 0.242 → 0.24, 0.339 → 0.34, 0.090 → 0.09. | `data/figP04_{phage,gtdb}_length_3di.tsv.gz`: matched phage 3.10743, bacteria 3.30170, difference 0.19427 | Change to 0.19, or the caption, IG and MS will disagree with the cited data. Also update the figP04 caption "(0.24, 0.34, 0.20 and 0.09)". |
| 4 | 67 (Table 1) | "AMD MI250X (ROCm, fp16), Setonix" | **UNSOURCED** device model. MI:34 and methods_provenance §2 say only "ROCm GPUs" / "Setonix ROCm fp16". MI250X is correct for Setonix GPU nodes but appears in no phage record. Same at line 341 ("AMD GPUs"). | MI:34; methods_provenance.md:40 | Accept as general Setonix knowledge, or write "AMD GPUs (ROCm, fp16)". |
| 5 | 336 | `<!-- src: …; sections/reproducibility_methods.md -->`; "on one NVIDIA V100" | **Dangling citation.** `sections/reproducibility_methods.md` does not exist in the worktree or in AH. "One" V100 is not stated in IG §5, which gives only the PBS job and SU. | IG:209–213 | Remove the dangling source, or cite IG `scripts/50_determinism.pbs`. Confirm the GPU count from that PBS script. |
| 6 | 272 | "every fifth part from 121 to 361, which points to a failed worker rather than to biology" | **Stronger than source.** IR:71 says "points to one failed worker or node, not to biology; **untested**", and IR:455 says "probably technical; the effect on composition was not checked". | IR:71, :455 | "…which suggests a failed worker rather than biology (untested)". |
| 7 | 307 | "Its labels equal the corrected labels ORF for ORF." | **Slightly stronger than source.** IG shows that the Gadi × legacy cross-tabulation reproduces the Setonix v2 × legacy table "cell for cell" (IG:20, :183–190). IG's §4 table says "equals v2 exactly", but no per-ORF join of Gadi and v2 labels is reported. | IG:17–21, :181–190 | "Its labels reproduce the corrected labels exactly (identical label cross-tabulation)", or cite a per-ORF comparison if one exists. |
| 8 | 536 | "Every estimate is far above the permutation null, whose 99th percentile is at most 0.0009 bits at 20 bins." | **Stronger than source for DNA entropy.** DNA H is 0.0029 bits, only about 3× the 0.00087 null. MI:147: "Every observed value **except DNA H** is more than 100× larger". All feature sets do reach p = 1/201. | MI:147 | "Every estimate except DNA entropy is more than 100 times the permutation null…", or "every estimate exceeds the null (p = 1/201)". |
| 9 | 580 | "Annotated phage ORFs have lower mean 3Di entropy than annotated bacterial ORFs at every length" | **Holds per bin, not per length.** The difference is positive in all 40 log-spaced bins of figP04 (minimum 0.012 bits, at 618–697 aa). At single-aa resolution, phage is higher at 115 of 910 lengths, all ≥447 aa, where per-length phage n is small (median n ≈ 325 at ≥600 aa). | `data/figP04_*_length_3di.tsv.gz` (recomputed) | "in every length bin from 90 to 999 aa". |
| 10 | 580 | "Annotated phage genes show the same separation as bacterial and archaeal genes" | **Loose.** The direction and widening are the same, but the phage matched–unmatched gap is smaller in every bin: phage 1.06 / 1.59 / 2.23 / 2.40 bits, bacteria 1.40 / 2.11 / 2.60 / 2.66, archaea 1.39 / 2.06 / 2.54 / 2.55. "The gap widens with length in every population" is OK. | figP04 data (recomputed) | "show the same pattern of separation as…". |
| 11 | 328 | "trained on nine nested feature sets" | **Inaccurate wording, copied from ASM §2.3.** The nine sets (length, nt, aa, 3Di, 12-state, length+nt+aa, length+3Di+12-state, length+aa+3Di, all five) are not nested: for example, nt and 3Di are disjoint singletons. | ASM Table 3 (lines 316–324); `data/figS01_models_by_split.tsv.gz` | "nine feature sets". |
| 12 | 601 | "genome_entropy therefore recovers a replication gene that the submitters annotated…" | **Mildly strong.** The model scores the 335-aa stop-to-stop ORF3, which contains the 326-aa submitted CDS, and does not define a gene model. The same paragraph later hedges correctly ("can flag such a locus for review"). D7 accepted the "recovers" framing. | decisions.md D7; `data/figP01_om869708_orfs.tsv.gz` | Optional: "genome_entropy flags the locus of a replication gene that…". |
| 13 | 540 | "Amino-acid composition therefore adds little once predicted structure is known." | **Mild.** MI supports "adds 0.057 bits (≈10% of 3Di's information)". "Therefore" follows from MI only. | MI:140 | Acceptable. Could add "(0.057 bits)". |

## OK items (compact)

**Table 1 (60–72).** All OK against methods_provenance §2: GTDB 189,715 of 189,801, 2,568,244,984 ORFs, 0.2.0 inferred at 90c8c15, 10,000, V100; archaea 10,122 and 54,858,398; LOGAN 50,009, 13 biomes (results_inventory L-1), 89,991,549,208, 30 aa, Prodigal 2.6.3, 0.3.0 at 750e2b92; phage MI 36,025 / 6,754,361 / ≈0.2.0 / 20,000; ranking 31,200 / 5,887,690 / 0.1.15 / 20,000; Gadi 0.3.0 at 45dd283, 10,000, V100; GMSC 43,726,909, 10–99 aa, 0.3.0, 50,000; model revision 2d68d506 not recorded for the Setonix phage runs.

**Methods, phage (244–311).**
- Genomes: 36,025; 2.316 Gb; 1,761–735,411 bp (IG:41); 7 Apr 2026 release, unconfirmed (MI:31); Pyrodigal-gv 0.3.2 and 3,336,744 CDS (MI:32).
- ORF floor: 90 aa and #103 (MI:36); 6,754,361 ORFs.
- Short CDS: 957,269 / 3,336,744 = **28.69% → 28.7%** (MI:59).
- Matcher: 90% / 98% / X wildcard (methods_provenance:21–27); 77,799 positives with the CDS under half the ORF (MI:60).
- MI set: 2,685,382 / 6,754,361 = **39.76% → 39.8%**.
- Ranking set: 49 of 361 parts, every fifth from 121 to 361 ((361 − 121)/5 + 1 = 49); 31,200 = 312 × 100; 5,887,690 ORFs; 2,338,039 / 5,887,690 = **39.71% → 39.7%**; **5,887,451 of 5,887,690** (IR:104).
- MI methods: 20 bins; 200,000 ORFs per class; 100 bootstraps (basic); 200 permutations at 10, 20 and 40 bins; 20 length strata; seed 20261014 (MI:75–87).
- Ranking model: XGBRF with 500 trees, depth 8, subsampling 0.8, GroupKFold(5) (IR:78; script lines 273–276); 8 intrinsic features (figPS01_feature_importance).
- Thresholds: p ≥ 0.8; probabilities span 0.11–0.89 (IR:140); replacement defaults 0.80 / 0.50 / 0.40 and score 0.50 p + 0.35 Δp + 0.15 overlap (`scripts/rank_missing_orfs.py`:118–132, 685–687); competitor is the greatest overlap (docstring line 400).
- Foldseek: 20,000 per arm; 10.941cd33; `--alignment-type 2 -e 1e-3`; database versions; coverage ≥0.8; mixture formula (IR:226–254).
- Gadi comparison: 0.3.0, 10,000, V100; GTDB 0.2.0 at the same size; 90–999 aa (figP04 provenance).
- OM869708 methods: NCBI efetch 2026-10-16 (the file mtime agrees); BLASTP 2.12.0, nr `blast-2026-08-01`, `-evalue 1e-5`, `-max_target_seqs 50`, job 50987603 (`work/om869708/blastp_nr.slurm`; 50 result rows).

**Methods, smORFs (313–332).**
- Sets: 33,647,794 / 10,079,115 (SR:4, 23); medians 80 / 70 aa (SR:25; tabS04).
- Internal stops: 1.71% / 1.58% (ASM:153).
- Encoding: 0.3.0, revision 2d68d506, 50,000, V100 (SR:34–40; figS01 provenance).
- Seven length bins (tabS01); HGB with early stopping and seed 42; 2 M train / 2 M test (SR:83).
- Clustering: 43,726,909 sequences; MMseqs2 14.7e284; linclust at 50% and cluster at 30%, both 80% coverage; weight 1/(size of the 50% cluster).
- Stripe bands: ±0.0055 bits, controls 0.03 bits away, k = 2, 3, 7 (figSS02 provenance).

**Reproducibility (334–351).**
- 50 genomes and 6,284 ORFs; PBS 181181761 (tabG01, IG:209–213).
- Repeat run: 100% bit-identical.
- Encoding size: 0.32 / 0.40 / 0.65% of ORFs → **0.3–0.7%**; maximum 0.0758 → 0.076.
- Gadi vs Setonix: 37.8%; median 0.0066 → 0.007; p99 0.31; maximum 0.92; r 0.99719 → 0.997; size-matched 37.7% / 0.310 / 0.920.
- Population agreement: 0.310 vs 0.317 and 0.99719 vs 0.99698 (IG:233).
- Class means move ~0.01 bits against a 1.46-bit gap (IG:239).
- 12-state p99 0.348 / max 1.05 match tabG01 (see finding 2 on scope).

**Results, phage (530–601).**
- 39.8% of 6,754,361; legacy labels 3 of 3,311,070 minus-strand ORFs (3 + 1,000,836 + 2,301,478 + 8,753 = 3,311,070); corrected 1,000,839 (figPS01_label_audit).
- 3Di MI: 0.5865 → 0.587 (CI 0.5837–0.5887 → 0.584–0.589); AUC 0.947.
- Pairs: protein + 3Di 0.6437 → 0.644, gain 0.057 (CI 0.055–0.058); other partners ≤0.026; DNA 0.0029 → 0.003.
- Null p99 ≤ 0.00087 → 0.0009.
- Length: 0.1298 → 0.130 (balanced; consistent with the balanced 3Di value); AUC 0.733.
- Retained given length: 0.471807 / 0.572457 = **82.4% → 82%**.
- Within length bins: 3Di 0.874 → 0.991; length 0.531–0.626.
- Ranking: strand gain 0.888 → 0.89; legacy model on corrected labels 0.7404 → 0.740; intrinsic 0.9764 / 0.9696; length only 0.7328 / 0.6598; length + 3Di 0.9527 / 0.9449; 3Di gain 0.7448 → 0.745; prevalence 0.397 → 40%.
- Candidates: 46,132 in 19,560 genomes, median 136 aa; 22,913 + 15,005 + 4,169 + 4,045 = 46,132.
- Foldseek any database: candidates 33.8% (CI 33.148–34.459 → 33.1–34.5); annotated 64.6%; low-score controls 2.195 / 0.24 / 0.095 → 2.2 / 0.24 / 0.10%; shuffled 0.015%.
- Mixture shares: any database (33.80 − 2.195) / (64.645 − 2.195) = 50.6% and (33.80 − 0.015) / (64.645 − 0.015) = 52.3% → "51–52%". Non-viral (6.135 − 0.085) / (18.04 − 0.085) = 33.7% and 6.135 / 18.04 = 34.0% → **"share of 34%"**.
- Non-viral rates: candidates 6.1% (CI 5.81–6.48 → 5.8–6.5); annotated 18.0%; every control ≤0.085% → "at most 0.09%".
- BFVD median identity 0.98.
- By context: antisense 9.9%, intergenic 66.5%, other-frame 61.6%.
- Replacement pairs: 4,102; 46.6% / 5.9%; hit rates 53.8% / 13.2% (IR:364, 375–376; foldseek_rates 0.53827 / 0.13196).
- 28.7% hidden by the floor; 310,360 positives matching a short CDS (MI:60).
- P04 bins: 0.242 / 0.339 / 0.090 match (0.194 is finding 3).
- OM869708 record: 5,753 bp; *Dipodfec virus* RodF1_38; *Microviricetes* (`OM869708_ncbi.gb`); UPW42009.1 at 4753..5730 (0-based 4752); CDS_0007 = ORF7 (5053–5650 contains 5092–5650, matches_cds 1).
- ORF scores: ORF3 4722–5730, same stop, 335 aa, 3Di 2.98049, p 0.889088; ORF7 198 aa, 1.20147, p 0.175411; rank 179 of 4,102.
- BLASTP: ORF7 has 0 rows; ORF3 row 1 is UPW42009.1 at 100%; rows 2–6 are 55.418 / 46.102 / 45.783 / 44.195 / 43.682% → "43.7–55.4%", one named microvirus (QXN75198.1, Microvirus mar41); best alignment 323 aa at E = 2.4e-127 → "2 × 10⁻¹²⁷".

**Results, smORFs (603–623).**
- Pooled AUC: length 0.640, 3Di 0.6053, 12-state 0.6015, nt 0.5037, aa 0.5254.
- Within bins: 3Di 0.526 (50–59) → 0.6257 (90–99); aa entropy 0.4564–0.4954 → 0.46–0.50, all <0.5; 10–29 aa 3Di 0.4727 with 11,382 / 80,578 sequences.
- Models: 0.6692 / 0.6993 / 0.7300; cluster splits 0.7275 / 0.7206 → 0.728 / 0.721; weighted at 30% 0.6948 → 0.695.
- Redundancy: 14.508 / 5.982 → 14.5 / 6.0; the 20 largest clusters are all positive at both thresholds (tabS03).
- Cluster weighting: 3Di 0.6122 → 0.5710; this is the largest drop (12-state −0.039, length −0.019).
- Density: enrichment at 2.3 / 3.6 and 1.0–1.6 bits (figSS01 caption).
- Stripes: 67.43 / 60.68% → 67 / 61% two-state, DP; 41.81 / 30.74% → 42 / 31% three-state, DPV.
- L34 / S14 identified by eye and hedged.
- Phage 3Di AUC 0.947.

## D. References (sub-audit; full detail)
### Reference audit: manuscript/references.bib (wt/ms-fix), 2026-10-16

Read-only. Sources: Crossref REST API (api.crossref.org/works/<doi>), PubMed, bioRxiv API, DataCite/Zenodo API, Hugging Face API, GitHub API.

## 1. Key resolution

- 29 distinct keys cited in genome_entropy.md, 29 entries in references.bib.
- **Unresolved keys: none.**
- **Bib entries never cited: none.**

## 2. Per-entry verification

No entry in the bib has a DOI except vanKempen2024, Heinzinger2024, Chikhi2024logan, Chen2016 and genome_entropy. Adding DOIs throughout is recommended; they are listed below.

| key | status | finding / correction | source |
|---|---|---|---|
| Akhter2013 | OK (add DOI) | Sci Rep 3:1033 (2013), authors correct. Content checked: computes Shannon's index of complete phage and bacterial genomes and of metagenomic sequences, so the citation fits "Shannon entropy ... composition of genomes and metagenomes". "Has long been used" rests on one 2013 paper; consider softening or adding a second citation. | doi:10.1038/srep01033, PMID 23301154 |
| Chikhi2024logan | CORRECTION | **Still an unpublished preprint.** bioRxiv reports no published version, and PubMed (PMID 40950083) indexes it as a preprint. The current version is v2 (2025-09-01), which carries the bib's author list. If you cite v2, set year = 2025. Fix author name: "Montoliu-Nerin, Merc{\`e}" (bib has M{\'e}rce). Title casing: "Logan: Planetary-Scale Genome Assembly Surveys Life's Diversity". | doi:10.1101/2024.07.30.605881; PMID 40950083; PMC12424806 |
| Hyatt2010 | OK (add DOI) | BMC Bioinformatics 11:119 | doi:10.1186/1471-2105-11-119 |
| Besemer2001 | CORRECTION (minor) | The full title continues "... in microbial genomes. Implications for finding sequence motifs in regulatory regions". Add issue 12. | doi:10.1093/nar/29.12.2607 |
| Delcher2007 | OK (add DOI) | Bioinformatics 23(6):673–679 | doi:10.1093/bioinformatics/btm009 |
| vanKempen2024 | OK | Nat Biotechnol 42(2):243–246. Published online 2023-05-08, issue 2024. Authors match. | doi:10.1038/s41587-023-01773-0 |
| Heinzinger2024 | OK | NAR Genom Bioinform 6(4):lqae150, 2024. Authors match (Crossref lists "Sanchez, Joaquin Gomez"). | doi:10.1093/nargab/lqae150 |
| modernprost | UNVERIFIED / needs decision | No paper or preprint exists. Revision 2d68d506d4ce8d59025c985fd00243a8860363c6 matches the current HF sha (repo created 2026-07-27, modified 2026-07-30). The modernprost-50M model card is an **unfilled template**: it gives no authors, licence or citation. The predecessor card (gbouras13/modernprost-base-deprecated) names the developers as **George Bouras and Victor Mihaila**, so add Mihaila as an author. The repo config (`ModernProst3Di12StModel`, `modernprost_3di12st`) supports the manuscript's "predicts 3Di and a 12-state alphabet" claim. Ask G. Bouras for the preferred citation. | https://huggingface.co/gbouras13/modernprost-50M ; https://huggingface.co/gbouras13/modernprost-base-deprecated |
| genome_entropy | CORRECTION | **Zenodo DOI 10.5281/zenodo.18312207 does resolve to genome_entropy, but it is the v0.1.1 version DOI (2026-01-20), titled "linsalrob/genome_entropy: zenododo" (sic).** The concept (all-versions) DOI is **10.5281/zenodo.18312206**, which resolves to the latest version, v0.3.0 (10.5281/zenodo.22225525, 2026-09-01). Archived versions: 0.1.1–0.1.16, 0.2.0 and 0.3.0. **v0.4.0 is not on Zenodo**, though the note says 0.1.15–0.4.0. Use the concept DOI 18312206 in the bib and in the Data availability section (manuscript line 678), or archive v0.4.0 and cite it. The entry also has **no year**, so it renders "n.d.". | https://zenodo.org/api/records/18312207/versions ; DataCite 10.5281/zenodo.18312207 |
| get_orfs | CORRECTION (was 'verify') | The repo exists (MIT licence). Its citation.cff gives doi 10.5281/zenodo.11218738, which is v0.2.0. The concept DOI is **10.5281/zenodo.11218737** (the README badge) and the latest version is v1.1.0 (10.5281/zenodo.11361300, 2024-05-28). Add doi = 10.5281/zenodo.11218737, year = 2024 and the title "get_orfs" (Zenodo: "linsalrob/get_orfs: v1.1.0"). The repo description is "translate a DNA sequence ... / Extract Open Reading Frames", so the bib subtitle is acceptable. | https://github.com/linsalrob/get_orfs ; Zenodo API |
| Parks2022 | OK (add DOI); consider update | NAR 50(D1):D785–D794 | doi:10.1093/nar/gkab776 |
| | suggestion | The manuscript uses GTDB **r232 (R11)**. A newer GTDB paper exists: Parks et al. 2026, "GTDB release 10: a complete and systematic taxonomy for 715 230 bacterial and 17 245 archaeal genomes", NAR 54(D1):D743–D754. Consider citing it alongside or instead. | doi:10.1093/nar/gkaf1040 |
| OLeary2024 | OK (add DOI) | Sci Data 11:732 | doi:10.1038/s41597-024-03571-y |
| Chklovski2023 | OK (add DOI) | Nat Methods 20(8):1203–1212 | doi:10.1038/s41592-023-01940-w |
| Varadi2024 | OK (add DOI) | NAR 52(D1):D368–D375 | doi:10.1093/nar/gkad1011 |
| Berman2000 | OK (add DOI) | NAR 28(1):235–242 | doi:10.1093/nar/28.1.235 |
| Sillitoe2021 | OK (add DOI) | NAR 49(D1):D266–D273. This paper describes CATH v4.3, so it matches "CATH50 (v4.3.0)". | doi:10.1093/nar/gkaa1079 |
| Kim2025bfvd | OK ('verify pages' resolved) | NAR 53(D1):D340–D347, 2025. Pages confirmed. Authors: Kim RS, Levy Karin E, Mirdita M, Chikhi R, Steinegger M. | doi:10.1093/nar/gkae1119 |
| Wood2019 | OK (add DOI) | Genome Biol 20:257 | doi:10.1186/s13059-019-1891-0 |
| Pedregosa2011 | OK | JMLR 12:2825–2830 (no DOI). Taken from the standard record; Crossref has no entry. | https://jmlr.org/papers/v12/pedregosa11a.html |
| Chen2016 | OK | KDD '16, pp. 785–794. Crossref title is "XGBoost". | doi:10.1145/2939672.2939785 |
| Cook2021 | OK (add DOI) | PHAGE (Therapy, Applications, and Research) 2(4):214–223. Authors match. | doi:10.1089/phage.2021.0007 |
| Bouras2023pharokka | OK (add DOI) | Bioinformatics 39(1):btac776. Authors match. | doi:10.1093/bioinformatics/btac776 |
| Bouras2025phold | CORRECTION (was 'verify') | **Now published:** Bouras G, Grigson SR, Mirdita M, Heinzinger M, Papudeshi B, Mallawaarachchi V, Green R, Kim RS, Mihaila V, Psaltis AJ, Wormald P-J, Vreugde S, Steinegger M, Edwards RA. "Protein structure-informed bacteriophage genome annotation with Phold." Nucleic Acids Research 54(1):gkaf1448, **2026** (online 2025-12-08). Change to @article with year 2026 and the key or year accordingly. The preprint was bioRxiv 10.1101/2025.08.05.668817. Note: Crossref spells the author "Mihalia", but the name is Mihaila. | doi:10.1093/nar/gkaf1448; bioRxiv pubs API |
| Larralde2022 | OK + suggestion ('verify' resolved) | JOSS 7(72):4296. Add doi 10.21105/joss.04296. The pyrodigal-gv README asks users of its gv extensions to also cite geNomad: Camargo AP et al. "Identification of mobile genetic elements with geNomad." Nat Biotechnol 42(8):1303–1312 (2024). The manuscript says "Pyrodigal-gv 0.3.2 [@Larralde2022]", so add this citation. | doi:10.21105/joss.04296; doi:10.1038/s41587-023-01953-y; https://github.com/althonos/pyrodigal-gv |
| Camacho2009 | OK (add DOI) | BMC Bioinformatics 10:421 | doi:10.1186/1471-2105-10-421 |
| Duan2024 | OK ('verify article number' resolved) | **Nature Communications** 15:7563 (2024), published 2024-08-31. It is not in Cell, so the bib's journal is correct. First authors match (full list: ... Zhu C, Kuhn M, Zhao X-M, Bork P, Coelho LP). | doi:10.1038/s41467-024-51894-6 |
| Washietl2011 | OK (add DOI) | RNA 17(4):578–594. Authors match. | doi:10.1261/rna.2536111 |
| Steinegger2017 | OK (add DOI) + suggestion | Nat Biotechnol 35(11):1026–1028. The manuscript uses `linclust` (line 330). Consider adding Steinegger M, Söding J. "Clustering huge protein sequence sets in linear time." Nat Commun 9:2542 (2018). | doi:10.1038/nbt.3988; doi:10.1038/s41467-018-04964-5 |

## 3. Citation-claim appropriateness (summary)

- All citations are used for appropriate claims: tools, databases and datasets are cited where they are used.
- Akhter2013 genuinely concerns Shannon entropy of genomes and metagenomes. The only weakness is the phrase "long been used", supported by a single paper.
- ModernProst: the claim of 3Di plus 12-state output is consistent with the model config, but nothing peer-reviewed or citable exists yet.
- LOGAN is still a preprint, so it should be labelled as bioRxiv in the reference list (it currently is).
- Pyrodigal-gv is attributed only to Pyrodigal; the geNomad citation is missing.
- GTDB r232 is cited to the 2022 paper; a 2026 GTDB paper is available.
- The `linclust` paper is not cited.
