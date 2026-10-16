# Results inventory (evidence matrix)

One row per result the manuscript may state. Each row gives the final accepted
value, where it is verified, its figure/table package, and whether the
current text matches. Built 2026-10-16 for checkpoint B of
[agent-handoffs#24](https://github.com/linsalrob/agent-handoffs/issues/24),
from read-only inventories of the Teams remote, this repo and agent-handoffs.
Where to find the data, and how each dataset was produced, is in
[`methods_provenance.md`](methods_provenance.md). Line-level problems in the
current text are in [`manuscript_audit.md`](manuscript_audit.md).

## Abbreviations

| abbreviation | meaning |
|---|---|
| **GR** | `Validation/gtdb/scientific_report.md` (main) |
| **MC** | `Validation/gtdb/matched_comparator.md` (main) |
| **MI** | `Validation/mutual_information/README.md` (main) |
| **IR** | `Validation/Inphared/README.md` (main) |
| **SR** | `Validation/smorfs/README.md` (main) |
| **AH** | `linsalrob/agent-handoffs:projects/genome_entropy_logan/` |
| **LMS** | `AH/manuscript/manuscript.md`, the LOGAN draft (authoritative LOGAN numbers, with tables `AH/manuscript/tables/S1–S35`) |
| **IG** | `AH/inphared_gadi/README.md` |

**MS status:**
- ✓ matches;
- **stale** (current text is wrong; correct value given);
- **absent** (not in the manuscript yet);
- n/a.

**Superseded results must not reappear.** They are listed at the end.

## 1. GTDB r232 reference population (Gadi; packages G01–G06)

| # | result | final value | verified in | package | MS status |
|---|---|---|---|---|---|
| G-1 | Scale | 199,837 representatives (189,715 bacterial encoded + 10,122 archaeal); 2,568,244,984 + 54,858,398 ORFs ≥90 aa | GR:16–19, §2 | Table 1 | absent |
| G-2 | Annotated fraction | 12.03% of bacterial and 17.26% of archaeal ORFs match a deposited CDS (309,065,661; 9,468,402) | GR §2, §4.1 (#99, #113) | G03 | absent |
| G-3 | Never-annotated genomes | 92,840 / 189,715 = 48.94% of bacterial genomes carry no CDS; excluded from every figure; ~23× high-3Di enrichment | GR §4, README | G01 | absent |
| G-4 | 3Di separates; protein barely does | bacterial CDS-bearing genomes: matched 3Di 3.123 vs unmatched 1.043 bits (gap 2.08); protein 3.979 vs 3.592 (gap 0.39); MI 0.960 vs 0.321; archaeal 3Di gap 1.99 | GR §4.1 | G01, G03 | absent (MS uses a test-case figure) |
| G-5 | log₂k ceiling | 87.6% (bac) / 84.1% (arc) of unmatched ORFs lie below log₂3 = 1.585 bits, against 3.3% / 3.7% of matched. D/V/P = 99.4% of residues (3 genomes only, D11) | GR §5 | G01 | absent |
| G-6 | DNA entropy uninformative | 1.930 vs 1.909 bits, overlapping IQRs | GR §4.1 | G03 | absent |
| G-7 | Length bias (#101) | Of the rise from 100 to 300 aa, estimator bias explains: protein 99–184% (bac) / 106–152% (arc); 3Di 12–15% (+0.11 of +0.88 bits); 12-state 7–9%. Only the empirical length × entropy reference is calibrated; i.i.d. z-scores are left-skewed | GR §4.2 | G02 | absent |
| G-8 | Candidate pool | 545,793 high-3Di (≥2.5 bits) non-shadow ORFs in annotated genomes (523,346 bac + 22,447 arc); 94.5% of the high-3Di unmatched pool were shadows | GR §1, §6 | G04a | absent |
| G-9 | Real-gene share, matched comparator | bacteria Swiss-Prot **10.4% (10.0–10.7)**, PDB 9.0%, CATH 9.9%; archaea 15.6 / 16.2 / 22.0%, estimator range 14.0–19.8% | GR §6.1, MC; #114 | G04b | absent |
| G-10 | Structural estimate | **~51,000–59,000** (bac 47.3–54.2k; arc 3.5–4.9k) | GR §6.1 | G04 | absent |
| G-11 | Direct-evidence floors | C3 any full-length **~27,400** (25,620 + 1,731); C7 strict **~5,400** (4,974 + 407). C3–C6 are a set, not a ladder; 13,996 BFVD-only bacterial candidates are invisible to C7 | GR §6.1 | G04c, Table 3 | absent |
| G-12 | Technical null | shuffled 3Di: ≤0.04% full-length hits over 523,346 candidates | GR §6.1 | G04 | absent |
| G-13 | Prodigal agreement | same-frame coincidence: candidates 47.4% / 55.4%, matched shadows 20.6 / 25.6%, annotated 94.2 / 96.1%; implies **~200,000** (190,597 + 9,468), 3.5–4.0× the structural estimate in bacteria and 1.9–2.7× in archaea (D2) | GR §6.1 | G05 | absent |
| G-14 | Convergence | strict set Prodigal-confirmed: 91.8% (21,617 / 23,540) bac, 95.9% (1,509 / 1,574) arc | GR §6.1 | G05 | absent |
| G-15 | What the missed genes are | strict set: metabolism 33.8% / 41.3%, mobile elements 13.6 / 11.8%; 8,065 distinct Swiss-Prot products | GR §6.1 | G06 | absent |
| G-16 | Examples | Cas5 completing a type I CRISPR–cas operon; SusC completing a PUL; 8 others. "RecF" should read "RecF/Rad50-family ABC ATPase" and is not used in a figure | GR §6.1; dossiers on Gadi/Teams | G06 | absent |
| G-17 | Assembly quality | **The excess is present in every assembly-quality stratum** (v7 matched comparator, quartiles of contig N50 and CheckM2 completeness, both domains, every database, both stratum schemes; smallest lower CI bound 3.0%; C3 and C7 excess positive everywhere). The pooled estimate is therefore not an artefact confined to poor assemblies, and the signal is not concentrated in the best ones. **Bacteria:** share higher in poorer assemblies on the combined Swiss-Prot/PDB/CATH readout, 17.4% → 10.3% by N50 and 19.9% → 9.0% by completeness, mostly from PDB/CATH; Swiss-Prot flat with N50 (9.2–12.1%), declining with completeness (14.2% → 7.8%). **Archaea:** no monotonic trend (CIs about ±5 points). **Prodigal share** falls with assembly quality in both domains (bacteria 44.8% → 31.1%, archaea 47.0% → 32.6% by N50). Quote scheme (i), strata by candidate genome (D15). Scheme (ii) agrees in direction on the combined readout and the Prodigal share in both domains, but not on every database: bacterial Swiss-Prot by N50 (9.8 → 9.2% under (i), 10.5 → 11.3% under (ii)) and archaeal PDB100 by completeness flip, both readouts with no trend under either scheme; bacterial cells differ by up to 3.3 points. In every stratum the matched comparator is the larger Prodigal background (intergenic_lo 8.2–13.7% bac, 13.0–18.9% arc), so the Prodigal shares follow the report's larger-background rule (#136). **Why** bacterial shares are higher in poorer assemblies is not established: assembly-driven false signal and less complete annotation both fit (report §11 item 9 could separate them) | `Validation/gtdb/assembly_strata_v7.md`, `matched_comparator/assembly_strata_v7_{bac,arc}.tsv` (#134); GR §6.1 (#128, #134) | GS* (to build from #134 tables) | absent |
| G-18 | Encoding failures | 39 genomes (0.021%) OOM on pathological ORFs (48k–766k aa); fixed by `--max-aa` (#112, 0.4.0) | GR §7 | Methods | absent |

## 2. LOGAN metagenomes (LOGAN agent; packages L01–L10, LS1, tabL01–09)

| # | result | final value | verified in | package | MS status |
|---|---|---|---|---|---|
| L-1 | Scale | 50,009 metagenomes (13 biomes); 4.895 Tbp; **89,991,549,208** six-frame ORFs ≥30 aa; 5.28 T aa | LMS:150–161, 282–288 | Table 1 | ✓ Results; **stale** in Abstract ("one million") and Introduction ("nine billion") |
| L-2 | ORF classes | antisense 47.55%, frameshift 26.97%, intergenic 8.71%, complete Prodigal+ 0.98% (881,348,601), partial 14.54%; ~59% open at a contig end | LMS:306–316; S3 | Table 2 | ✓ text; **stale** Table 2 (partial-shard) |
| L-3 | Length is a strong baseline | length AUC 0.964 / AP 0.58 overall | S1, S2 | L01 | ✓ (qualitative) |
| L-4 | Structure beyond length | within 100–149 aa: 3Di 0.896, aa 0.805, MI 0.902; 150–199: 0.960; 200–299: 0.988 | LMS:332–336 | L03 | numbers dropped; restore |
| L-5 | Nested models | AP 0.582 (length) → 0.860 (+3Di) → 0.880 (all), fold SD ≤0.002. Antisense negatives: length 0.632 → length + 3Di **0.877** → all features 0.899 (S4; LOGAN agent, #24) | LMS:44–45, 396–410; S4 | L02 | absent (numbers) |
| L-6 | Model sample | 11.7 M ORFs (9.5 M complete) from all 90 shards | LMS:264–265 | Methods | **stale** ("8.2 M, 61 shards") |
| L-7 | Fragments | 5′-open AUC 0.791 / AP 0.559; 3′-open 0.854 / 0.751 | LMS:433–438 | L04 | ✓ |
| L-8 | Six-frame specificity | 8.82 M genes, 5,722,417 loci; 3Di ranks the Prodigal frame first in **92.4%** (12-state 91.3, length 80.9, aa 81.8, chance 16.7); median Δ 1.99 bits | LMS:452–493; S8 | L05 | **stale** (92.5%; 7.74 M) |
| L-9 | No inheritance between frames | ρ length 0.33–0.45, aa 0.29–0.36; 3Di −0.01 to 0.02 (S7: −0.0137 to 0.0249); MI −0.04 to 0.01; **12-state 0.06–0.13 (weak)** | LMS:457–465 | L05 | **stale** ("neither correlated") |
| L-10 | Biome/tranche replication | 3Di AUC at 100–149 aa: 0.819 (plant) to 0.920 (wastewater); tranches 0.906 / 0.893 / 0.898 | S21 | LS1 | absent |
| L-11 | Candidates | 49.95 M intergenic ORFs ≥100 aa scored; 3.43 M at P0.90 in 47,754 metagenomes; 5.5% (soil) to 8.7% (gut) | LMS:532–554 | L06, L07 | ✓ rates; thresholds table absent |
| L-12 | Structural validation, five databases | full-length hits: candidates **41.9%**, Prodigal genes 72.6%, low-score intergenic **0.21%** (41/20,000), antisense **0.45%** (89/20,000), frameshift 0.36%, shuffled 1/20,000 (rates corrected by the LOGAN agent, #24); ~200× controls; **~1.2 M** candidates with homologues. The 0.90 precision target is calibrated on complete ORFs only: complete candidates 41.5%, 5′-open 47.5% (1,123 of 20,000; Table S28) | LMS:569–616; S33; #24 LOGAN comment | L06 | **stale** (4-DB: 19.3%, 56.1%, 0.03%, 0.13%, "several hundred times", 0.5 M) |
| L-13 | Evidence classes | structure-only 21.2%, both 20.7%, sequence-only 4.0% | LMS:606–607 | L07 | **stale** (16.6 / 2.7%) |
| L-14 | Target superkingdom | bacterial 64.5%, eukaryotic 30.1%, archaeal 1.9%, viral 1.9% (Prodigal 88.4 / 2.9 / 2.1 / 4.7); the earlier 38% viral was a BFVD artefact | LMS:620–630 | L07 | **stale** |
| L-15 | Eukaryotic candidates are on eukaryotic contigs | 69.8% on Kraken2-eukaryotic contigs | LMS:657–678 | L08 | absent |
| L-16 | Why Prodigal skipped candidates | coding score median −0.6 vs +42.1 | LMS:~700–718 | L09 | absent |
| L-17 | Low-scoring Prodigal genes | 17.4% of Prodigal genes score low; 6.8% full-length structural vs 87.5% (6.8% corrected from 6.9%, #24) | LMS:725–771 | L10 | absent |
| L-18 | Compute | ~1.06 M GPU SU + 264k CPU SU; ~167k GPU SU were re-encoding after an inode-quota failure; 176k SU/Tbp excludes it; 215–220k aa/s/GCD | LMS:783–791 | Methods | ✓ but caveat missing; FLOPs claim unsupported (D12) |

## 3. Phage (Setonix integrator: P01–P03, P05; Gadi: P04, determinism)

| # | result | final value | verified in | package | MS status |
|---|---|---|---|---|---|
| P-1 | Label correction | the legacy matcher marked 3 of 3.31 M minus-strand ORFs as genes; corrected labels: 2,338,039 positives (39.7%) in the ranking set; Gadi run 2,685,382 (39.76%), 1,000,839 on the minus strand | MI:7; IR:7–13, 102; IG:14–25 | Methods | absent |
| P-2 | Mutual information | I(3Di; label) **0.587 bits**, AUC 0.947; protein + 3Di 0.644 (+0.057); permutation null <0.003; 82% survives conditioning on length | MI:10–16 | P01 | absent |
| P-3 | Ranking | intrinsic features (entropies + lengths) AUC **0.976** / AP **0.970** vs length-only 0.733 / 0.660; position and strand features removed (strand shortcut) | IR:14–24, 130–134 | P02 | **stale** (Methods describe positional XGBoost) |
| P-4 | Candidates | 46,132 putative missing ORFs; 4,102 replacement pairs | IR:29–30 | P02, P05 | absent |
| P-5 | Foldseek validation | full-length, 4 DB / non-viral: annotated 64.6 / 18.0%, candidates 33.8 / 6.1%, shuffled 0.015 / 0%; real-gene share ~51% (4 DB) / ~34% excluding BFVD | IR:34–41, 266–279 | P02 | absent |
| P-6 | OM869708 (*Dipodfec virus* RodF1_38) | ORF3 (+, 335 aa, 3Di 2.98) p **0.889**; re-annotation CDS_0007 = ORF7 p **0.175**; replacement 179 / 4,102. The submitters annotated ORF3's gene as a replication initiator (UPW42009.1, 4753..5730), and Pharokka + phold dropped it. BLASTP: top hit is the genome's own record (not independent); next hits are microvirus replication initiators at 43–55% identity (E ≤ 2 × 10⁻¹²⁷); ORF7 has no hit (D7) | `work/phage_rank`, NCBI record, BLASTP job 50987603; #122 `figP01_om869708_*` | P01d | **stale** (0.91 / 0.27; "independent" BLAST hit is self) |
| P-7 | Phage vs bacteria | matched phage CDS sit below matched bacterial CDS in 3Di in every length bin: 0.24, 0.34 and 0.20 bits in the bins below 600 aa, but only **0.09 bits at 600–999 aa**. The earlier "0.20–0.34 in every bin" used a "≥600 aa" bin that lumped in ORFs ≥1,000 aa (Gadi, #131). Cause untested (D11) | IG:112–117; #131 | P04 (#131) | absent |
| P-8 | Determinism | Gadi repeat run bit-identical; 10k vs 20k encoding moves 0.3–0.7% of ORFs by <0.08 bits; Gadi vs Setonix p99 0.31 bits, r 0.997, class means ±0.01 | IG §5 | supp table | absent |
| P-9 | Phage ORF floor | effective 90 aa: 28.8% of annotated phage genes invisible (D6) | IR:73–78; #103 | Methods | absent |

## 4. GMSC smORFs (Gadi; S01+)

| # | result | final value | verified in | package | MS status |
|---|---|---|---|---|---|
| S-1 | Dataset | 43,726,909 smORFs, 10–99 aa (33,647,794 positive / 10,079,115 negative) | SR:3, 19–25 | S01 | absent |
| S-2 | Limit of the method | AUC: length 0.640, 3Di 0.605, 12-state 0.602; all features 0.730 random split, 0.728 / 0.721 under 50% / 30% cluster splits; 3Di 0.47 at 10–29 aa | SR:75–90 | S01+ | absent |
| S-3 | Stripe families | L34/S14 picked by eye (D11) | `AH/smorfs/manuscript.md` | S* | absent |

## 5. Methods facts every section needs

| item | value | source |
|---|---|---|
| ORF floor | effective 90 aa for GTDB, phage and MI (`-l` was amino acids, #103/#104); 30 aa for LOGAN (`-l 29`) | GR:135–141; #103 |
| `in_genbank` rule | same record and strand; compatible codon phase; overlap ≥90% of the *shorter* interval; ≥98% identity of coordinate-aligned translations (`X` wildcard); **no stop-codon check**, so a short CDS inside a long ORF can match. Not "overlaps a CDS". LOGAN uses Prodigal labels instead | `src/genome_entropy/io/genbank.py` (`evaluate_orf_genbank_cds_match`); see `methods_provenance.md` §1 |
| Model | `gbouras13/modernprost-50M`; revision 2d68d506 recorded for LOGAN and smORFs only | LMS:222; methods_provenance.md |
| Per-dataset configuration | see `methods_provenance.md` §2 (D9) | |

## 6. Superseded: do not use

| superseded | replaced by | why |
|---|---|---|
| GTDB v6 real-gene shares 23–36%, ~128–149k, C3 ~65.8k, C7 ~20,350, Prodigal ~244k / 1.7× | G-9 to G-13 | clean comparator was unmatched after the frame filter (#109) |
| GTDB first-draft 3,562,431 candidates, "candidates and shadows indistinguishable" | G-8 | coordinate defect; `in_genbank` used as an annotation proxy |
| GTDB 20-chunk figures (`figures/superseded_20chunk/`) | G01 | sample |
| "The signal is in high-N50 genomes" (archaea 43.5% vs 32.6%, GR §6.1) | G-17 | any-hit readout with all shadows; same-frame and truncated shadows concentrate in fragmented assemblies (withdrawn, #128) |
| Pre-matching assembly strata (`supplementary_analyses.md` §7), e.g. bacterial Swiss-Prot 28% → 20% with N50 | G-17 (v7 strata, #134) | comparator imbalance; the N50 decline does not survive re-matching, the completeness decline survives at about half the size |
| Phage 3Di gap "0.20–0.34 bits in every length bin" | P-7 | top bin mixed 600–999 aa with ≥1,000 aa |
| Bacterial Swiss-Prot share 10.3% | 10.4% | rounding (#114) |
| Phage MI 0.383 bits; legacy in_genbank tables on Teams (`Validation/mutual_information/*`) | P-2 | minus-strand label bug |
| Phage ranking 52,057 candidates / 1,797 replacements (`Validation/Inphared/rank_orfs/` on Teams) | P-3 to P-5 | strand shortcut |
| LOGAN 4-database Foldseek rates (19.3%, 56.1%, 38% viral), `figures_candidates_4db/` | L-12 to L-14 | AFDB50 added (five databases) |
| LOGAN partial-shard numbers (61 / 77 shards, Table 2 with 620 M) | L-2, L-6, L-8 | final all-shard run |
| January 2026 test cases (`test_cases/`, current Fig. 1) | G01 | superseded pilot |
| OM869708 p 0.91 / 0.27 | P-6 | model trained on legacy labels |
