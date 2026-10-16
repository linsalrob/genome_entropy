<!-- Draft Results text for GTDB r232, Setonix section drafter, agent-handoffs#24 checkpoint D. The integrator merges it into genome_entropy.md. -->
<!-- Abbreviations in src comments: GR = Validation/gtdb/scientific_report.md; MC = Validation/gtdb/matched_comparator.md; AS = Validation/gtdb/assembly_strata_v7.md; SA = Validation/gtdb/supplementary_analyses.md; v7 = Validation/gtdb/matched_comparator/{bac,arc}_v7_matched.tsv. -->

## Real proteins and spurious ORFs occupy different structural-entropy regimes

We called every six-frame ORF of at least 90 aa in 199,837 GTDB r232 species representatives: 2,568,244,984 bacterial and 54,858,398 archaeal ORFs. Of these, 12.03% and 17.26% match a deposited CDS. <!-- src: GR §1, §2 -->

In genomes that carry any annotation, matched and unmatched ORFs separate on 3Di entropy, not protein entropy (Fig. 1a–c).
- **3Di entropy:** the mean differs by 2.08 bits in bacteria (3.123 against 1.043) and by 1.99 bits in archaea.
- **Protein entropy:** the means differ by only 0.39 and 0.37 bits.
- **Mutual information:** 3Di–12-state MI differs threefold (0.960 against 0.321 bits in bacteria). <!-- src: GR §4.1 item 1; figG01 caption -->
- **DNA entropy:** it does not separate the classes (1.930 against 1.909 bits). <!-- src: GR §4.1 item 3 -->

Matched ORFs are also about twice as long (376 against 178 aa), so part of the 3Di gap is length (next section). <!-- src: GR §4.1 item 2 -->

Unmatched ORFs pile up against a sharp edge at 1.585 bits (Fig. 1b). This is log₂3, the maximum entropy of a string that uses only three distinct states. It is an arithmetic ceiling, not a biological threshold, and it falls at the same place in both domains. <!-- src: GR §5; figG01 caption --> Over the full population, ORFs below log₂3 are:
- **bacteria:** 87.6% of unmatched against 3.3% of matched ORFs;
- **archaea:** 84.1% against 3.7% (Fig. 1d). <!-- src: GR §4; figG01_population_bands.tsv.gz -->

In three genomes, states D, V and P make up 99.4% of residues below the line. <!-- src: GR §5; RI G-5 --> We report this as an observation only. Whether state D marks disorder has not been tested; DSSP or a disorder predictor could test it. <!-- decision D11 -->

**Never-annotated genomes.** 48.9% of bacterial and 45.7% of archaeal representatives carry no deposited CDS. <!-- src: GR §2, §6 --> Their ORFs are not low-3Di noise: 19.1% (bacteria) and 28.3% (archaea) reach 3Di ≥ 2.5 bits. Unmatched ORFs in annotated genomes reach that level only 0.83% and 1.25% of the time, a 23-fold difference in both domains. <!-- src: GR §4; GR §6 "The entropy-matched intergenic arm" --> These genomes therefore appear to hold many unlabelled genes, and we exclude them from every comparison (Methods).

## Entropy rises with length mostly as estimator bias for protein, little for 3Di

Between 100 and 300 aa, mean protein entropy of matched ORFs rises by 0.08 bits (bacteria) and 0.09 bits (archaea). That is less than the 0.11 bits that plug-in estimator bias alone produces (Fig. S1). <!-- src: figG02 caption; figG02_mean_curves.tsv.gz --> The apparent length trend in protein entropy is therefore an artefact.

3Di entropy rises by 0.88 and 0.76 bits over the same range. Only 0.11 and 0.10 bits of that (12% and 14%) is bias, so longer proteins genuinely use more predicted structural states. <!-- src: figG02 caption; figG02_mean_curves.tsv.gz -->

Only a reference built from real ORFs of the same length is calibrated: its z-scores have mean ≈ 0 and SD 1.00. Against i.i.d. draws, the SD of z is 3.9–6.6 (Fig. S2). <!-- src: figGS02 caption; figGS02_summary.tsv.gz -->

## Missed genes in GTDB

**Candidates.** In annotated genomes we kept unmatched ORFs with 3Di ≥ 2.5 bits: 9,598,500 bacterial ORFs. Of these, 94.5% overlap a deposited CDS (shadows). That leaves 523,346 bacterial and 22,447 archaeal candidates (Fig. 4a). <!-- src: GR §6 table; figG04_funnel.tsv.gz --> Candidate burden per genome rises with genome size (Spearman ρ = 0.55 in bacteria) and is almost flat against contig N50 (ρ = 0.10) (Fig. S8). <!-- src: figGS03 caption; figGS03_burden_rho.tsv.gz -->

**Structural homology against a matched comparator.** High 3Di entropy alone is not evidence of a gene. Above 3.38 bits nearly every ORF hits a structure database, because a sequence-only encoder emits confident 3Di for any protein-like sequence. <!-- src: GR §6.1 --> We therefore compared candidates with antisense and frameshift shadows matched on length, 3Di entropy and truncation, counting only full-length hits (Methods).

For the technical null, we shuffled the 3Di strings of the bacterial candidates; at most 0.04% then had full-length hits. <!-- src: GR §6.1 --> In every database, candidates fall between the comparator and annotated CDS. Against Swiss-Prot, the full-length hit rates are 10.3% for bacterial candidates, 7.4% for the comparator and 35.3% for annotated CDS (Fig. 4b). <!-- src: GR §6.1 table; v7 -->

The implied real-gene share in bacteria is 10.4% (95% CI 10.0–10.7) for Swiss-Prot, 9.0% (8.6–9.5) for PDB100 and 9.9% (9.3–10.5) for CATH50 (Fig. 4c). <!-- src: GR §6.1; MC "Result"; v7 share_* --> In archaea it is 15.6% (13.4–17.9), 16.2% (13.7–18.7) and 22.0% (18.5–25.4). <!-- src: GR §6.1; v7 -->

The bacterial share barely moves when the comparator is weighted differently (10.3–10.6%). The archaeal Swiss-Prot share ranges from 14.0% to 19.8%, because few shadows match the long, high-3Di candidates. <!-- src: GR §6.1; MC "Result", "Reading" 2 -->

Together these imply **~51,000–59,000 protein-coding genes absent from GenBank annotation**: 47,300–54,200 in bacteria and 3,500–4,900 in archaea. <!-- src: GR §6.1; v7 structural_estimate; decision D1 -->

The estimate assumes a two-component mixture, and its CIs cover binomial noise only. Any real genes among the comparator shadows would make it an underestimate. <!-- src: GR §10; MC "Caveats" -->

**Direct evidence.** Without the mixture model, the candidate excess over the comparator gives two floors (Fig. 4d, Table 3):
- **any full-length structural hit (C3):** ~27,400 (25,620 bacterial + 1,731 archaeal);
- **the strict rule (C7):** ~5,400 (4,974 + 407). <!-- src: GR §6.1; v7 ladder_C3/C7_excess -->

Both are conservative. 18,566 scaled comparator ORFs also meet C7. C7 also cannot see the 13,996 bacterial C3 candidates whose only full-length hit is viral (BFVD). <!-- src: GR §6.1 -->

**Table 3. Direct evidence for GTDB candidates against the matched comparator.** For each criterion the table gives the number of candidates, the comparator count and the excess. The comparator count is weighted by draw multiplicity and scaled to the candidate arm. The excess is a floor on the number of real genes. C3–C6 are each C2 plus one requirement and are not nested (in archaea C6 exceeds C4); C7 is the strict combined rule (Methods). C2 has no excess by construction, because the comparator is matched on truncation.

| criterion | bacteria: candidates | bacteria: comparator | bacteria: excess | archaea: candidates | archaea: comparator | archaea: excess |
|---|---:|---:|---:|---:|---:|---:|
| C1 all candidates | 523,346 | 523,346 | 0 | 22,447 | 22,447 | 0 |
| C2 not contig-truncated | 483,767 | 483,767 | 0 | 19,835 | 19,835 | 0 |
| C3 C2 + full-length hit, any database | 90,658 | 65,038 | 25,620 | 5,141 | 3,410 | 1,731 |
| C4 C2 + full-length in ≥2 databases | 44,325 | 32,868 | 11,457 | 2,777 | 1,872 | 905 |
| C5 C2 + full-length in Swiss-Prot, PDB100 or CATH50 | 76,662 | 57,343 | 19,319 | 4,396 | 3,030 | 1,366 |
| C6 C2 + structure-only full-length hit | 81,257 | 59,429 | 21,828 | 4,439 | 3,030 | 1,409 |
| C7 strict combined rule | 23,540 | 18,566 | 4,974 | 1,574 | 1,167 | 407 |

<!-- src: Validation/gtdb/matched_comparator/{bac,arc}_v7_matched.tsv ladder_C*_{candidates,comparator_scaled,excess}; GR §6.1 ladder table -->

**An independent gene caller.** GTDB's Prodigal annotation uses neither GenBank nor structure. It calls a gene with the same stop and strand for these shares of each arm (Fig. S7a):

| arm | bacteria | archaea |
|---|---:|---:|
| candidates | 47.4% | 55.4% |
| matched comparator | 20.6% | 25.6% |
| annotated CDS | 94.2% | 96.1% |

<!-- src: GR §6.1; v7 prodigal_* -->

The implied Prodigal share is 36.4% (36.2–36.6) in bacteria and 42.2% (41.1–43.2) in archaea, or **~200,000 genes** (190,597 + 9,468). <!-- src: GR §6.1; v7 prodigal_share_larger_bg --> We treat this as the upper bracket. It is 3.5–4.0 times the structural estimate in bacteria and 1.9–2.7 times in archaea. <!-- src: GR §6.1; MC "Result" -->

We cannot resolve this gap with these data. Prodigal may over-call: the 20.6% comparator rate is its false-positive rate on ORF-shaped non-genes. The structural assay may also undercount, because it finds full-length hits for only 30–35% of known genes, against 94–96% for Prodigal. <!-- src: GR §6.1 "Where the two estimates disagree"; decision D2 --> Prodigal also shares the structural assay's preference for coding-like composition, which the matched comparator bounds but does not remove. <!-- src: GR §10 --> The ~5,400 strict floor depends on neither estimate. <!-- src: GR §6.1 -->

The two lines of evidence converge on the same ORFs. Prodigal calls 91.8% of the bacterial C7 set (21,617 of 23,540) and 95.9% of the archaeal set (1,509 of 1,574), although C7 never uses Prodigal. <!-- src: GR §6.1; SA §10 item 10 --> Part of this agreement is shared sensitivity to protein-like sequence. Against the earlier, unmatched comparator, shadows meeting a C7-like clause coincided with Prodigal 56.7% (bacteria) and 68.4% (archaea) of the time. <!-- src: SA §10 item 10 table and reading (pre-v7 comparator) -->

**What the missed genes are.** In the C7 set (Fig. S9a):
- metabolism accounts for 33.8% of bacterial and 41.3% of archaeal candidates;
- mobile elements account for 13.6% and 11.8%. <!-- src: GR §6.1 functional table; figG06 caption --> Mobile elements are genuinely under-annotated in GenBank.

The 23,393 bacterial C7 candidates with a Swiss-Prot product carry 8,065 distinct products. <!-- src: GR §6.1 --> These classes are homology labels, not evidence of function, and C7 favours well-characterised folds. <!-- src: figG06 caption -->

Ten worked examples were chosen across functional classes (Fig. S9b). Each one:
- meets C7;
- is called by Prodigal;
- lies between two deposited CDS;
- sits in a genome at least 95% complete. <!-- src: GR §6.1 "Examples"; figG06 caption -->

Two fill the only gap in an otherwise complete module:
- **Cas5** in GCA_049515795.1. A 226-aa candidate lies 5 bp from *cas7* and 96 bp from *cas1*, in a co-oriented type I CRISPR–*cas* operon annotated with Cas6, Cas7, Cas8a1, Cas3′, Cas1 and Cas2 but not Cas5, which the type I Cascade complex requires. <!-- src: GR §6.1 "Examples"; figG06_examples.tsv.gz; figG06_neighbours.tsv.gz -->
- **SusC** in GCA_900230115.1. A 917-aa candidate fills the 3,029-bp gap in a polysaccharide utilisation locus that is annotated with SusD and SusE but lacks SusC. <!-- src: GR §6.1 "Examples"; figG06 data -->

The other eight include a RecF/Rad50-family ABC ATPase. <!-- src: RI G-16 --> These examples illustrate the kind of gene found; they are not evidence for the rate. Structural homology shows that an ORF is protein-like, not that it is expressed or functional.

**Assembly quality.** We recomputed the estimate within quartiles of contig N50 and of CheckM2 completeness, keeping each comparator draw with its candidate's genome (scheme (i); Methods). <!-- src: AS; decision D15 --> <!-- integrator: no figure/table number is assigned to the strata; cite assembly_strata_v7 tables or add a supplementary table. --> The excess is present in every stratum, in both domains and for every database, with the smallest lower CI bound at 3.0%. The C3 and C7 excesses are positive throughout. <!-- src: AS "Reading" 1 --> The pooled estimate is therefore not an artefact of poor assemblies.

**Bacteria.** The share is higher in poorer assemblies. Counting a full-length hit in any of Swiss-Prot, PDB100 or CATH50, it falls from 17.4% to 10.3% across N50 quartiles and from 19.9% to 9.0% across completeness quartiles. <!-- src: AS "Reading" 2 --> The decline comes mostly from PDB100 and CATH50. Swiss-Prot alone is flat with N50 (9.2–12.1%). <!-- src: AS "Reading" 2 -->

**Archaea.** There is no monotonic trend; the CIs are about ±5 points. <!-- src: AS "Reading" 3 -->

**Prodigal.** The Prodigal share falls with assembly quality in both domains. By N50 it falls from 44.8% to 31.1% in bacteria and from 47.0% to 32.6% in archaea. <!-- src: AS "Reading" 4 -->

**Sensitivity analysis.** Placing every ORF by its own genome (scheme (ii)) gives the same direction for the combined share and the Prodigal share in both domains. It does not agree on every database or cell:
- bacterial Swiss-Prot by N50 changes endpoint direction (9.8% → 9.2% under (i), 10.5% → 11.3% under (ii));
- archaeal PDB100 by completeness also changes endpoint direction;
- neither readout shows a trend under either scheme;
- bacterial cells differ by up to 3.3 points. <!-- src: AS "Reading" 6 -->

Why bacterial shares are higher in poorer assemblies is not established. Assembly-driven false signal and less complete annotation of lower-quality genomes both fit the data. <!-- src: AS "What it does and does not establish" -->
