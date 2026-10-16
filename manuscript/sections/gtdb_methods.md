<!-- Draft Methods text for GTDB r232, Setonix section drafter, agent-handoffs#24 checkpoint D. The integrator merges it into genome_entropy.md. -->
<!-- Abbreviations in src comments: GR = Validation/gtdb/scientific_report.md; MC = Validation/gtdb/matched_comparator.md; AS = Validation/gtdb/assembly_strata_v7.md; MP = manuscript/methods_provenance.md; RI = manuscript/results_inventory.md. -->

## GTDB r232 reference genomes

### Genomes, ORFs and labels

We analysed the species representatives of GTDB release 232 (R11-RS232) [@Parks2022]: 189,801 bacterial and 10,122 archaeal genomes. <!-- src: GR:3; GR §2 --> GenBank records were downloaded with NCBI `datasets` 18.36.0 [@OLeary2024]. Forty-seven bacterial assemblies were no longer served and 39 failed to encode (below). That left 189,715 bacterial genomes with 2,568,244,984 ORFs and 10,122 archaeal genomes with 54,858,398 ORFs. <!-- src: GR §2, §3 -->

Counted directly from the GenBank records, 92,840 bacterial (48.9%) and 4,630 archaeal (45.7%) genomes carry no deposited CDS. <!-- src: GR §2; GR §6 table --> Every ORF in such a genome is unmatched by construction, whatever it encodes. We therefore excluded these genomes from every matched-versus-unmatched comparison, every figure and the candidate analysis. That left 96,875 bacterial and 5,492 archaeal genomes. <!-- src: GR §4; figGS03 caption -->

`genome_entropy` enumerated ORFs stop to stop in all six frames. The intended 30-aa minimum was passed to `get_orfs -l` as 90, which reads it in amino acids (genome_entropy #103). The effective floor was therefore 90 aa, and every GTDB statistic is over ORFs of at least 90 aa. <!-- src: GR §3 (GR:135–141); MP §1 -->

ORFs were labelled *matched* or *unmatched* against the deposited CDS with the corrected matcher (*Annotation labels*). The matcher has no stop-codon check, so "unmatched" is not ground truth for non-coding. The six-frame caller recovers only about 90% of deposited CDS. <!-- src: MP §1; GR §6 "Sensitivity of the ORF caller"; GR §10 -->

### Encoding and entropies

ORFs were encoded with `genome_entropy` 0.2.0 (commit `90c8c15`, output schema 2.2.0). The version is inferred from installation and clone times, not logged. <!-- src: MP §2, §4 item 2 --> The encoder was ModernProst (`gbouras13/modernprost-50M`, revision 2d68d506) [@modernprost]. It predicts 20-state 3Di [@vanKempen2024] and a 12-state alphabet from sequence alone. <!-- src: MP §1; GR:4 --> Encoding ran on NVIDIA V100-SXM2-32GB GPUs (NCI Gadi), with an encoding size of 10,000 and four processes per GPU (Table 1). <!-- src: GR §3; MP §2 --> Thirty-nine genomes (0.021%) failed with out-of-memory errors on pathological ORF calls of roughly 48,000–766,000 aa. They are absent from the dataset; the `--max-aa` option added in 0.4.0 prevents this failure. <!-- src: GR §7; RI G-18 -->

Entropies and MI are as defined under *Entropy features*, except that protein entropy counted `X` as a 21st symbol. <!-- src: figG01 caption "Definitions"; GR §4.2 ("The protein alphabet is 21 symbols") --> Population statistics were streamed over every ORF row. Counts and means are exact. Medians and quartiles come from histograms with 10⁻⁴-bit bins. <!-- src: GR §4.1 --> Fig. 1a–c and Fig. S2 use a systematic sample of every 300th bacterial and every 30th archaeal row (4,893,192 and 1,138,714 ORFs after exclusion). <!-- src: GR §4 "Figures" -->

**Length references (genome_entropy #101).** To separate estimator bias from biology, we built two references: <!-- src: GR §4.2 -->
- **i.i.d.:** the plug-in entropy of 400,000 multinomial draws at each of 671 lengths, using each domain's pooled residue composition (351 genomes);
- **empirical:** the length × entropy distribution of every ORF, in 1-aa length bins up to 999 aa and coarser bins above. <!-- src: GR §4.2 -->

Percentiles and z-scores are read against ORFs of the same length. We assessed calibration on the matched ORFs of the figure sample. The estimator bias is the rise of the i.i.d. mean curve from 100 aa. We compared it with the observed rise in all matched ORFs shorter than 1,000 aa. <!-- src: GR §4.2; figG02 and figGS02 captions -->

### Candidates and the matched comparator

Within annotated genomes, unmatched ORFs with 3Di entropy of at least 2.5 bits were split by overlap with deposited CDS coordinates on either strand, both placed on the forward genomic axis:
- a **shadow** overlaps a CDS;
- a **candidate** overlaps none. <!-- src: GR §6; scripts/10_missed_genes.py overlaps_annotated() -->

The 2.5-bit threshold was set by eye, well above the 1.585-bit ceiling (Results), and the candidate count depends on it. <!-- src: GR §10 --> ORFs running off a contig end, which have no stop codon, were flagged as contig-truncated. <!-- src: GR §6 -->

We classed shadows by frame relative to the overlapped CDS: same frame, same-strand frameshift, antisense, or frame undefined. Frame was read from the start on the plus strand and from the end on the minus strand. Frame is undefined when the CDS is partial. <!-- src: GR §6.1 "The comparator had to be fixed first" --> Same-frame shadows are largely the annotated protein, so the comparator was drawn only from antisense and frameshift shadows. That pool held 3,925,368 bacterial and 133,748 archaeal ORFs. <!-- src: MC item 1 -->

Each candidate received one partner without replacement (genome_entropy #109; seed 0). The partner had the same truncation status, 3Di entropy within ±0.025 bits and length within ±10%. <!-- src: MC item 2 --> Partners were sought first in the same genome, then the same 250-genome chunk, then the whole domain. Candidates left without a partner were retried with wider calipers and finally matched with replacement. In bacteria, 36.7% of partners came from the same genome and 0.16% of draws were with replacement; in archaea the figures were 26.5% and 4.3%. <!-- src: MC item 2 --> The total variation distance from the candidates on a 5 × 5 length × 3Di grid was 0.003 (bacteria) and 0.010 (archaea). <!-- src: MC "Why" table --> Comparator rates are weighted by draw multiplicity.

There were three further arms:
- annotated CDS, length-matched, as the positive control;
- `intergenic_lo`, low-3Di non-shadow ORFs, length-matched;
- the candidates' 3Di strings shuffled, as a technical null.

The bacterial control arms were capped at 200 per chunk, giving 151,927 ORFs each. <!-- src: GR §6.1 "Design"; figG04 caption -->

### Structural search and estimation

Query databases were built from amino-acid and predicted 3Di sequences. They were searched with Foldseek 10 (`10.941cd33`) [@vanKempen2024] against AlphaFold DB/Swiss-Prot [@Varadi2024], PDB100 [@Berman2000], CATH50 [@Sillitoe2021] and BFVD [@Kim2025bfvd]. <!-- src: MC item 3; scripts/17_pilot_search.pbs --> <!-- TODO integrator: database release/download dates not found in GR or MC. --> We ran a structural search (`--alignment-type 2`) and a sequence-only search (`base:search`), both at E ≤ 10⁻³, and kept the best hit by bit score. <!-- src: MC item 3; 17_pilot_search.pbs --> A hit was **full-length** when it covered at least 80% of both query and target. We did not use raw hit rate, because above 3.38 bits nearly every ORF hits. <!-- src: GR §6.1 -->

For each of Swiss-Prot, PDB100 and CATH50, the real-gene share is f = (c − s)/(a − s). Here c, s and a are the full-length hit rates of candidates, comparator and annotated CDS. <!-- src: MC "Result" --> The 95% confidence interval (CI) is first-order (delta method). It covers binomial noise only, not the two-component mixture assumption. <!-- src: scripts/41_comparator_summary.py share(); MC "Caveats" --> The implied number of missed genes is f times the number of candidates, reported as the range across the three databases. BFVD was excluded from estimation, because annotated CDS reach only about 11% there. <!-- src: GR §6.1; {bac,arc}_v7_matched.tsv --> We tested three variants of the comparator weighting:
- 5 × 5 standardisation;
- unweighted;
- excluding draws made with replacement. <!-- src: GR §6.1 -->

**Direct evidence (Table 3).** These criteria need no mixture model (genome_entropy #97): <!-- src: GR §6.1; scripts/22_rank_candidates.py tiers() -->
- **C1:** all candidates;
- **C2:** not truncated;
- **C3:** C2 and a full-length structural hit in any database;
- **C4:** C2 and full-length hits in at least two databases;
- **C5:** C2 and a full-length hit in Swiss-Prot, PDB100 or CATH50;
- **C6:** C2, a full-length structural hit and no full-length sequence-mode hit;
- **C7:** the strict rule, which is C4 and C5 together, plus an interpretable Swiss-Prot or PDB100 product name, at least 100 aa and best E < 10⁻¹⁰. <!-- src: GR §6.1; 22_rank_candidates.py -->

The excess is the number of candidates minus the weighted comparator count. It is a floor on the number of real genes.

**Prodigal coincidence.** GTDB's own Prodigal calls [@Hyatt2010] (618,638,921 genes) use neither GenBank nor structure. <!-- src: GR §6.1 "An independent test" --> <!-- TODO integrator: Prodigal version used by GTDB r232 not stated in GR/MC. --> An ORF coincides with a call when the two share contig, strand and 3′ end. The 3′ offset was calibrated rather than assumed; it peaked at 0 bp. <!-- src: GR §6.1; matched_comparator/prodigal_v7_{bac,arc}.txt --> The Prodigal share applies the same mixture to coincidence rates. Its background is the larger of the comparator and `intergenic_lo` rates; the comparator was the larger in both domains. <!-- src: GR §6.1; MC "Result" -->

**Assembly-quality strata.** We recomputed the estimate within quartiles of contig N50 and of CheckM2 completeness [@Chklovski2023] (GTDB r232 metadata; decision D15). <!-- src: AS "Method" --> An unstratified run reproduced all 51 published metrics. <!-- src: AS "Method" --> We used two ways of assigning comparator ORFs to strata:
- **(i) the primary analysis:** comparator draws follow their candidate's genome, so matched pairs stay together;
- **(ii) a sensitivity analysis:** every ORF follows its own genome. <!-- src: AS "Method"; decisions.md D15 -->
