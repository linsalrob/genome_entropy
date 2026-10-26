# Title

Structural entropy distinguishes protein-encoding genes from spurious open reading frames.

# Authors

<!-- Author list, affiliations and contributions are built from Author_Information.xlsx (see README.md). -->

# Abstract <!-- AGENT: overall this is quite good. Can you make it one/two paragraphs and not with dot points and bold text? And make the english flow better -->

Most open reading frames (ORFs) in a microbial genome do not encode proteins. Gene callers choose among them from nucleotide statistics, without asking whether the translated sequence resembles a protein.

We asked whether the information content of predicted protein structure separates real genes from spurious ORFs. With genome_entropy we computed, for every ORF, the Shannon entropy of the amino-acid sequence, the Foldseek 3Di structural alphabet, and the 12-state structure. The 3Di string is predicted from sequence alone by a protein language model. The datasets are:
- 2.62 billion ORFs from 199,837 GTDB species representatives;
- 90 billion six-frame ORFs from 50,009 assembled metagenomes;
- 6.75 million ORFs of at least 90 aa, and 102 million of 10–90 aa, from 36,025 phage genomes;
- 43.7 million small ORFs.

**Annotated genes and unannotated ORFs separate on 3Di entropy, not amino-acid entropy.** In bacteria the mean difference is 2.08 bits for 3Di against 0.39 bits for amino acids, and 87.6% of unannotated bacterial ORFs have 3Di entropy below log₂3, the most that a string of three states can reach. The structural signal:
- persists at fixed length;
- is specific to the translated frame, ranking a gene's own frame first among six in 92.4% of genes;
- carries most of the information about phage annotation (0.587 of 1 bit).

**Missed genes.** Against length-, entropy- and truncation-matched shadow ORFs, about 51,000–59,000 ORFs in GTDB genomes behave like protein-coding genes missing from GenBank. Under a strict individual test, which 25,114 candidates pass, the excess over the matched comparator gives a conservative floor of about 5,400; an independent gene caller gives an upper bracket of about 200,000. In metagenomes, 41.9% of high-scoring intergenic ORFs that Prodigal did not call have full-length structural homologues, against at most 0.45% of matched controls. The approach also recovers a phage replication gene lost in re-annotation.

**Limit.** Below about 100 amino acids the signal weakens, which marks the method's lower length limit.

# Introduction

Almost every analysis in microbial genomics, from metabolic reconstruction to comparative and evolutionary studies, starts from a list of protein-coding genes. Gene callers such as Prodigal [@Hyatt2010-qv], GeneMarkS [@Besemer2001-un] and Glimmer [@Delcher2007-uf] build that list from nucleotide statistics: codon usage, coding potential, start codons and ribosome-binding sites. They are accurate for complete bacterial genomes. They are weaker where those statistics are weakest: short genes, fragmented contigs, unusual codon usage, and metagenomes in which a contig may come from an organism unlike any the model was trained on.

None of these statistics asks directly whether the translated sequence looks like a protein. Most ORFs in a genome are not genes. In every six-frame translation, the ORFs of real genes are outnumbered by their own shadows in other frames on the same strand and on the opposite strand, and by intergenic ORFs. 

We can use information theory to compare predicted ORFs and protein-encoding ORFs. We previously used Shannon entropy to describe the composition of genomes and metagenomes [@Akhter2013-pl]. We reasoned that real proteins are constrained by folding and function, and this constraint should show less in amino-acid composition than in structure.

Protein language models now make structure cheap to approximate. Foldseek's 3Di alphabet describes each residue's local tertiary interactions in 20 states [@Van_Kempen2024-il]. Models such as ProstT5 [@Heinzinger2024-ja] and ModernProst [@modernprost] predict 3Di strings directly from amino-acid sequences, fast enough to encode every ORF of a genome, or of a metagenome.

Here we test whether the entropy of predicted structure separates protein-coding genes from spurious ORFs, and whether it finds genes that annotations missed. We compare structural evidence with matched controls using genome_entropy [@Edwards2026-jk] at four scales:
1. **GTDB r232:** the species representatives [@Parks2022-eo], to describe the entropy landscape of annotated and unannotated ORFs across bacteria and archaea, and to estimate how many genes GenBank annotation misses.
2. **LOGAN:** 50,009 assembled metagenomes [@Chikhi2025-ak], comprising 90 billion six-frame ORFs, to test the signal beyond ORF length, across reading frames and biomes, and against Prodigal's own calls.
3. **Inphared:** 36,025 phage genomes [@Cook2021-ay], a gene-dense case in which we rank candidate missing genes and examine an annotation error.
4. **GMSC small ORFs** [@Duan2024-ab], to find where the method stops working.



# Methods

## Overview of the genome_entropy framework

All entropy features were computed with genome_entropy (https://github.com/linsalrob/genome_entropy) [@Edwards2026-jk]. The pipeline takes nucleotide sequences (from either a GenBank or FASTA format file) and:
1. calls every open reading frame (ORF) in all six frames;
2. translates each ORF with the bacterial genetic code (NCBI table 11);
3. predicts a structural-alphabet representation of each translated ORF;
4. calculates per-ORF entropy features.

When an input record carries CDS annotations, each ORF is also labelled by whether it matches one of them. Different datasets were processed with different package releases, encoding settings and hardware. The configuration for each is given in Table 1, and its effect on the values is quantified under *Reproducibility of the structural encoding*.


**Table 1. Datasets and encoding configurations.**

| dataset | genomes / samples | ORFs analysed | Shortest ORF | labels | genome entropy version | encoding size | hardware |
|---|---|---:|---|---|---|---:|---|
| GTDB r232 bacteria | 189,715 species representatives (of 189,801) | 2,568,244,984 | 90 aa | deposited GenBank CDS | 0.2.0 (inferred; commit 90c8c15) | 10,000 | NVIDIA V100, NCI Gadi |
| GTDB r232 archaea | 10,122 species representatives | 54,858,398 | 90 aa | deposited GenBank CDS | 0.2.0 (inferred) | 10,000 | NVIDIA V100, NCI Gadi |
| LOGAN metagenomes | 50,009 assembled metagenomes, 13 biomes | 89,991,549,208 | 30 aa | Prodigal 2.6.3 `-p meta` | 0.3.0 (commit 750e2b92) | 10,000 | AMD MI250X, Pawsey Setonix |
| Inphared phage, mutual information | 36,025 genomes | 6,754,361 | 90 aa | Pharokka + phold CDS (corrected matcher) | ≈0.2.0 | 20,000 | AMD GPUs (ROCm, fp16), Setonix |
| Inphared phage, ORF ranking | 31,200 genomes | 5,887,690 | 90 aa | as above | ORFs from 0.1.15, relabelled | 20,000 | Setonix |
| Inphared phage, comparison with bacteria | 36,025 genomes | 6,754,361 | 90 aa | as above | 0.3.0 (commit 45dd283) | 10,000 | NVIDIA V100, Gadi |
| Inphared phage, short ORFs | 36,025 genomes (35,921 analysed) | 102,133,250 (10–90 aa) | 10 aa | as above | 0.4.0 (commit 3597911) | 10,000 | NVIDIA V100, Gadi |
| GMSC small ORFs | — | 43,726,909 (10–99 aa) | — | GMSC curated positive/negative | 0.3.0 | 50,000 | NVIDIA V100, Gadi |



## ORF calling

ORFs are maximal stop-to-stop open reading frames reported by `get_orfs` [@Edwards2024-cn], so an ORF extends upstream of any start codon to the preceding in-frame stop. Each ORF records whether it is complete or runs off a contig end.

The minimum-length argument behaves differently by source of data. The GTDB, phage and mutual-information datasets have an effective shortest sequence of 90 amino acids. The LOGAN analysis explicitly set the shortest sequence to 30 amino acids. The GMSC small ORFs are a curated set of ORFs from 10–99 amino-acids and were not ORF-called. <!-- src: methods_provenance.md §1; Validation/gtdb/scientific_report.md l.135–141; #103 --> After the unit error was fixed, the phage genomes were also re-encoded with ORFs of 10–90 amino acids (Methods, *Short phage ORFs*).

## Annotation labels

For datasets with deposited annotations (GTDB, phage), an ORF was labelled as matching a CDS when all of the following held:
- it lies on the same record and strand as the CDS;
- its biological start is in compatible codon phase with the CDS's, differing by a multiple of three, measured from the start on the plus strand and from the end on the minus strand;
- the two genomic intervals overlap by at least 90% of the shorter one;
- the coordinate-aligned, ungapped translations over the shared codons are at least 98% identical, with `X` treated as a wildcard.

We did not require a stop-codon because many ORFs, especially in metagenomics data, run off the end of the contigs. A short CDS that lies wholly inside a longer stop-to-stop ORF can therefore match it. <!-- src: src/genome_entropy/io/genbank.py evaluate_orf_genbank_cds_match; MIN_CDS_OVERLAP_FRACTION=0.90, MIN_SHARED_AA_IDENTITY=0.98 -->

LOGAN has no deposited annotation, so its ORFs were labelled against *de novo* Prodigal 2.6.3 [@Hyatt2010-qv] gene calls (`-p meta`, contigs ≥ 120 nt) instead. <!-- src: AH genome_entropy_logan/PROJECT.md:71, :87; AH manuscript/manuscript.md:30; Table 1 -->



## Structural encoding

Translated ORFs were encoded with ModernProst (`gbouras13/modernprost-50M`) [@modernprost]. This multitask protein language model predicts two per-residue structural alphabets directly from the amino-acid sequence, with no structure or alignment as input:
- the 20-state Foldseek 3Di alphabet [@Van_Kempen2024-il];
- a 12-state alphabet.

We used model revision `2d68d506d4ce8d59025c985fd00243a8860363c6` (Table 1). Sequences were encoded in batches of up to a fixed number of residues, the *encoding size*. Because ModernProst predicts structural states from sequence alone, it assigns confident 3Di states to any protein-like sequence.

## Entropy features

For each ORF, Shannon entropy was computed in bits from the observed symbol frequencies of the sequence in a given alphabet:

$$H(X) = -\sum_{i} p_i \log_2 p_i ,$$

where $p_i$ is the fraction of positions in the ORF that carry symbol $i$. This was done separately for the nucleotide sequence (4 symbols), the amino-acid sequence (20 standard residues; the GTDB analyses also counted `X` as a symbol), the 3Di string (20) and the 12-state string (12). The maxima are therefore $\log_2 4 = 2$, $\log_2 20 \approx 4.32$, $\log_2 20$ and $\log_2 12 \approx 3.58$ bits.

The per-ORF mutual information between the aligned 3Di and 12-state strings was computed from their joint frequencies:

$$I(\mathrm{3Di};\,\mathrm{12st}) = \sum_{a,b} p(a,b)\,\log_2 \frac{p(a,b)}{p(a)\,p(b)} .$$
<!-- src: src/genome_entropy/entropy/shannon.py shannon_entropy, mutual_information -->

Two properties of these estimators shape the analyses below.
- **Length bias.** The estimators are biased downwards by an amount that shrinks with sequence length. Part of any rise of entropy with ORF length is therefore estimation bias, not biology. We quantify this against an i.i.d. reference and an empirical length-conditioned reference (Fig. S1, S2), and control for length in every comparison.
- **The log₂k ceiling.** A string that uses only $k$ symbols has entropy of at most $\log_2 k$ bits. An ORF whose predicted 3Di uses two or three states therefore has 3Di entropy of at most 1 or $\log_2 3 \approx 1.585$ bits, whatever its length.

## Statistics

The statistics shared across datasets are:
- **Discrimination:** Receiver-Operator Characterists (ROC) area under the curve (AUC) and average precision (AP).
- **Proportions:** reported with Wilson or first-order (delta-method) 95% confidence intervals.
- **Correlations:** Spearman's ρ.

Dataset-specific estimands are described under each dataset:
- the mixture estimate of real-gene share and its comparator (GTDB);
- grouped cross-validation (LOGAN, phage);
- binned mutual information with permutation nulls (phage).

Analyses used Python 3.12 with NumPy, pandas, SciPy, scikit-learn and XGBoost. The exact versions for each dataset are recorded in its analysis directory.

## GTDB r232 reference genomes

### Genomes, ORFs and labels

We analysed the species representatives of GTDB release 232 (R11-RS232) [@Parks2022-eo; @Parks2026-rp]: 189,801 bacterial and 10,122 archaeal genomes. <!-- src: GR:3; GR §2 --> GenBank records were downloaded with NCBI `datasets` 18.36.0 [@O-Leary2024-nb]. Forty-seven bacterial assemblies were no longer served and 39 failed to encode (below). That left 189,715 bacterial genomes with 2,568,244,984 ORFs and 10,122 archaeal genomes with 54,858,398 ORFs. <!-- src: GR §2, §3 -->

Counted directly from the GenBank records, 92,840 bacterial (48.9%) and 4,630 archaeal (45.7%) genomes carry no deposited CDS. <!-- src: GR §2; GR §6 table --> We therefore excluded these genomes from the analysis, leaving 96,875 bacterial and 5,492 archaeal genomes. <!-- src: GR §4; figGS03 caption -->

`genome_entropy` enumerated ORFs stop to stop in all six frames with a 30-aa minimum, so every GTDB statistic is over ORFs of at least 90 aa. <!-- src: GR §3 (GR:135–141); MP §1 -->

ORFs were labelled *matched* or *unmatched* against the deposited CDS. The algorithm has no stop-codon check, so "unmatched" is not ground truth for non-coding. In archaeal genomes the six-frame caller recovers about 90% of deposited CDS (median per genome); recall was not measured for bacteria. <!-- src: MP §1; GR §6 "Sensitivity of the ORF caller"; GR §10 -->

### Encoding and entropies

ORFs were encoded with `genome_entropy` 0.2.0 (commit `90c8c15`, output schema 2.2.0). <!-- src: MP §2, §4 item 2 --> The encoder was ModernProst (`gbouras13/modernprost-50M`, revision 2d68d506) [@modernprost]. It predicts 20-state 3Di [@Van_Kempen2024-il] and a 12-state alphabet from sequence alone. <!-- src: MP §1; GR:4 --> Encoding ran on NVIDIA V100-SXM2-32GB GPUs (NCI Gadi), with an encoding size of 10,000 and four processes per GPU (Table 1). <!-- src: GR §3; MP §2 --> Thirty-nine genomes (0.021%) failed with out-of-memory errors on pathological ORF calls of roughly 48,000–766,000 aa. 

Entropies and MI are as defined under *Entropy features*, except that protein entropy counted `X` as a 21st symbol. <!-- src: figG01 caption "Definitions"; GR §4.2 ("The protein alphabet is 21 symbols") --> Population statistics were streamed over every ORF row. Counts and means are exact, while medians and quartiles come from histograms with 10⁻⁴-bit bins. <!-- src: GR §4.1 --> Fig. 1a–c and Fig. S2 use a systematic sample of every 300th bacterial and every 30th archaeal row (4,893,192 and 1,138,714 ORFs after exclusion). <!-- src: GR §4 "Figures" -->

**Length references.** To separate estimator bias from biology, we built two references: <!-- src: GR §4.2 -->
- **independent, and identically distributed (i.i.d.):** the entropy of 400,000 multinomial draws at each of 671 lengths, using each domain's pooled residue composition (351 genomes);
- **empirical:** the length × entropy distribution of every ORF, in 1-aa length bins up to 999 aa and coarser bins above. <!-- src: GR §4.2 -->

Percentiles and z-scores are read against ORFs of the same length. We assessed calibration on the matched ORFs of the figure sample. The estimator bias is the rise of the i.i.d. mean curve from 100 aa. We compared it with the observed rise in all matched ORFs shorter than 1,000 aa. <!-- src: GR §4.2; figG02 and figGS02 captions -->

### Candidates and the matched comparator

Within annotated genomes, unmatched ORFs with 3Di entropy of at least 2.5 bits were split by overlap with deposited CDS coordinates on either strand:
- a **shadow** overlaps a CDS;
- a **candidate** overlaps none. <!-- src: GR §6; scripts/10_missed_genes.py overlaps_annotated() -->

The 2.5-bit threshold was set by eye, well above the 1.585-bit ceiling (Results). <!-- src: GR §10 --> ORFs running off a contig end, which have no stop codon, were flagged as contig-truncated. <!-- src: GR §6 -->

We classed shadows by frame relative to the overlapped CDS: same frame, same-strand frameshift, antisense, or frame undefined. Frame was read from the start on the plus strand and from the end on the minus strand. Frame is undefined when the CDS is partial. <!-- src: GR §6.1 "The comparator had to be fixed first" --> Same-frame shadows are largely the annotated protein, so the comparator was drawn only from antisense and frameshift shadows. That pool held 3,925,368 bacterial and 133,748 archaeal ORFs. <!-- src: MC item 1 -->

Each candidate received one partner without replacement. The partner had the same truncation status, 3Di entropy within ±0.025 bits and length within ±10%. <!-- src: MC item 2 --> Partners were sought first in the same genome, then the same 250-genome chunk, then the whole domain. Candidates left without a partner were retried with wider calipers and finally matched with replacement. In bacteria, 36.7% of partners came from the same genome and 0.16% of draws were with replacement; in archaea the figures were 26.5% and 4.3%. <!-- src: MC item 2 --> The total variation distance from the candidates on a 5 × 5 length × 3Di grid was 0.003 (bacteria) and 0.010 (archaea). <!-- src: MC "Why" table --> Comparator rates are weighted by draw multiplicity.

There were three further arms:
- annotated CDS, length-matched, as the positive control;
- `intergenic_lo`, low-3Di non-shadow ORFs, length-matched;
- the candidates' 3Di strings shuffled, as a technical null.



### Structural search and estimation

Query databases were built from amino-acid and predicted 3Di sequences. They were searched with Foldseek 10 (`10.941cd33`) [@Van_Kempen2024-il] against AlphaFold DB/Swiss-Prot (v6, 2025_03) [@Varadi2024-ze], PDB100 (2025-01-01) [@Berman2000-je], CATH50 (v4.3.0) [@Sillitoe2021-lf] and BFVD (2023_02_v2) [@Kim2025-fs]. The phage searches used a later BFVD release (Methods, *Phage genomes*). <!-- src: MC item 3; scripts/17_pilot_search.pbs; Gadi review on #137 (.version files in gtdb_entropy/foldseek_db/; PDB_DATE 250101) --> We ran a structural search (`--alignment-type 2`) and a sequence-only search (`base:search`), both at E ≤ 10⁻³, and kept the best hit by bit score. <!-- src: MC item 3; 17_pilot_search.pbs --> A hit was **full-length** when it covered at least 80% of both query and target. We did not use raw hit rate: in the top candidate 3Di quintile (≥3.22 bits in bacteria, ≥3.32 in archaea), 88–99% of every protein-like arm hits a structure database, antisense and frameshift shadows included. <!-- src: SA §2 (l.97–130); SA §12 item 5 -->

For each of Swiss-Prot, PDB100 and CATH50, the real-gene share is f = (c − s)/(a − s). Here c, s and a are the full-length hit rates of candidates, comparator and annotated CDS. <!-- src: MC "Result" --> We computed the 95% confidence interval (CI) with the delta method, treating c, s and a as independent binomial proportions with variance p(1 − p)/n. For candidates and annotated CDS, n is the number of proteins searched. For the comparator, whose rate is weighted by draw multiplicity, n is the Kish effective sample size (Σw)²/Σw², where w is the number of times each protein was drawn. Propagating these variances through f to first order gives Var(f) = [Var(c) + (1 − f)² Var(s) + f² Var(a)] / (a − s)², and the CI is f ± 1.96 √Var(f). The implied number of missed genes is f times the number of candidates, reported as the range across the three databases. BFVD was excluded from estimation, because annotated CDS reach only about 11% there. <!-- src: GR §6.1; {bac,arc}_v7_matched.tsv --> We tested three variants of the comparator weighting:
- 5 × 5 standardisation;
- unweighted;
- excluding draws made with replacement. <!-- src: GR §6.1 -->

**Direct evidence (Table 3).** These criteria need no mixture model: <!-- src: GR §6.1; scripts/22_rank_candidates.py tiers() -->
- **C1:** all candidates;
- **C2:** not truncated;
- **C3:** C2 and a full-length structural hit in any database;
- **C4:** C2 and full-length hits in at least two databases;
- **C5:** C2 and a full-length hit in Swiss-Prot, PDB100 or CATH50;
- **C6:** C2, a full-length structural hit and no full-length sequence-mode hit;
- **C7:** the strict rule, which is C4 and C5 together, plus an interpretable Swiss-Prot or PDB100 product name, at least 100 aa and best E < 10⁻¹⁰. <!-- src: GR §6.1; 22_rank_candidates.py -->

The excess is the number of candidates minus the weighted comparator count. It is the minimum likely number of real genes.

**Prodigal coincidence.** GTDB's own Prodigal v2.6.3 calls [@Hyatt2010-qv] (618,638,921 genes) use neither GenBank nor structure. <!-- src: GR §6.1 "An independent test"; GTDB r232 METHODS.txt line 4 (release 232.0, 15 April 2026), per Gadi review on #137 --> An ORF coincides with a call when the two share contig, strand and 3′ end. The 3′ offset was calibrated rather than assumed; it peaked at 0 bp. <!-- src: GR §6.1; matched_comparator/prodigal_v7_{bac,arc}.txt --> The Prodigal share applies the same mixture to coincidence rates. Its background is the larger of the comparator and `intergenic_lo` rates; the comparator was the larger in both domains. <!-- src: GR §6.1; MC "Result" -->

**Assembly-quality strata.** We recomputed the estimate within quartiles of contig N50 and of CheckM2 completeness [@Chklovski2023-gv] (GTDB r232 metadata). <!-- src: AS "Method" --> An unstratified run reproduced all 54 published metrics. <!-- src: AS "Method" --> We used two ways of assigning comparator ORFs to strata:
- **(i) the primary analysis:** comparator draws follow their candidate's genome, so matched pairs stay together;
- **(ii) a sensitivity analysis:** every ORF follows its own genome. <!-- src: AS "Method"; decisions.md D15 -->

## LOGAN metagenomes

### Accession selection

We drew shotgun metagenomes from the assembled contigs of LOGAN v1.2 [@Chikhi2025-ak]. Of 38,072,291 runs in the LOGAN SRA metadata, we kept those with `librarysource = METAGENOMIC` (7,754,394), LOGAN v1.2 contigs available (7,653,092), `assay_type = WGS` (1,122,461), random-shotgun library selection (863,841), a short-read platform and an NCBI "… metagenome" organism name other than synthetic, mock or mixed-culture (761,663). <!-- src: LMS:125-133 (AH/manuscript/manuscript.md); 7,653,092 from the LOGAN agent's review on #137 (01_select_accessions output) --> Each run was assigned to one of 13 biomes from its organism name, with BioSample host and isolation source as fallback. <!-- src: LMS:138-141 --> We required a contig sum of 5–300 Mbp and N50 ≥150 bp (from LOGAN seqstats, preferring v1.1 values because v1.2 carries stale v1.0 statistics for pre-2024 assemblies), sampled stratified by biome and four log-spaced assembly-size bins, and capped each BioProject at 25 runs. <!-- src: LMS:141-148 --> A pilot and two tranches of 25,000 runs (seed 102) gave 50,091 runs; 81 had no contig object in LOGAN and one failed contig-count validation, leaving 50,009 metagenomes. <!-- src: LMS Table 1 (l.150-161); Validation/logan/README.md:34,59; AH/manuscript/tables/S10,S11 -->

### Contigs, ORFs and Prodigal labels

Contig files were validated against S3 and seqstats (tolerance 0.1%), and contigs ≥120 nt were kept, as LOGAN does before its own gene calling. <!-- src: LMS:188-192 --> ORFs were enumerated in all six frames with genome_entropy 0.3.0 (`find_orfs`, genetic code 11). <!-- src: methods_provenance.md §2 (LOGAN row) --> We kept ORFs 30 amino acids or longer from LOGAN contigs. <!-- src: Validation/logan/README.md:139-141; results_inventory.md §5 --> ORFs running off a contig's 3′ end were trimmed to the last complete codon.

Prodigal 2.6.3 [@Hyatt2010-qv] was run on the same contigs (`prodigal -q -p meta`, GFF output); its output was byte-identical on repeated runs of all 91 pilot accessions and every 100th later accession. <!-- src: LMS:193-196; LOGAN agent review on #137 --> Prodigal supplied labels only, not the ORFs. An ORF was  called *Prodigal+* when a Prodigal gene on the same contig and strand ended at the same stop and met our acceeptance criteria (same codon phase, ≥90% overlap of the shorter interval, ≥98% amino-acid identity, with Prodigal's initiator M aligned to the ORF's own residue). <!-- src: LMS:208-212; Validation/logan/README.md:124-127 --> Every other ORF was  called *Prodigal−* and was classed by its largest-overlapping Prodigal gene as an antisense shadow (opposite strand), frameshift shadow (same strand, other frame), frame-undefined overlap, marginal overlap (1–29 bp), same-frame unmatched (almost all genetic-code-4 models reading through TGA) or intergenic (no overlap); no ORF fell in the frame-undefined class. <!-- src: Validation/logan/README.md:128-130; LMS:213-220; manuscript/data/tabL01_orf_classes_by_truncation.tsv.gz --> A Prodigal+ ORF was *complete* when its gene was `partial=00` and the ORF ended in a stop; a Prodigal− ORF was complete when it ended in a stop and had an in-frame stop upstream.

### Encoding and features

Translations were encoded with ModernProst (`gbouras13/modernprost-50M`, revision 2d68d506) [@modernprost] at encoding size 10,000 with genome_entropy 0.3.0 (commit 750e2b92), as recorded in all 89,994 chunk manifests, on Pawsey Setonix AMD MI250X nodes (8 GCDs × 4 encoder workers). <!-- src: #24 LOGAN agent comment ("your three requests"); methods_provenance.md §1-2; LMS:223-225 --> Amino-acid, 3Di and 12-state strings were stored one row per ORF in zstd chunks of 1 million ORFs and re-read with per-column SHA-256 checks. <!-- src: LMS:226-229 --> Per-ORF values are not bit-comparable with the V100-encoded datasets (Table S1).

Features were length (aa), the Shannon entropy (bits) of the amino-acid, 3Di and 12-state strings, and 3Di–12-state mutual information (MI), computed to match `genome_entropy.entropy.shannon` (maximum difference 2.7 × 10⁻¹⁵). <!-- src: LMS:241-247 --> Per-shard tables matched the library values on 2.7 million ORFs to within 5 × 10⁻⁵ bits, with 100% class agreement. <!-- src: LMS:251-255 -->

### Statistics and models

Single-feature separation was computed exactly from histograms over all ORFs, stratified by class, truncation, length bin, biome and tranche (0.01-bit entropy bins). <!-- src: LMS:257-260 --> We report exact ROC AUC (ties counted ½) and average precision (AP), overall and within nine length bins, and a length-stratified AUC. <!-- src: LMS:260-262 -->

Multivariable models used a class-stratified sample of 11.7 million ORFs (9.5 million complete) from all 90 shards, weighted by the inverse of each class's sampling rate. <!-- src: LMS:264-265; AH/manuscript/tables/S4_models_overall.tsv (1,760,915 + 7,751,712 complete) --> `HistGradientBoostingClassifier` (scikit-learn [@Pedregosa2011-tz]; 300 iterations, learning rate 0.1, 63 leaves, early stopping) was fitted on feature sets nested on length, with five-fold cross-validation grouped by accession, trained unweighted and evaluated population-weighted on out-of-fold scores. <!-- src: Validation/logan/scripts/13_models.py:78-85; LMS:266-268 --> The complete-ORF model was also applied unchanged to contig-end fragments.

### Six-frame experiment

A deterministic 1% hash sample of complete Prodigal+ genes gave 8.82 million genes. <!-- src: LMS:271-272; Validation/logan/scripts/14_six_frame.py:19-24 --> For each gene we took the best-overlapping Prodigal− ORF in each of the five alternative frames (same strand +1, +2; opposite strand a0–a2). Inheritance was the Spearman correlation across genes between the selected and each alternative frame; discrimination was the tie-aware top-k rate against chance 1/(k+1), and Δ, the selected-frame value minus the median of the five alternatives. <!-- src: LMS:273-276 -->

### Candidate scan and Foldseek validation

The final complete-ORF model (length plus four features) scored all 49.95 million intergenic Prodigal− ORFs ≥100 aa and 4.58 million sampled controls. <!-- src: LMS:532-534 --> Thresholds are the out-of-fold scores at which population-weighted precision of Prodigal+ among complete ORFs reaches 0.80, 0.90, 0.95 and 0.99 (scores 0.633, 0.784, 0.874, 0.963; Table S4). <!-- src: manuscript/data/tabL04_final_model_thresholds.tsv.gz --> Because they were calibrated on complete ORFs of all lengths against all Prodigal− ORFs, they rank intergenic ORFs without giving their precision, and 5′-open ORFs are admitted at the same score without calibration. <!-- src: LMS:535-539; #24 Setonix integrator comment on 21_candidates_select.py -->

We sampled 20,000 complete or 5′-open candidates (score ≥ the 0.90 threshold), stratified by length, and four arms of 20,000 matched to them by nearest length (within truncation state for the three negative arms): low-scoring intergenic ORFs (below the 0.80 threshold), antisense shadows, frameshift shadows and complete Prodigal genes (positive control). <!-- src: Validation/logan/scripts/21_candidates_select.py:12-33; LMS:556-561, 592-596 --> A null arm held the candidates with each 3Di string shuffled and amino acids unchanged. <!-- src: Validation/logan/scripts/23a_build_queries.slurm:16,46 --> All arms were searched with Foldseek 10 [@Van_Kempen2024-il] in structure (3Di+AA, `--alignment-type 2`) and sequence (`base:search`) modes, E ≤ 10⁻³, against PDB100, AFDB/Swiss-Prot, CATH50, BFVD and AFDB50 (v6, 2025_03; 66.7 million entries) [@Berman2000-je; @Varadi2024-ze; @Sillitoe2021-lf; @Kim2025-fs]. <!-- src: LMS:561-566; Validation/logan/scripts/23_foldseek_search.slurm:31,48-52 --> The primary readout was a hit with full-length mutual coverage (query and target coverage ≥0.8), because shadows with structured 3Di also give partial hits. <!-- src: LMS:564-566 --> Population estimates weight each length bin by its share of candidates.

### Contig taxonomy and Prodigal diagnostics

The 79,774 contigs carrying the 80,000 queries of the candidate, low-score, antisense and Prodigal arms were re-streamed from LOGAN (24,824 accessions) and classified with Kraken2 2.17.2 [@Wood2019-zn] against PlusPFP (2023-06-05; default confidence). <!-- src: LMS:657-660; Validation/logan/scripts/28_kraken_contigs.slurm:12-22 --> We reran Prodigal 2.6.3 `-p meta -s` on these contigs, which reproduced the original call for every Prodigal-arm gene, and took the best-scoring potential gene ending at each query's stop (Table S8). <!-- src: LMS:680-688 --> For the reverse discordance, from the scan's 1-in-1,000 sample of complete Prodigal genes (880,405) we searched 10,269 genes below the 0.80 threshold (up to 1,500 per length bin) and 8,769 length-matched genes at or above the 0.90 threshold against the same databases. <!-- src: LMS:725-740; notebooks/figL10_prodigal_lowscore.ipynb caption -->

### Compute

Encoding 5.28 trillion amino acids gave a 5.95 TB compressed cache at 215–220 thousand amino acids s⁻¹ per GCD. <!-- src: LMS:783-786 --> The run was billed 1.06 million GPU SU (8,260 GCD-hours) and 264,000 CPU SU. Encoding the retained chunks took 6,724 GCD-hours (861,000 SU), or 176,000 SU per Tbp of assembled sequence. Time other than encoding made up 6.8% of billed GPU time in the first tranche and 26.8% in the second (7.2% and 36.6% more than the encoding time). The second tranche's share includes repeat encoding after an inode-quota failure, about 134,000 SU after allowing for the first tranche's overhead rate. The first tranche's billed rate, 191,000 SU per Tbp, is the best estimate for a further run. <!-- src: AH/manuscript/tables/S10_phaseb1_measured.json, S11_phaseb2_measured.json (gcd_hours_billed, gcd_hours_encoding_only, su_gpu_encoding, contig_Tbp); LOGAN agent review on #137; agent-handoffs@9886e67 -->

## Phage genomes

### Genomes, annotation and ORFs

We used 36,025 complete phage genomes from Inphared [@Cook2021-ay] (2.316 Gb in total; 1,761–735,411 bp), <!-- src: AH inphared_gadi/README.md §1 table; Validation/mutual_information/README.md §2 table --> realeased on 7 April 2026. <!-- src: Validation/mutual_information/README.md §2 table ("Phage release", UNCONFIRMED) --> Each genome had been annotated with Pharokka [@Bouras2023-wx] and then phold [@Bouras2026-yi]. Gene calls in these annotations came from Pyrodigal-gv 0.3.2 [@Larralde2022-xa; @Camargo2024-ox], which gave 3,336,744 CDS. <!-- src: Validation/mutual_information/README.md §2 table ("Gene calls in the reference"); methods_provenance.md §4 item 1 --> <!-- Gadi counts 3,340,852 CDS, all Pyrodigal-gv, in its copy of the same archive (AH inphared_gadi/short_orfs/README.md (90f4fc6), 53_cds_lengths.py); the 4,108 difference is unreconciled (agent-handoffs#27). --> The re-annotated CDS were the labels for every phage analysis. The submitters' own annotations were used only for the OM869708 example.

genome_entropy found six-frame stop-to-stop ORFs longer than 90 aa. <!-- src: Validation/mutual_information/README.md §2 table ("ORF calling"); results_inventory.md §5 --> That gave 6,754,361 ORFs, of which 18,322 run off a contig end. <!-- src: Validation/mutual_information/README.md §2 table ("ORFs") --> 957,269 of the 3,336,744 annotated CDS (28.7%) are shorter than 90 aa and cannot appear as an ORF of their own. <!-- src: Validation/mutual_information/README.md §2.1 ("Unrepresentable CDS"); IR:75 gives 28.8% for the 31,200-genome ranking subset -->

### Gene labels

An ORF was labelled a gene when it matched an annotated CDS under all of the following criteria:

- the CDS is on the same record and strand;
- the codon phase is compatible;
- the genomic overlap covers at least 90% of the shorter interval;
- at least 98% of the coordinate-aligned amino acids are identical, with `X` as a wildcard.

<!-- src: methods_provenance.md §1 (`in_genbank` label); src/genome_entropy/io/genbank.py -->

There is no stop-codon check, so a short CDS inside a longer stop-to-stop ORF labels that ORF. In 77,799 positives the CDS covers less than half of the ORF. <!-- src: Validation/mutual_information/README.md §2.1 ("Short CDS inside long ORFs") -->



### Two ORF sets

Two Setonix encodings of the phage ORFs were used.

- **Mutual-information set.** All 36,025 genomes and 6,754,361 ORFs were encoded with `gbouras13/modernprost-50M` [@modernprost] at an encoding size of 20,000 residues, on AMD GPUs (ROCm, fp16). <!-- src: Validation/mutual_information/README.md §2 table ("Re-encoding") --> The package version is genome_entropy 0.2.0, inferred from the JSON schema version, and the model revision was not recorded. <!-- src: Validation/mutual_information/README.md §2 table ("Package version"); methods_provenance.md §1 --> The corrected labels give 2,685,382 genes (39.8%). <!-- src: Validation/mutual_information/README.md TL;DR; IG:14 (39.76%) -->
- **Ranking set.** The ranking used an earlier encoding (ORFs from genome_entropy 0.1.15; same model and encoding size). <!-- src: Validation/Inphared/README.md §2 table (lines 74, 76) --> 49 of the 361 annotation parts had no encoder output, every fifth part from 121 to 361, which suggests a failed worker rather than biology (not tested). <!-- src: Validation/Inphared/README.md:71 --> The set therefore covers 31,200 genomes and 5,887,690 ORFs. <!-- src: Validation/Inphared/README.md:72, :74 --> With corrected labels it has 2,338,039 genes (39.7%). <!-- src: Validation/Inphared/README.md:102 --> The two independent relabellings agree on 5,887,451 of these 5,887,690 ORFs. <!-- src: Validation/Inphared/README.md:104 -->

A third encoding of all 36,025 genomes on Gadi was used only for the comparison with cellular genomes and for the determinism test, and a fourth, of ORFs of 10–90 aa, only for the short-ORF analysis (see below).

### Mutual information

For ORF length, the four entropies and the 3Di–12-state mutual information, we estimated binned plug-in mutual information with the label, with 20 empirical-quantile bins per axis and the 2-D product grid for pairs. <!-- src: Validation/mutual_information/README.md §3 item 4 --> Estimates were made on a class-balanced sample of 200,000 ORFs per class, where H(label) = 1 bit, and on the full population. <!-- src: Validation/mutual_information/README.md §3 item 4 -->

- **Uncertainty:** 100 bootstrap resamples, giving basic 95% intervals. <!-- src: Validation/mutual_information/README.md §3 item 4 -->
- **Null:** 200 label permutations, at 10, 20 and 40 bins. <!-- src: Validation/mutual_information/README.md §3 item 4 -->
- **Length control:** I(X; label | L), with L in 20 length-quantile strata and X re-binned within each stratum, plus single-feature ROC AUC within length bins. <!-- src: Validation/mutual_information/README.md §3 item 4 -->

The random seed was 20261014 throughout. <!-- src: Validation/mutual_information/README.md §3 item 4 -->

### Ranking candidate genes

Coding probabilities came from an XGBoost random forest (`XGBRFClassifier`; 500 trees, depth 8, row and per-node column subsampling 0.8) [@Chen2016-cv]. <!-- src: Validation/Inphared/README.md:78; scripts/rank_missing_orfs.py make_model --> We used out-of-fold predictions from five-fold cross-validation grouped by genome. <!-- src: Validation/Inphared/README.md:78 --> The features were intrinsic to each ORF: its four entropies and the lengths of the corresponding sequences, eight in all. <!-- src: Validation/Inphared/README.md:17, Table 1 line 130 --> Position, strand, frame and codon flags were excluded (Fig. S15b; Results).

Candidate missed genes were unannotated ORFs with p ≥ 0.8. <!-- src: Validation/Inphared/README.md:210; scripts/rank_missing_orfs.py defaults --> The forest's probabilities span about 0.11–0.89, so this threshold lies near the top of the scale and is not a calibrated 80%. <!-- src: Validation/Inphared/README.md:140 --> Replacement candidates were unannotated ORFs with p ≥ 0.8 that overlap an annotated ORF with p ≤ 0.5, by a margin of at least 0.4. <!-- src: scripts/rank_missing_orfs.py defaults (--minimum-probability 0.80, --maximum-annotated-probability 0.50, --minimum-probability-difference 0.40) --> The competitor was the overlapping annotated ORF with the greatest overlap. Pairs were ranked by 0.50 p + 0.35 Δp + 0.15 × the overlap fraction of the shorter ORF. <!-- src: scripts/rank_missing_orfs.py replacement_rank_score -->

### Structural validation

We searched 20,000 ORFs per arm with Foldseek 10 (10.941cd33) [@Van_Kempen2024-il]: <!-- src: Validation/Inphared/README.md §5.1 -->

- candidates: a random sample of the intergenic, antisense and other-frame candidates;
- annotated genes, as the positive control;
- low-scoring (p < 0.5) antisense, other-frame and intergenic ORFs, as negative controls;
- the candidate arm with each 3Di string shuffled, as a null.

Controls were matched to the candidate arm on length without replacement. <!-- src: Validation/Inphared/README.md:226 --> The replacement pairs were searched in full.

Queries combined the amino-acid sequence with the predicted 3Di string. We ran structure search with `--alignment-type 2 -e 1e-3` and sequence-only search at the same E-value. <!-- src: Validation/Inphared/README.md:240–243 --> The databases were PDB100 (2025-01-01) [@Berman2000-je], AFDB-SwissProt (v6, 2025_03) [@Varadi2024-ze], CATH50 (v4.3.0) [@Sillitoe2021-lf] and the viral BFVD (2025_03_v3) [@Kim2025-fs]. <!-- src: Validation/Inphared/README.md:247–252 --> A hit was full-length when query and target coverage were both ≥0.8. <!-- src: Validation/Inphared/README.md:254 --> We give Wilson 95% intervals. The share of candidates that behave like annotated genes was estimated as (candidate − control) / (annotated − control). <!-- src: Validation/Inphared/README.md:254; scripts/07_evidence.py header -->

### Comparison with cellular genomes

For Fig. S14 we used the Gadi encoding of all 36,025 genomes: genome_entropy 0.3.0, encoding size 10,000, NVIDIA V100. <!-- src: AH inphared_gadi/README.md §1 table --> Its labels reproduce the corrected labels exactly (identical label cross-tabulation). <!-- src: AH inphared_gadi/README.md TL;DR, §4 --> Bacterial and archaeal ORFs came from GTDB r232 genomes that carry annotation, encoded with genome_entropy 0.2.0 at the same encoding size and summarised by length (#101). <!-- src: notebooks/figP04_phage_vs_cellular.ipynb provenance cell --> Mean 3Di entropy of matched and unmatched ORFs was compared by length from 90 to 999 aa.

### Short phage ORFs

To extend the phage analysis below the 90-aa floor, we encoded all 36,025 genomes again with ORFs of 10–90 aa (`--min-aa 10 --max-aa 90`), using genome_entropy 0.4.0 (commit 3597911), which fixes the unit error (#103). The model and its revision, the encoding size (10,000), the hardware (NVIDIA V100, Gadi), the genome batches and the labels were those of the Gadi encoding above. Protein length excludes the stop codon, so ORFs of exactly 90 aa are in both runs. <!-- src: AH inphared_gadi/short_orfs/README.md (90f4fc6) "What changed"; 51_short_orfs.pbs (PBS 181639266) -->

ORFs are called with genetic code 11. Pyrodigal-gv called the genes of 104 genomes with genetic code 15 (61 genomes) or 4 (43 genomes), which read a TAG or TGA codon as an amino acid. In these genomes a reassigned stop splits one CDS into several shorter ORFs that each match it. We therefore excluded these genomes from the short-ORF analysis, leaving 35,921. <!-- src: AH inphared_gadi/short_orfs/README.md (90f4fc6) check 2; tables/all_genomes/transl_table_by_genome.tsv.gz, recoded_genomes.txt -->

Within each length bin, single-feature ROC AUC compares matched with unmatched ORFs. It is the Mann–Whitney statistic with average ranks for ties. Applied to the GMSC smORFs, the same code reproduces the values in Fig. S16d exactly. <!-- src: AH inphared_gadi/short_orfs/README.md (90f4fc6); 52_short_orf_analysis.py auc(); tables/code11/gmsc_auc_by_bin.tsv -->

### OM869708

We took the submitters' GenBank record of OM869708 from NCBI (retrieved 2026-10-16). <!-- src: notebooks/figP01_phage.ipynb provenance, upstream (d) --> We used BLASTP 2.12.0+ [@Camacho2009-uu] to search the stop-to-stop translations of ORF3 and ORF7 against a local copy of NCBI nr (snapshot of 1 August 2026), with E ≤ 10⁻⁵ and up to 50 target sequences. <!-- src: work/om869708/blastp_nr.slurm (Setonix job 50987603); methods_provenance.md §3 Phage -->

## Small ORFs from the Global Microbial smORF Catalogue

To find the method's lower length limit, we used the curated benchmark sets of the Global Microbial smORF Catalogue (GMSC) [@Duan2024-ab], as supplied by its authors.

- **Positives:** 33,647,794 high-quality smORFs supported by RNAcode coding potential [@Washietl2011-yi] together with metatranscriptomic or metaproteomic evidence, or by homology to known small proteins. <!-- src: Validation/smorfs/README.md:4, :23; AH smorfs/manuscript.md §2.1 -->
- **Negatives:** 10,079,115 conserved, in-frame GMSC sequences from clusters that RNAcode called non-coding. Sequences with conserved-domain hits or homology to the positives were removed. <!-- src: Validation/smorfs/README.md:4, :23; AH smorfs/manuscript.md §2.1 -->

The negatives are therefore not random ORFs. They are assembled, conserved, in-frame sequences that failed tests for coding, and they look protein-like by construction.

Proteins are 10–99 aa long: medians 80 aa (positives) and 70 aa (negatives). No 90-aa floor applies (Table S13). <!-- src: Validation/smorfs/README.md:24–28; data/tabS04_dataset_summary.tsv.gz -->

ORFs were translated with genetic code 11, the terminal stop was removed, and internal stops were kept. 1.71% of positives and 1.58% of negatives contain an internal stop. <!-- src: AH smorfs/manuscript.md §2.2 ("Translation") -->

Encoding used genome_entropy 0.3.0 and `gbouras13/modernprost-50M` (revision 2d68d506) at an encoding size of 50,000 residues, on NVIDIA V100 GPUs (Gadi). <!-- src: Validation/smorfs/README.md:34–40; notebooks/figS01_smorf_separation.ipynb provenance --> We computed the nucleotide, amino-acid, 3Di and 12-state entropies of each sequence.

We measured separation by ROC AUC, over all sequences and within seven length bins (10–29, 30–49, then 10-aa bins to 99 aa; Table S10). <!-- src: AH smorfs/manuscript.md §2.3; data/tabS01_length_bins.tsv.gz --> Gradient-boosted classifiers (scikit-learn `HistGradientBoostingClassifier`, early stopping, seed 42) [@Pedregosa2011-tz] were trained on nine feature sets. <!-- src: AH smorfs/manuscript.md §2.3 ("Models") --> Each was trained on 2 million sequences and tested on a disjoint 2 million. <!-- src: Validation/smorfs/README.md:83 -->

To control for homology, all 43,726,909 proteins were clustered together with MMseqs2 14.7e284 [@Steinegger2017-qw; @Steinegger2018-pg], at 50% identity (`linclust`) and at 30% identity (`cluster`), both at 80% coverage (Table S11). <!-- src: AH smorfs/manuscript.md §2.3 ("Homology clusters"); data/tabS02_cluster_stats.tsv.gz --> We repeated the models with train and test sets drawn from disjoint halves of the clusters. We also report a cluster-weighted AUC in which each test sequence has weight 1/(size of its 50%-identity cluster).

To explain sharp stripes in the 3Di-entropy distribution, we took the 3Di strings within ±0.0055 bits of log₂k (k = 2, 3, 7). <!-- src: notebooks/figSS02_smorf_stripes.ipynb provenance ("Bands") --> We compared them with control bands of the same width 0.03 bits below and above. <!-- src: notebooks/figSS02_smorf_stripes.ipynb provenance ("Bands") -->

## Reproducibility of the structural encoding

The datasets in this paper were encoded with different package versions, encoding sizes and hardware (Table 1). To separate run-to-run variation from the effects of configuration, we re-encoded the first 50 genomes of the Gadi phage encoding (6,284 ORFs). <!-- src: AH inphared_gadi/README.md §5; data/tabG01_encoding_determinism.tsv.gz (n_orfs) --> We used genome_entropy 0.3.0 and `gbouras13/modernprost-50M` on one NVIDIA V100, at encoding sizes of 10,000 and 20,000 residues. <!-- src: AH inphared_gadi/README.md §5; inphared_gadi/scripts/50_determinism.pbs (ngpus=1) -->

We compared these re-encodings with two earlier encodings (Table S1):

- the production run they repeat (V100, encoding size 10,000);
- an independent encoding of the same ORFs on AMD GPUs on Pawsey Setonix (ROCm, fp16, genome_entropy 0.2.0, encoding size 20,000). <!-- src: data/tabG01_encoding_determinism.tsv.gz (run_a, run_b) -->

The comparisons gave three results.

- **Repeat run.** Repeating the production configuration reproduced every 3Di entropy, 12-state entropy and 3Di–12-state mutual information value bit for bit. <!-- src: data/tabG01_encoding_determinism.tsv.gz rows "det es10k vs production" (100%) -->
- **Encoding size.** Changing the encoding size changed 0.3–0.7% of ORFs, by at most 0.076 bits. <!-- src: data/tabG01_encoding_determinism.tsv.gz rows "det es20k vs production" (99.68/99.60/99.35% identical; max 0.0758) -->
- **Hardware, precision and package version.** Between the two systems only 37.8% of 3Di entropies were bit-identical. <!-- src: data/tabG01_encoding_determinism.tsv.gz row "det es10k vs Setonix", three_di_entropy --> The median absolute difference was 0.007 bits, the 99th percentile 0.31 bits and the maximum 0.92 bits (Pearson r = 0.997). <!-- src: data/tabG01_encoding_determinism.tsv.gz (0.0066, 0.310, 0.920, 0.99719) --> Matching the encoding size left this spread unchanged. <!-- src: data/tabG01_encoding_determinism.tsv.gz row "det es20k vs Setonix" (37.7%, 0.310, 0.920) -->

We could not re-run the Setonix side in the same configuration. The design therefore cannot apportion the residual between hardware, numerical precision and package version. The 50 genomes reproduce the whole-population agreement between the two systems (99th percentile 0.310 against 0.317 bits; r = 0.99719 against 0.99698). <!-- src: AH inphared_gadi/README.md §5 point 3 -->

Population-level quantities are unaffected: class means move by about 0.01 bits, against a 1.46-bit difference between classes. <!-- src: AH inphared_gadi/README.md §5 ("Consequences") --> Per-ORF entropies are not reproducible bit for bit across platforms. In the 50-genome test, 99% of ORFs agree to within 0.31 bits (3Di) and 0.35 bits (12-state), and single ORFs differ by up to 0.92 and 1.05 bits; across the whole phage population the corresponding values are 0.32 and 0.35 bits, and up to 2.0 and 2.3 bits. <!-- src: AH inphared_gadi/README.md l.201–202 (p99 0.317 / 0.354; max 2.016 / 2.284) --> <!-- src: data/tabG01_encoding_determinism.tsv.gz (p99 0.310 / 0.348; max 0.920 / 1.050) --> An ORF near a decision threshold can therefore change class between platforms. We report per-ORF thresholds and candidate rankings with the hardware, numerical precision, encoding size and software version that produced them.

## Figures and data availability

Every figure and supplementary figure is drawn by a Jupyter notebook in `manuscript/notebooks/`, from compact plotting data committed in `manuscript/data/`. `manuscript/figures_manifest.tsv` maps every panel to:
- its notebook;
- its plotting data;
- its upstream source;
- its producing code.

Running `manuscript/tools/run_notebooks.py` in the pinned environment (`manuscript/environment.yml`, `environment.lock.txt`) regenerates every committed image byte for byte. Datasets too large for the repository are listed, with checksums, in `manuscript/roads_deposition_candidates.md` and `manuscript/roads/`.

# Results

## Real proteins and spurious ORFs occupy different structural-entropy regimes <!-- AGENT: as an overall comment about the whole Results section, i think the subsections and flow need to be more punchy, there are a few subsections that aren't particularly important -->

We called every six-frame ORF of at least 90 aa <!-- AGENT: why 90 aa as a threshold? Why not 50 or 100 or 75? --> in 199,837 GTDB r232 species representatives: 2,568,244,984 bacterial and 54,858,398 archaeal ORFs. Of these, 12.03% and 17.26% match <!-- AGENT: what does match mean - 100% sequence identity match? --> a deposited CDS. <!-- src: GR §1, §2 -->

Overall, we found that in genomes that carry any annotation, matched and unmatched ORFs separate on 3Di entropy, not protein entropy (Fig. 1a–c).
- **3Di entropy:** the mean differs by 2.08 bits in bacteria (3.123 against 1.043) and by 1.99 bits <!-- AGENT: put the specific numbers in brackets like you did for bacteria --> in archaea.
- **Protein entropy:** the means differ by only 0.39 and 0.37 bits.
- **Mutual information:** 3Di–12-state MI differs threefold (0.960 against 0.321 bits in bacteria). <!-- src: GR §4.1 item 1; figG01 caption -->
- **DNA entropy:** it does not separate the classes (1.930 against 1.909 bits). <!-- src: GR §4.1 item 3 -->

![**Figure 1.** Unannotated six-frame ORFs separate from annotated genes in 3Di entropy, not protein entropy, and most of them sit under the log₂3 ceiling (GTDB r232). **(a)** Protein against 3Di entropy for ORFs in GTDB r232 genomes that carry at least one deposited CDS, split by whether the ORF matches a GenBank CDS. Each panel is normalised within its class; the colour scale is logarithmic. The data are a systematic sample of every 300th bacterial and every 30th archaeal ORF (4.89 M and 1.14 M ORFs). **(b, c)** 3Di and protein entropy densities by class and domain. 3Di bins are 0.005 bits wide and closed on the right, so the edge at log₂3 = 1.585 bits is unsmoothed. Dotted lines mark log₂k for k = 1–4, the most a string that uses k distinct states can score. **(d)** The share of ORFs below log₂3, over the full population with no sampling: 87.6% of unmatched against 3.3% of matched bacterial ORFs, and 84.1% against 3.7% in archaea.](images/figG01_entropy_landscape.png){#fig:1 width=100%}


Matched ORFs are also about twice as long (376 against 178 aa), so part of the 3Di gap is explained by length differences (next section). <!-- src: GR §4.1 item 2 -->

Unmatched ORFs pile up against a sharp edge at 1.585 bits (Fig. 1b). This is $\log_2 (3)$, the maximum entropy of a string that uses only three distinct states. It is an arithmetic ceiling, not a biological threshold, and it falls at the same place in both domains. <!-- src: GR §5; figG01 caption --> Over the full population, ORFs below $\log_2 (3)$ are:
- **bacteria:** 87.6% of unmatched against 3.3% of matched ORFs;
- **archaea:** 84.1% against 3.7% (Fig. 1d). <!-- src: GR §4; figG01_population_bands.tsv.gz -->

In three genomes, states D, V and P make up 99.4% of residues below the line. <!-- src: GR §5; RI G-5 --> <!-- AGENT: consider and cite the original ProstT5 paper, this degeneracy of 3Di states is known -->

**Never-annotated genomes.** 48.9% of bacterial and 45.7% of archaeal representatives carry no deposited CDS. <!-- AGENT: deposited where? please check this, this seems very high --> <!-- src: GR §2, §6 --> Their ORFs are not low-3Di noise: 19.1% (bacteria) and 28.3% (archaea) reach 3Di ≥ 2.5 bits. Unmatched ORFs in annotated genomes reach that level only 0.83% and 1.25% of the time, a 23-fold difference in both domains. <!-- src: GR §4; GR §6 "The entropy-matched intergenic arm" --> These genomes therefore appear to hold many gene-like, unlabelled ORFs that are excluded them from all comparisons (Methods).

## Entropy rises with length mostly as estimator bias <!-- AGENT: what is estimator bias in this context? it is not immediately clear to the reader. Use less jargon. I think it is clearer to say that entropy rises with length, and that <100AA entropy is less useful - e.g. cite https://academic.oup.com/bioinformaticsadvances/article/2/1/vbab043/6501461  --> for protein, little for 3Di

Between 100 and 300 aa, mean protein entropy of matched ORFs rises by 0.08 bits (bacteria) and 0.09 bits (archaea). That is less than the 0.105 bits that estimator bias alone produces in each domain (Fig. S1). <!-- AGENT: what does estimator bias mean? --> <!-- src: figG02 caption; figG02_mean_curves.tsv.gz --> The apparent length trend in protein entropy is therefore an artefact.

3Di entropy rises by 0.88 and 0.76 bits over the same range. Only about 0.105 bits of that (12% and 14%) is bias, so longer proteins genuinely use more predicted structural states than shorter proteins. <!-- src: figG02 caption; figG02_mean_curves.tsv.gz -->

Only a reference built from real ORFs of the same length is calibrated: its z-scores have mean ≈ 0 and SD 1.00. Against i.i.d. draws, the SD of z is 3.9–6.6 (Fig. S2). <!-- src: figGS02 caption; figGS02_summary.tsv.gz --> <!-- AGENT: this sentence can be deleted i think, doesn't really flow --> <!-- AGENT: this whole subsection doesn't need such prominence, it can be relegated to a sentence or two I reckon -->

## The six-frame ORF space of 50,000 metagenomes reveals ... <!-- I think the subheading can be improves - fill in the ... --> <!-- AGENT: this subsection is a bit confusing, needs more explanation of the different classes. I suggest a diagram (ARTEMIS Style) as a main or supp figure -->

The 50,009 LOGAN metagenomes <!-- AGENT: randomly? --> chosen spanned 13 biomes and 4.895 Tbp of assembled contigs, and contain 89,991,549,208 six-frame ORFs ≥30 aa (5.28 trillion amino acids). <!-- src: LMS Table 1 (l.150-161), LMS:282-288; methods_provenance.md §2 --> Most of this space consists of alternative translations of genes <!-- AGENT: what are genes in this context? --> (Table 2). Antisense shadows of Prodigal-called CDS make up 47.55% of ORFs and same-strand frameshift shadows 26.97%, against 8.71% intergenic ORFs. <!-- src: manuscript/data/tabL01_orf_classes_by_truncation.tsv.gz; LMS:306-316 --> Only 0.98% of ORFs (881,348,601) are complete Prodigal genes, and a further 14.54% match partial genes. <!-- src: tabL01 --> We found pervasive fragmentation: about 59% of ORFs are open at a contig end (18.1% at the 5′ end, 41.2% at the 3′ end). <!-- src: tabL01 (column sums: open5 16.28 B, open3 37.08 B of 89.99 B) -->

**Table 2. Classes of the 89.99 billion six-frame ORFs ≥30 aa in 50,009 LOGAN metagenomes, by truncation state.** Counts in billions for the last three columns. Prodigal+ ORFs match a Prodigal 2.6.3 (`-p meta`) gene; Prodigal− ORFs are classed by their largest-overlapping Prodigal gene.  <!-- src: manuscript/data/tabL01_orf_classes_by_truncation.tsv.gz -->

| class | share of ORFs | complete | open 5′ | open 3′ |
|---|---:|---:|---:|---:|
| Prodigal+, complete gene | 0.98% | 0.881 | — | — |
| Prodigal+, partial gene | 14.54% | 0.008 | 1.97 | 11.12 |
| antisense shadow | 47.55% | 19.31 | 8.08 | 15.40 |
| frameshift shadow | 26.97% | 11.82 | 4.27 | 8.19 |
| intergenic | 8.71% | 3.90 | 1.74 | 2.20 |
| marginal overlap (<30 bp) | 1.18% | 0.687 | 0.192 | 0.184 |
| same frame, unmatched | 0.07% | 0.031 | 0.029 | — |
| all ORFs | 100% | 36.64 | 16.28 | 37.08 |

## Structural entropy separates genes <!-- AGENT: from what? --> beyond length

One of the simplest indicators of a protein-coding gene is the presence of a long open reading frame (ORF) without an in-frame stop codon <!-- AGENT: citation? -->. Under bacterial translation table 11, three out of the 64 codons code for stop codons, giving an expected frequency of approximately one stop codon every 21 codons in a random sequence with uniform nucleotide composition. Although nucleotide composition and codon biases alter this expectation, nonetheless, long ORFs are consequently unlikely to arise by chance. Therefore, ORF length can distinguish complete Prodigal-predicted genes from complete Prodigal-negative ORFs with high discrimination (AUC = 0.964), but substantially lower average precision (AP = 0.580; Table S2). <!-- src: manuscript/data/tabL02_single_feature_overall.tsv.gz (aa_len, vs all Prodigal−) --> However, using 3Di entropy alone reaches AP 0.783 over all 36.6 billion complete ORFs. <!-- src: tabL02 (three_di_entropy, vs all Prodigal−) --> In cross-validated models nested on length, combining 3Di entropy with length raised AP from 0.582 to 0.860, against 0.727 for amino-acid entropy; all four features together reached 0.880 (fold SD ≤0.002; Fig. 2a). <!-- src: AH/manuscript/tables/S4_models_overall.tsv; LMS:396-410 --> 3Di entropy therefore carries most of the additional signal atop of length and the remaining features only contribute 0.020 AP.

![**Figure 2.** Structural entropy separates Prodigal genes from spurious ORFs beyond what ORF length explains (50,009 LOGAN metagenomes). Average precision (a–d) and ROC AUC (e–h) of gradient-boosted classifiers that separate complete Prodigal+ ORFs from complete Prodigal− ORFs, for four negative sets: all Prodigal− ORFs, antisense shadows, frameshift shadows and intergenic ORFs. Feature sets are nested on ORF length; the dashed line is length alone. Five-fold cross-validation was grouped by accession. Metrics are population-weighted (sample weight = 1/sampling rate) over 9.5 million sampled complete ORFs from all 90 shards. Error bars, ± SD across folds.](images/figL02_models_beyond_length.png){#fig:2 width=100%}



Within fixed length bins, the structural signal persists while length unsurprisingly falls towards chance (Fig. S3, S4; Table S3). For example, at 100–149 aa, within-bin AUC was 0.612 for length, 0.805 for amino-acid entropy, 0.896 for 3Di entropy and 0.902 for MI; at 150–199 aa, 3Di entropy reached 0.960 compared to 0.833 for amino acids 0.833, and at 200–299 aa 3Di entropy reached 0.988. <!-- src: manuscript/data/tabL03_single_feature_by_length.tsv.gz; LMS:332-336 --> Cross-validated models gave the same picture: at 100–149 aa, AUC 0.612 for length, 0.898 with 3Di and 0.940 with all features (Fig. S4a). <!-- src: AH/manuscript/tables/S5_models_by_length.tsv --> However, the shortest ORFs show scant evidence of 3Di entropy carrying much signal: within-bin 3Di AUC is 0.52 at 30–49 aa rising to 0.66 at 50–74 aa. <!-- src: tabL03 (three_di_entropy, 30-49, 50-74) -->

The complete-ORF model also recognised contig-end fragments without retraining (Fig. S5). For 5′-open fragments, AUC rose from 0.714 (length) to 0.791 and AP from 0.331 to 0.559; for 3′-open fragments, AUC from 0.697 to 0.854 and AP from 0.471 to 0.751. <!-- src: AH/manuscript/tables/S6_models_fragments.tsv; LMS:433-438 -->



## Structural entropy is specific to the reading frame <!-- AGENT: of what? The gene? -->

We hypothesised that if high 3Di entropy were a property of coding DNA, it might be shared by the other frames of a gene. We compared each of 8.82 million complete Prodigal genes with the best-overlapping Prodigal− ORF in each of its five alternative frames (Fig. 3). <!-- src: LMS:452-455; notebooks/figL05_six_frame.ipynb caption --> Length and amino-acid entropy are partly shared between frames (Spearman ρ 0.33–0.45 and 0.29–0.36), 12-state entropy very weakly (ρ 0.06–0.13), while 3Di entropy (ρ −0.01 to 0.02) and MI (ρ −0.04 to 0.01) are not (Fig. 3a). <!-- src: AH/manuscript/tables/S7_six_frame_inheritance.tsv (spearman); LMS:457-465 -->

![**Figure 3.** Structural entropy is specific to the reading frame: six-frame inheritance, discrimination and Δ for 8.82 million Prodigal genes. A 1% deterministic sample of 8.82 million complete Prodigal genes from all 90 shards. **(a)** Spearman correlation between the value in Prodigal's selected frame and in each of the five alternative frames (same strand +1 and +2; opposite strand a0–a2). Length and amino-acid entropy are partly shared between frames; 3Di entropy and the 3Di–12-state mutual information are not. **(b)** The fraction of genes for which each feature ranks the selected frame first, in the top two or in the top three of six, over the 5,722,417 genes with all five alternatives. Ties are counted fractionally; dotted lines give chance (1/6, 1/3, 1/2). **(c–g)** The distribution of Δ, the selected-frame value minus the median of the five alternatives (central 99% shown). The fraction of genes with Δ > 0 is computed over all genes in the sample.](images/figL05_six_frame.png){#fig:3 width=100%}

Among the 5,722,417 genes with all five alternative frames containing an ORF, 3Di entropy ranked <!-- AGENT: ranked how? --> Prodigal's frame first in 92.4% of genes, 12-state entropy in 91.3%, MI in 85.2%, amino-acid entropy in 81.8% and length in 80.9%, against chance 16.7% (Fig. 3b). <!-- src: AH/manuscript/tables/S8_six_frame_discrimination.tsv; LMS:476-479 --> Among the same 5,722,417 genes, the selected frame exceeded the median of its alternatives by a median 1.99 bits of 3Di entropy, and did so in 97.0% (Fig. 3c–g, drawn from a 324,690-gene sample); over all 8.8 million genes with at least one alternative the values are 1.82 bits and 94.2%. <!-- src: S8_six_frame_discrimination.tsv; audit F (B, l.426) --> <!-- src: S8_six_frame_discrimination.tsv (median_delta, frac_delta_positive) --> Length still ranks the selected frame within the top three more often (98.8% against 97.0%), so the two sources of signal are complementary. <!-- src: S8 (top3_rate) --> 

## 3Di Entropy identifies missed protein-coding genes in GTDB <!-- AGENT: Overall this section is pretty good but needs less concise wording and a bit more explanation -->

**Identifying missing genes.** We hypothesised that the GTDB annotated genomes likely contain numerous real protein-encoding genes that were not identified correctly using gene prediction algorithms. Therefore, from the annotated genomes we kept 9,598,500 bacterial unmatched ORFs with 3Di entropy ≥ 2.5 bits. Of these, 94.5% overlap a called CDS (i.e. they are shadow ORFs). That leaves 523,346 bacterial and 22,447 archaeal candidate ORFs that might be real (Fig. 4a). <!-- src: GR §6 table; figG04_funnel.tsv.gz --> As expected, the number of candidates we identify is dependent on the genome size (Spearman ρ = 0.55 in bacteria) but is almost flat against contig N50 (ρ = 0.10) (Fig. S8). <!-- src: figGS03 caption; figGS03_burden_rho.tsv.gz --> <!-- AGENT: how many is this per genome -->

![**Figure 4.** Six-frame ORF calls that GenBank misses carry real genes in excess over a matched non-coding comparator (GTDB r232). **(a)** The candidate funnel, from every six-frame ORF (2,568,244,984 bacterial ORFs in 189,715 genomes; 54,858,398 archaeal ORFs in 10,122 genomes) to ORFs in annotated genomes that match no GenBank CDS, have 3Di entropy ≥ 2.5 bits and are not shadows of an annotated CDS: 523,346 bacterial and 22,447 archaeal candidates. **(b)** The percentage of each arm with a full-length Foldseek structural hit (query and target coverage ≥ 0.8) in each database. The arms are the candidates; antisense and frameshift shadows matched to them on length, 3Di entropy and contig truncation; and annotated CDS as the positive control (bacterial control arms capped at 200 ORFs per chunk). **(c)** The implied real-gene share, (candidate − comparator) / (annotated − comparator), with 95% confidence intervals. BFVD is excluded because annotated CDS reach only about 11% there. **(d)** Candidates meeting each evidence criterion (Table 3) against the matched comparator, weighted by draw multiplicity and scaled to the candidate arm. The labels give the excess, a conservative floor on the number of real genes.](images/figG04_missed_genes.png){#fig:4 width=100%}

**Structural homology against a matched comparator.** <!-- AGENT: another ARTEMIS etc style schematic showing a dummy example of how this was set up with the gene vs compartor would be great to add here or as a panel in the above figure --> In the top candidate 3Di quintile <!-- AGENT: is the quintile of 3Di entropy? it is unclear here -->, 88–99% of every protein-like arm hits a structure database <!-- AGENT: using foldseek search?? -->, antisense and frameshift shadows included, probably because a sequence-only encoder emits confident 3Di for any protein-like sequence. <!-- src: SA §2 (l.97–130); SA §12 item 5 --> We therefore compared candidates with antisense and frameshift shadows matched on length, 3Di entropy and truncation, counting only full-length hits (Methods).

For the technical null, we shuffled the 3Di strings of the bacterial candidates; at most 0.04% then had full-length hits. <!-- src: GR §6.1 --> In every database, candidates fall between the null and annotated CDS <!-- where are these annotated CDS? in teh same genome? -->. Against Swiss-Prot, the full-length hit rates are 10.3% for bacterial candidates, 7.4% for the comparator and 35.3% for annotated CDS (Fig. 4b). <!-- src: GR §6.1 table; v7 -->

The implied real-gene share using this measure of gene annotation as a measure in bacteria is 10.4% (95% CI 10.0–10.7) for Swiss-Prot, 9.0% (8.6–9.5) for PDB100 and 9.9% (9.3–10.5) for CATH50 (Fig. 4c). <!-- src: GR §6.1; MC "Result"; v7 share_* --> In archaea it is 15.6% (13.4–17.9), 16.2% (13.7–18.7) and 22.0% (18.5–25.4). <!-- src: GR §6.1; v7 --> <!-- AGENT: if only 30-40% of annotated CDS have hits - doesn't this imply higher percentage of real gene share? e.g 10.4/35.3 or something -->

The bacterial share barely moves when the comparator is weighted differently (10.3–10.6%). The archaeal Swiss-Prot share ranges from 14.0% to 19.8%, because few shadows match the long, high-3Di candidates. <!-- src: GR §6.1; MC "Result", "Reading" 2 -->

Together these imply **~51,000–59,000 protein-coding genes absent from GenBank annotation**: 47,300–54,200 in bacteria and 3,500–4,900 in archaea. <!-- src: GR §6.1; v7 structural_estimate; decision D1 --> <!-- AGENT: Across how many genomes? Better to quote this is the number per genome -->









**Table 3. Direct evidence for GTDB candidates against the matched comparator.** For each criterion the table gives the number of candidates, the comparator count and the excess. The comparator count is weighted by draw multiplicity and scaled to the candidate arm. The excess is a lower limit on the number of real missed genes. C3–C6 are each C2 plus one requirement and are not nested (in Archaea C6 exceeds C4); C7 is the strict combined rule (Methods). C2 has no excess by construction, because the comparator is matched on truncation.

| criterion | Bacteria: candidates | Bacteria: comparator | Bacteria: excess | Archaea: candidates | Archaea: comparator | Archaea: excess |
|---|---:|---:|---:|---:|---:|---:|
| C1 all candidates | 523,346 | 523,346 | 0 | 22,447 | 22,447 | 0 |
| C2 not contig-truncated | 483,767 | 483,767 | 0 | 19,835 | 19,835 | 0 |
| C3 C2 + full-length hit, any database | 90,658 | 65,038 | 25,620 | 5,141 | 3,410 | 1,731 |
| C4 C2 + full-length in ≥2 databases | 44,325 | 32,868 | 11,457 | 2,777 | 1,872 | 905 |
| C5 C2 + full-length in Swiss-Prot, PDB100 or CATH50 | 76,662 | 57,343 | 19,319 | 4,396 | 3,030 | 1,366 |
| C6 C2 + structure-only full-length hit | 81,257 | 59,429 | 21,828 | 4,439 | 3,030 | 1,409 |
| C7 strict combined rule | 23,540 | 18,566 | 4,974 | 1,574 | 1,167 | 407 |

<!-- src: Validation/gtdb/matched_comparator/{bac,arc}_v7_matched.tsv ladder_C*_{candidates,comparator_scaled,excess}; GR §6.1 ladder table -->

**An independent gene caller <!-- AGENT: I doubt it is full independent -->.** GTDB's Prodigal annotation uses neither GenBank nor structure. It calls a gene with the same stop and strand for these shares of each arm (Fig. S7a):

| arm | bacteria | archaea |
|---|---:|---:|
| candidates | 47.4% | 55.4% |
| matched comparator | 20.6% | 25.6% |
| annotated CDS | 94.2% | 96.1% |

<!-- src: GR §6.1; v7 prodigal_* -->

The implied Prodigal share <!-- AGENT: share of what? --> is 36.4% (36.2–36.6) in bacteria and 42.2% (41.1–43.2) in archaea, or **~200,000 genes** (190,597 + 9,468). <!-- src: GR §6.1; v7 prodigal_share_larger_bg --> We treat this as the upper limit of novel uncalled genes in these GTDB genomes. It is 3.5–4.0 times the structural estimate in bacteria and 1.9–2.7 times in archaea. <!-- src: GR §6.1; MC "Result" -->

However, Prodigal likely over-calls genes, given it calls 20.6% of the matched comparator, which are likely to be non-coding. Its false-positive rate on confidently non-coding sequence was not measured, and could be higher. The structural assay <!-- AGENT: assay is the wrong word here, maybe analysis is better --> may also undercount, because it finds full-length hits for only 24–36% of known genes per database (42–44% in any of Swiss-Prot, PDB100 or CATH50), against 94–96% for Prodigal. <!-- src: GR §6.1 "Where the two estimates disagree"; decision D2 --> Prodigal also shares the structural assay's preference for coding-like composition, which the matched comparator bounds but does not remove. <!-- src: GR §10 --> The ~5,400 strict floor depends on neither estimate. <!-- src: GR §6.1 -->

However, the two lines of evidence converge on the same ORFs as being likely missed genes in GTDB. Prodigal calls 91.8% of the bacterial C7 set (21,617 of 23,540) and 95.9% of the archaeal set (1,509 of 1,574), despite the fact that C7 never uses Prodigal. <!-- src: GR §6.1; SA §10 item 10 --> This compares with the earlier, unmatched comparator, with shadows meeting a C7-like clause coincided with Prodigal 56.7% (bacteria) and 68.4% (archaea) of the time. <!-- src: SA §10 item 10 table and reading (pre-v7 comparator) --> <!-- AGENT: a lot less than 90%+ is the point right? -->

**What the missed genes are.** 

In the C7 set of the most confident missed genes (Fig. S9a), metabolism function proteins account for 33.8% of bacterial and 41.3% of archaeal candidates, while mobile elements account for 13.6% and 11.8%. <!-- src: GR §6.1 functional table; figG06 caption --> Overall, the 23,393 bacterial C7 candidates with a Swiss-Prot product carry 8,065 distinct products. <!-- src: GR §6.1 -->



We then analysed ten examples in more detail, chosen across functional classes (Fig. S9b). Each one:
meets C7;
is called by Prodigal;
lies between two deposited CDS;
and sits in a genome at least 95% complete. <!-- src: GR §6.1 "Examples"; figG06 caption -->

Two of these examples fill the only gap in an otherwise complete genome locus:
- **Cas5** in GCA_049515795.1. A 226-aa candidate ORF lies 5 bp from *cas7* and 96 bp from *cas1*, in a co-oriented type I CRISPR–*cas* operon annotated with Cas6, Cas7, Cas8a1, Cas3′, Cas1 and Cas2 but not Cas5, which the type I Cascade complex requires. <!-- src: GR §6.1 "Examples"; figG06_examples.tsv.gz; figG06_neighbours.tsv.gz -->
- **SusC** in GCA_900230115.1. A 917-aa candidate fills the 3,029-bp gap in a polysaccharide utilisation locus that is annotated with SusD and SusE but lacks SusC. <!-- src: GR §6.1 "Examples"; figG06 data -->

The other eight include a RecF/Rad50-family ABC ATPase whose genomic context is metabolic rather than replicative. <!-- src: RI G-16; GR §6.1 "Examples" -->  <!-- AGENT: what are the others? Maybe it is best to add it as a table or something?? -->

We then looked at the effects of various different confounders and variables that could explain the rate of missed genes. We first looked at assembly quality. We recomputed the estimate within quartiles of contig N50 and of CheckM2 completeness, keeping each comparator draw with its candidate's genome (scheme (i); Methods). <!-- src: AS; decision D15 --> The excess is present in every stratum (Table S14), in both domains and for every database, with every lower CI bound above 2.9% (smallest 2.97%). The C3 and C7 excesses are positive throughout. <!-- src: AS "Reading" 1 --> Therefore, missed genes do not appear to be artefacts of poor assemblies.











## 3Di entropy also identifies missed-gene candidates in metagenomes

We then used the final model to score 49.95 million intergenic Prodigal− ORFs ≥100 aa. <!-- src: LMS:532-534 --> in the 50,009 Logan metagenomes. At the 0.90 precision <!-- AGENT: precision of what? the model --> threshold, 3,434,482 ORF scored highly as being likely protein coding genes <!-- AGENT: what does high mean? -->. These ORFs were detected across in 47,754 of the 50,009 metagenomes (Table S5). The overall rate of highly scoring intergenic Prodigal ORFs across biomes ranged from 5.5% of scanned ORFs in soil to 8.7% in human gut. <!-- src: manuscript/data/tabL05_candidate_counts.tsv.gz; AH/manuscript/tables/S14_candidate_biomes.tsv -->

We hypothesised that valid candidates would carry full-length structural homologues while the matched controls would not (Fig. 5a). Of 20,000 candidates <!-- AGENT: were these randomly sampled? -->, 41.9% had a full-length structural hit across five databases, compared to 72.6% of complete Prodigal genes, 0.21% of low-scoring intergenic ORFs (41 of 20,000), 0.45% of antisense shadows (89 of 20,000), 0.36% of frameshift shadows and 1 of 20,000 candidates with shuffled 3Di. <!-- src: AH/manuscript/tables/S33_all_arms_evidence.tsv; #24 LOGAN agent corrections --> Candidates thus reach 58% of the Prodigal-gene rate and about 200 times the low-score rate. <!-- src: S33 (0.41875/0.7259; 0.41875/0.00205) --> The shuffled null keeps each query's amino acids and 3Di composition, so the hits depend on the order of the predicted states. Any sequence hit <!-- AGENT: what method? MMseqs2? Foldseek? --> at E ≤ 10⁻³ is not evidence: it reached 23–26% for the low-score and antisense controls (Fig. 5b). <!-- src: S33 (hit_rate_seq 0.2608, 0.2291) -->

![**Figure 5.** Foldseek evidence for metagenomic missed-gene candidates and matched controls. Structural (3Di + amino acids, solid) and sequence-only (hatched) Foldseek evidence for 20,000 high-scoring intergenic ORFs ≥ 100 aa that Prodigal did not call (candidates, score ≥ the 0.90-precision threshold), and for arms matched to them on length and truncation: complete Prodigal genes (positive control), low-scoring intergenic ORFs, antisense shadows, frameshift shadows, and the candidates with their 3Di strings shuffled and amino acids unchanged. **(a)** Hits with full-length mutual coverage (query and target coverage ≥ 0.8), the primary readout. **(b)** Any hit at E ≤ 10⁻³. The databases are PDB100, AFDB/Swiss-Prot, CATH50, BFVD and AFDB50. The shuffled null has one full-length hit in 20,000.](images/figL06_missed_gene_evidence.png){#fig:5 width=100%}


AFDB50 gives full-length structural hits for 41.5% of candidates, against 7–12% for each other databases <!-- AGENT: this seems very unsuually, i would expect AFDB50 not to be that much better than e.g. swissprot -->, so a hit to AFDB50 often means that a homologous protein is already annotated in another genome. <!-- src: AH/manuscript/tables/S16_evidence_rates.tsv; LMS:600-604 --> Among candidates, 20.7% had both full-length sequence and structural evidence, 21.2% structure only and 4.0% sequence only, against 62.7%, 9.9% and 7.9% for Prodigal genes (Fig. S10b). <!-- src: AH/manuscript/tables/S17_evidence_classes.tsv --> The rate rose with length from 38.1% at 100–149 aa to 70.6% at 300–499 aa, roughly parallel to Prodigal genes (57.7% to 96.9%; Fig. S10a). <!-- src: AH/manuscript/tables/S18_evidence_by_length.tsv --> Weighted by the candidates' length distribution, about 38% of the 3.07 million complete and 5′-open candidates, roughly 1.2 million ORFs, would have a full-length structural homologue. <!-- src: LMS:613-616; tabL05 (2,759,781 + 305,573) -->

The best full-length target database hits <!-- AGENT: check this --> of candidates were 64.5% bacterial, 30.1% eukaryotic, 1.9% archaeal and 1.9% viral, against 88.4%, 2.9%, 2.1% and 4.7% for Prodigal genes (Fig. S10d). <!-- src: AH/manuscript/tables/S19_hit_superkingdom.tsv --> The eukaryotic share was highest in plant (73%), food (71%) and built-environment (51%) metagenomes. <!-- src: AH/manuscript/tables/S24_euk_by_biome.tsv (euk_hit_share) --> Kraken2 classified 84.4% of the query contigs (Fig. S11a). <!-- src: LMS:660-661 --> Candidates sat on eukaryotic contigs more often than Prodigal genes (31.6% against 9.7%), and of the 2,525 candidates with a eukaryotic best hit, 69.8% lay on eukaryotic contigs, 20.9% on unclassified and 9.3% on bacterial contigs (Fig. S11b; Table S9). <!-- src: AH/manuscript/tables/S22_contig_group_by_arm.tsv; manuscript/data/tabL09_euk_explained.tsv.gz (S23) --> Part of what Prodigal misses is therefore probably eukaryotic coding sequence, which a prokaryotic caller like Prodigal is not designed to find, although low-scoring intergenic ORFs also sit on eukaryotic contigs (38.2%). <!-- src: S22 (intergenic_lo) -->

For the 5,402 candidates <!-- AGENT: cadidates of what? --> with a bacterial best hit, Prodigal's own scores fall just below its decision boundary (Fig. S12; Table S8). 49.6% have a negative coding score under the model chosen for the contig, 42.2% a positive coding score made negative by start, RBS and upstream penalties, 5.0% no potential gene at that stop, and 3.3% a positive score but were not chosen. <!-- src: manuscript/data/tabL08_skip_reasons.tsv.gz (S25) --> Their median coding score was −0.6 against +42.1 for called genes with bacterial hits; 42.9% of their best starts were ATG (89.6% for called genes), and 61.9% had no RBS motif (19.2%). <!-- src: AH/manuscript/tables/S26_skip_feature_summary.tsv --> These bacterial candidates look like genes whose codon usage and start signals fall outside Prodigal's metagenomic models <!-- AGENT: duoes this suggest a way to imrpove prodigal's models using Logan? Thoughts> -->.

AFDB50 alone gives full-length structural hits for 41.5% of candidates, against 7–12% for each other database, so a hit often means that a homologous protein is already annotated in another genome. AFDB50 is also by far the largest database searched: its 66.7 million entries are more than 100 times as many as Swiss-Prot (0.59 million), BFVD (0.35 million), PDB100 (0.32 million) or CATH50 (0.20 million). <!-- src: Foldseek .index entry counts on Gadi, gtdb_entropy/foldseek_db: afdb50 66,725,340 (v6 2025_03, the release LOGAN searched); afdb_swissprot 590,183 (v6 2025_03), bfvd 351,242 (2023_02_v2), pdb100 324,204 (2025-01-01), cath50 195,223 (v4.3.0). LOGAN's releases of the four smaller databases are not recorded. --> <!-- src: AH/manuscript/tables/S16_evidence_rates.tsv; LMS:600-604 --> Among candidate protein-encoding genes, 20.7% had both full-length sequence and structural evidence, 21.2% structure only and 4.0% sequence only, against 62.7%, 9.9% and 7.9% for Prodigal genes (Fig. S10b). <!-- src: AH/manuscript/tables/S17_evidence_classes.tsv --> The rate rose with length from 38.1% at 100–149 aa to 70.6% at 300–499 aa, roughly parallel to Prodigal genes (57.7% to 96.9%; Fig. S10a). <!-- src: AH/manuscript/tables/S18_evidence_by_length.tsv --> Weighted by the candidates' length distribution, about 38% of the 3.07 million complete and 5′-open candidates, roughly 1.2 million ORFs, would have a full-length structural homologue. <!-- src: LMS:613-616; tabL05 (2,759,781 + 305,573) -->

Taxonomically, we found that the best full-length targets of candidates were 64.5% Bacterial, 30.1% Eukaryotic, 1.9% Archaeal and 1.9% Viral, against 88.4%, 2.9%, 2.1% and 4.7% for Prodigal genes (Fig. S10d). Perhaps not surprisingly, Prodigal is less able to predict Eukaryotic and Viral protein-encoding genes than Bacterial or Archael sequences. <!-- src: AH/manuscript/tables/S19_hit_superkingdom.tsv --> The Eukaryotic share was highest in plant (73%), food (71%) and built-environment (51%) metagenomes. <!-- src: AH/manuscript/tables/S24_euk_by_biome.tsv (euk_hit_share) --> Kraken2 classified 84.4% of the query contigs (Fig. S11a). <!-- src: LMS:660-661 --> Candidate protein-encoding genes were present on contigs Kraken2 annotated as Eukaryotic more often than Prodigal genes (31.6% against 9.7%), and of the 2,525 candidates with a Eukaryotic best hit, 69.8%, lay on contigs classified as Eukaryotic, 20.9% on unclassified contigs, and 9.3% on contigs classified as Bacterial by Kraken2 (Fig. S11b; Table S9). <!-- src: AH/manuscript/tables/S22_contig_group_by_arm.tsv; manuscript/data/tabL09_euk_explained.tsv.gz (S23) --> Part of what Prodigal misses is therefore probably Eukaryotic coding sequence, which a prokaryotic caller is not designed to find. <!-- src: S22 (intergenic_lo) -->

For the 5,402 candidates with a Bacterial best hit, Prodigal's own scores fall just below its decision boundary (Fig. S12; Table S8). 49.6% have a negative coding score under the model chosen for the contig, 42.2% a positive coding score made negative by start, RBS and upstream penalties, 5.0% no potential gene at that stop, and 3.3% a positive score but were not chosen. <!-- src: manuscript/data/tabL08_skip_reasons.tsv.gz (S25) --> Their median coding score was −0.6 against +42.1 for called genes with Bacterial hits; 42.9% of their best starts were ATG (89.6% for called genes), and 61.9% had no RBS motif (19.2%). <!-- src: AH/manuscript/tables/S26_skip_feature_summary.tsv --> These Bacterial candidates look like genes whose codon usage and start signals fall outside Prodigal's metagenomic models, suggesting they may originate in organisms whose genomes were not present in Prodigals training sets.

## Structural Entropy can identify likely Prodigal false-positive genes

Of the scan's 880,405 sampled complete Prodigal genes <!-- AGENT: this seems super low for 50,000 metagenomes?? -->, 17.4% scored below the 0.80 threshold, almost all of them short: more than 99.9% of genes at 30–49 aa, 88% at 50–74 aa, 55% at 75–99 aa, 23% at 100–149 aa and 2% at 200–299 aa (Fig. S13a). <!-- src: AH/manuscript/tables/S29_lowscore_population.tsv; LMS:725-730 --> Within each length bin from 50 aa, low-scoring genes had much lower 3Di entropy than high-scoring genes (0.9–1.5 against 2.9–3.5 bits), while amino-acid entropy differed by less than 0.3 bits (Table S7). <!-- src: manuscript/data/tabL07_lowscore_features.tsv.gz -->

Length-matched, 6.8% of 10,269 low-scoring genes had a full-length structural homologue <!-- AGENT: against which DB -->, against 87.5% of 8,769 high-scoring genes; full-length sequence hits were 28.3% against 79.8% (Fig. S13b). <!-- src: AH/manuscript/tables/S33_all_arms_evidence.tsv (prodigal_lo, prodigal_hi); tabL06; #24 correction 6.8% --> 70.8% of low-scoring genes had no full-length evidence of either kind, and 22.4% had sequence hits only, mostly to UniProt entries in AFDB50 (89%), often near-identical (48% at ≥90% identity). <!-- src: AH/manuscript/tables/S31_extra_evidence_classes.tsv; LMS:752-758 --> The relation is graded: among Prodigal genes in the positive-control arm, the full-length structural rate rose from 2.7% at scores below 0.2 to 95.1% at 0.95 and above. <!-- src: AH/manuscript/tables/S32_prodigal_by_score.tsv --> Outside the shortest genes, a low score therefore marks Prodigal calls that lack independent protein-level support; whether these are spurious calls or proteins without a folded domain was not tested. Below 75 aa, low scores are nearly universal and likely reflect a limit of 3Di entropy for short proteins.

## Structural entropy finds missing and misannotated genes in phage

### Structural entropy carries most of the information about phage annotation

Phage genomes are extremely dense (often with coding densities above 90-95% of the genome [@10.1093/ismeco/ycae079]).: 39.8% of the 6,754,361 ORFs of at least 90 aa in 36,025 INPHARED genomes match an annotated CDS. <!-- src: Validation/mutual_information/README.md TL;DR (2,685,382 / 6,754,361); IG:14 -->  

3Di entropy alone carries 0.587 bits of information <!-- AGENT: what does this mean concretely? --> about the label (95% CI 0.584–0.589), out of 1 bit on a class-balanced sample (Fig. 6a). <!-- src: Validation/mutual_information/README.md §4.2 table; data/figP01_mutual_information.tsv.gz --> As a single score it gives a ROC AUC of 0.947. <!-- src: Validation/mutual_information/README.md §4.6 --> The best pair, protein plus 3Di entropy, reaches 0.644 bits, 0.057 bits (95% CI 0.055–0.058) more than 3Di entropy alone. <!-- src: Validation/mutual_information/README.md TL;DR, §4.2 table -->  <!-- src: Validation/mutual_information/README.md TL;DR --> DNA entropy carries essentially nothing (0.003 bits). <!-- src: Validation/mutual_information/README.md §4.2 table -->  <!-- src: Validation/mutual_information/README.md TL;DR -->

![**Figure 6.** Phage: structural entropy predicts annotation, ranks missing genes, and corrects an annotation. ORFs ≥ 90 aa in Inphared phage genomes, labelled as genes when they match a Pharokka + phold CDS under the corrected matcher. **(a)** Mutual information between each per-ORF variable and the label, on a class-balanced sample (20 bins per axis; bars, bootstrap 95% CI); 36,025 genomes, 6,754,361 ORFs, 39.8% genes. Numbers give the share of each variable's information that survives conditioning on length. **(b)** Out-of-fold ROC AUC and average precision of ranking models grouped by genome (XGBoost random forest; 31,200 genomes, 5,887,690 ORFs) on corrected labels. The legacy model, trained on labels that missed every minus-strand gene, learned that the plus strand means a gene and fails on corrected labels. The dotted line is the prevalence (0.40), the average precision of a random ranking. **(c)** Full-length Foldseek structural hits (query and target coverage ≥ 0.8) for 20,000 ORFs per arm (4,102 and 4,039 for the replacement pairs) against AFDB/Swiss-Prot, PDB100, CATH50 and BFVD, or excluding the viral BFVD; Wilson 95% CI. The controls are low-scoring antisense, frameshift and intergenic ORFs, and the candidates with their 3Di strings shuffled. **(d)** Phage OM869708 (*Dipodfec virus* RodF1_38, 5,753 bp). Top: CDS in the submitters' GenBank record and in the Pharokka + phold re-annotation used as labels (grey). Below: all ORFs ≥ 90 aa by frame, coloured by the corrected model's coding probability; a black outline marks an ORF that matches a re-annotation CDS. The submitters' replication initiator (UPW42009.1, red outline) was dropped by the re-annotation, which called a hypothetical protein in another frame (CDS_0007 = ORF7). The model scores ORF3, which contains the submitters' gene, at p = 0.89 and ORF7 at p = 0.18.](images/figP01_phage.png){#fig:6 width=100%}

Unlike with Bacteria, phage ORF predictions are less susceptible to the length effect. Every ORF here is at least 90 aa, and length alone is a weak predictor (0.130 bits; AUC 0.733). <!-- src: Validation/mutual_information/README.md §4.2 table, §4.6 --> Within length bins, 3Di entropy gives AUC 0.874 at 90–99 aa, rising to 0.991 at 300 aa and above, against 0.53–0.63 for length. <!-- src: Validation/mutual_information/README.md §4.3 AUC table --> This maybe because phage genes are inherently shorter than their bacterial counterparts [@Akhter2012-an].

The signal is not a length effect. Every ORF here is at least 90 aa, and length alone shows a weak effect (0.130 bits; AUC 0.733). <!-- src: Validation/mutual_information/README.md §4.2 table, §4.6 --> Conditioning on length keeps 82% of the information in 3Di entropy (0.472 of 0.572 bits, full population). <!-- src: Validation/mutual_information/README.md §4.3 conditional-MI table --> Within length bins, 3Di entropy gives AUC 0.874 at 90–99 aa, rising to 0.991 at 300 aa and above, against 0.53–0.63 for length. <!-- src: Validation/mutual_information/README.md §4.3 AUC table -->

Amino-acid composition therefore adds little to predicted structure. We note that the pseudo ground-truth labels here are Pyrodigal-gv gene calls, so these values measure agreement with that gene caller.

### Ranking missing genes with corrected labels

We trained a random forest [@Breiman2001-yx], implemented as XGBoost's `XGBRFClassifier` [@Chen2016-cv] (Methods), <!-- src: Validation/Inphared/scripts/03_models.py:41, :60-61; Validation/Inphared/README.md:78 --> on labels using intrinsic features only: entropies and lengths, with no strand, frame or position. On the 31,200-genome ranking set this model gives AUC 0.976 and average precision (AP) 0.970, against 0.733 and 0.660 for length alone (Fig. 6b). <!-- src: Validation/Inphared/README.md Table 1 (intrinsic 0.9764 / 0.9696; length only 0.7328 / 0.6598); data/figP01_ranking_models.tsv.gz --> Length plus 3Di entropy already reaches 0.953 and 0.945. <!-- src: Validation/Inphared/README.md Table 1 --> The ranking set's positives make up 40% of ORFs, so AP here is not comparable with AP on metagenomes. <!-- src: Validation/Inphared/README.md:139, :192 -->

Using entropies and lengths, with no strand, frame or position, on the 31,200-genome ranking set, this model gives AUC 0.976 and average precision (AP) 0.970, against 0.733 and 0.660 for length alone (Fig. 6b) <!-- AGENT: remove the legacy model from Figure 6b, no one cares about it -->. <!-- src: Validation/Inphared/README.md Table 1 (intrinsic 0.9764 / 0.9696; length only 0.7328 / 0.6598); data/figP01_ranking_models.tsv.gz --> Length plus 3Di entropy already reaches 0.953 and 0.945. <!-- src: Validation/Inphared/README.md Table 1 --> 3Di entropy takes 0.745 of the model's gain. <!-- src: Validation/Inphared/README.md §4.2 intrinsic table --> We note that ranking set's positives make up 40% of ORFs, so AP here is not comparable with AP on metagenomes presented previouslt. <!-- src: Validation/Inphared/README.md:139, :192 -->

The model flags 46,132 unannotated ORFs at p ≥ 0.8 <!-- AGENT: what is P here? precision? -->, in 19,560 genomes (median 136 aa) <!-- AGENT: what is the range per genome? -->. <!-- src: Validation/Inphared/README.md:210 --> Most overlap an annotated gene:

- 22,913 are antisense to a called CDS;
- 15,005 lie in another frame on the same strand;
- 4,169 are intergenic.

<!-- src: Validation/Inphared/README.md §5.1 candidate context table -->

The remaining 4,045 are fragments of annotated genes, mostly partial CDS at genome ends, and were not considered further. <!-- src: Validation/Inphared/README.md:224 -->

Structural searches <!-- AGENT: against what databases? --> support the candidates (Fig. 6c). Full-length Foldseek hits in any database were found for:

- 33.8% of candidates (95% CI 33.1–34.5%);
- 64.6% of annotated genes;
- 2.2%, 0.24% and 0.10% of low-scoring intergenic, other-frame and antisense ORFs;
- 0.015% of candidates with shuffled 3Di strings.

<!-- src: Validation/Inphared/README.md §5.2 rate table; data/figP01_foldseek_rates.tsv.gz (fullcov_ci) -->

Scaled between the controls and the annotated genes, this places 51–52% of candidates with the genes <!-- AGENT: This sentence is garbled and makes no sense -->. <!-- src: Validation/Inphared/README.md §5.2 excess table (50.6–52.3%); tables/evidence_excess.tsv --> Most of that evidence comes from BFVD, a database of predicted structures of viral proteins. Its hits are mostly near-identical <!-- AGENT: in terms of sequence identity? --> (median identity 0.98 for candidates), so they mostly show that another annotation of a related genome that called the same ORF. <!-- src: Validation/Inphared/README.md §5.2 identity table --> The independent test is the non-viral databases. There candidates reach 6.1% (95% CI 5.8–6.5%), against 18.0% for annotated genes and at most 0.09% for every control. <!-- src: Validation/Inphared/README.md §5.2; data/figP01_foldseek_rates.tsv.gz --> That gives a share of 34%. <!-- src: Validation/Inphared/README.md §5.2 excess table (33.7–34.0%) -->

Not surprisingly, the antisense candidate ORFs have far fewer structural similarities: 9.9% have a full-length hit. In contrast, 66.5% of intergenic ORFs and 61.6% of other-frame ORF candidates have similarity, although many of these may be shadow ORFs from genes on the other strand. <!-- src: Validation/Inphared/README.md §5.2 by-context table --> 

A replacement pair links an unannotated ORF that the model scores highly (p ≥ 0.8) to an annotated gene that it overlaps and that the model scores low (p ≤ 0.5, at least 0.4 below the unannotated ORF). Where several annotated genes overlap the ORF, the pair uses the one with the greatest overlap (Methods). Each pair is therefore a choice between the gene caller's model at that locus and the overlapping ORF that the structural score prefers. In 57% of the 4,102 pairs the two ORFs are on the same strand, and the median overlap is 70% of the shorter ORF. Only the replacement ORF has a full-length hit in 46.6% of pairs, and only the annotated gene in 5.9%; both hit in 7.2% and neither in 40.3%. The replacement pairs suggest incorrectly identified gene features in the sequence databases, however phages are known to carry overlapping genes, so each case needs checking before it is called an annotation error.


A replacement pair links an unannotated ORF that the model scores highly (p ≥ 0.8) to an annotated gene that it overlaps and that the model scores low (p ≤ 0.5, at least 0.4 below the unannotated ORF). Where several annotated genes overlap the ORF, the pair uses the one with the greatest overlap (Methods). <!-- src: scripts/rank_missing_orfs.py defaults; Methods ¶ on replacement candidates --> Each pair is therefore a choice between the gene caller's model at that locus and the overlapping ORF that the structural score prefers. In 57% of the 4,102 pairs the two ORFs are on the same strand, and the median overlap is 70% of the shorter ORF. <!-- src: Validation/Inphared/README.md §6 --> Only the replacement ORF has a full-length hit in 46.6% of pairs, and only the annotated gene in 5.9%; both hit in 7.2% and neither in 40.3%. <!-- src: Validation/Inphared/README.md §6 table --> Replacement ORFs hit 53.8% of the time, the genes they would replace 13.2%. <!-- src: Validation/Inphared/README.md §6 hit-rate table --> This test shows only which ORF of a pair has better external evidence. Phages do carry overlapping genes, so each case needs checking before it is called an annotation error.

Two limitations apply:

- **Short genes.** The 90-aa floor hides 28.7% of the annotated CDS, so these results say nothing about short phage genes. <!-- src: Validation/mutual_information/README.md §2.1 (957,269 / 3,336,744) -->
- **Genetic code.** Every phage run calls ORFs with genetic code 11, including in the 104 genomes whose genes use code 4 or 15.
- **Near-duplicate genomes.** Inphared contains near-duplicate genomes, and cross-validation was grouped by genome, not by homology cluster. Out-of-fold metrics may therefore be slightly optimistic.

### Phage genes sit lower in 3Di entropy than cellular genes

Annotated phage genes show the same pattern of structural entropy separation as bacterial and archaeal genes, and the gap widens with length (Fig. S14). However,aAnnotated phage ORFs have lower mean 3Di entropy than annotated bacterial ORFs in every length bin from 90 to 999 aa:

| length (aa) | difference (bits) |
|---|---|
| 90–149 | 0.24 |
| 150–299 | 0.34 |
| 300–599 | 0.19 |
| 600–999 | 0.09 |

<!-- src: notebooks/figP04_phage_vs_cellular.ipynb caption (a); recomputed from data/figP04_{phage,gtdb}_length_3di.tsv.gz as n-weighted means: 0.242, 0.339, 0.194, 0.090 -->

This indicats that phage proteomes may be richer in small or poorly folded proteins. <!-- AGENT: I disagree with this, it is likely thinner MSAs and worse predictions aka lower pLDDTs for structures, shouldnt be richer in length being relevant here as these are per length bin --> Alternatively, the labels may contribute: 310,360 phage positives match a CDS shorter than 90 aa inside a longer stop-to-stop ORF. <!-- src: AH inphared_gadi/README.md §3.2; Validation/mutual_information/README.md §2.1 --> Restricting positives to CDS that cover most of their ORF would separate the two. <!-- AGENT: don't just say this, otherwise test this -->

### An annotation error in OM869708

OM869708 is the 5,753-bp genome of *Dipodfec virus* RodF1_38, a member of the *Microviricetes* (Fig. 6d). <!-- src: work/om869708/OM869708_ncbi.gb (DEFINITION, ORGANISM, REFERENCE bases 1 to 5753) --> The submitters' GenBank record annotates a replication initiator protein at 4753..5730 (UPW42009.1). <!-- src: data/figP01_om869708_ncbi_cds.tsv.gz (0-based 4752–5730); decisions.md D7 --> The Pharokka and phold re-annotation used for our labels does not contain this gene. <!-- AGENT: was phanotate or prodigal used to do gene calling? That is more important than pharokka/phold, which does the functional annotation --> Instead it calls a hypothetical protein in another frame, CDS_0007, which matches ORF7. <!-- src: data/figP01_om869708_cds.tsv.gz; data/figP01_om869708_orfs.tsv.gz -->

The corrected model disagrees with the re-annotation. ORF3 is the 335-aa stop-to-stop ORF that contains the submitters' gene and shares its stop codon. It has 3Di entropy 2.98 bits and scores p = 0.889. <!-- src: data/figP01_om869708_orfs.tsv.gz (orf3); decisions.md D7 --> The annotated ORF7 (198 aa) has 3Di entropy 1.20 bits and scores p = 0.175. <!-- src: data/figP01_om869708_orfs.tsv.gz (orf7); decisions.md D7 --> The model ranks ORF3 → ORF7 179th of the 4,102 replacement candidates. <!-- src: data/figP01_om869708_orfs.tsv.gz (replacement_rank); decisions.md D7 --> <!-- AGENT: how does the ranking work? -->

Running BLASTP against nr <!-- AGENT: what is the full name of nr? --> agrees with the model. ORF7 has no hit at E ≤ 10⁻⁵, <!-- src: data/figP01_om869708_blastp.tsv.gz (no OM869708_orf7 rows) --> while ORF3's top hit is the genome's own submitted protein (UPW42009.1, 100% identity). <!-- src: data/figP01_om869708_blastp.tsv.gz row 1 --> The next five hits are replication initiator proteins of other viruses, one of them a named microvirus, at 43.7–55.4% sequence identity. The best of these aligns over 323 aa with E = 2 × 10⁻¹²⁷. <!-- src: data/figP01_om869708_blastp.tsv.gz rows 2–6 (XHH71331.1 55.4%, E 2.4e-127 … QXN75198.1 43.7%) -->

The genome_entropy entropy therefore recovers a replication gene that the submitters annotated but that the automated phage annotation pipeline lost <!-- AGENT: add this as a citation https://doi.org/10.1002/cpz1.70405 -->. The functional annotation step only describes proteins that its gene caller <!-- AGENT: which gene caller --> reports, so a wrong gene model is not corrected downstream. This example shows how one score calculated via genome_entropy per ORF can flag such a locus for review.

## Short ORFs mark the lower limit of the genome entropy method

On the GMSC benchmark<!-- AGENT: can you please add a sentence describing the GMSC benchmark in more detail -->, structural entropy separates real from spurious small ORFs only weakly (Fig. S16) <!-- AGENT: probably this needs to be a main figure not a supp? -->. Pooled over all 43.7 million smORFs from GMSC, length is the best single predictive feature (AUC 0.640). 3Di entropy (0.605) and 12-state entropy (0.602) come next. <!-- src: Validation/smorfs/README.md:74–80; data/figS01_single_feature_auc.tsv.gz --> Nucleotide and amino-acid entropy barely separate the classes (0.504 and 0.525). <!-- src: data/figS01_single_feature_auc.tsv.gz (0.5037, 0.5254) --> Positives are on average longer (median 80 against 70 aa; Table S13). <!-- src: data/tabS04_dataset_summary.tsv.gz -->

3Di entropy keeps a small signal that grows with length: AUC rises from 0.526 at 50–59 aa to 0.626 at 90–99 aa (Fig. S16d). <!-- src: AH smorfs/manuscript.md Table 2; data/figS01_single_feature_auc.tsv.gz --> Amino-acid entropy falls to or below chance in the same bins (0.46–0.50). <!-- src: AH smorfs/manuscript.md Table 2 --> At 10–29 aa, 3Di entropy is completely uninformative (AUC 0.47). However, this bin holds only 11,382 positives against 80,578 negatives (Table S10). <!-- src: data/figS01_single_feature_auc.tsv.gz (0.4727); data/tabS01_length_bins.tsv.gz -->

Combining features helps, but the ceiling is low (Fig. S16e) <!-- AGENT: this should be a main figure, i think S16 overall should be -->:

- length plus nucleotide and amino-acid entropy reaches held-out AUC 0.669;
- length plus 3Di and 12-state entropy reaches 0.699;
- all five features reach 0.730.

<!-- src: AH smorfs/manuscript.md Table 3; data/figS01_models_by_split.tsv.gz (0.6692, 0.6993, 0.7300) -->

Holding out homology clusters barely changes this: 0.728 at 50% sequence identity and 0.721 at 30%. <!-- src: Validation/smorfs/README.md:86–90 --> Weighting each 50%-identity cluster once lowers the all-feature AUC to 0.695 under the 30% split. <!-- src: Validation/smorfs/README.md:90 --> The positives are far more redundant than the negatives <!-- AGENT: note that this makes sense via the definition of the cluisters, which is based on number of members I believe? Please check it -->: 14.5 against 6.0 sequences per 50%-identity cluster, and the 20 largest clusters at each threshold are all positive (Tables S11, S12). <!-- src: data/tabS02_cluster_stats.tsv.gz; data/tabS03_cluster_largest.tsv.gz; AH smorfs/manuscript.md §3.7 --> Cluster weighting lowers 3Di AUC more than the other features (0.612 to 0.571, random split). Part of the structural signal therefore sits in large positive families. <!-- src: data/figS01_models_by_split.tsv.gz (3Di, random: 0.6122 / 0.5710) -->

The amino-acid × 3Di entropy plane shows where the classes differ (Fig. S17) <!-- AGENT: perhaps add this as a pane to SF16 and put it in the main figure as discussed above -->. Positives are enriched above about 2.3 bits of 3Di entropy when amino-acid entropy exceeds about 3.6 bits. Negatives are enriched in a band at 1.0–1.6 bits of 3Di entropy. <!-- src: notebooks/figSS01_smorf_density.ipynb caption --> Sharp stripes at 1.0 and 1.585 bits are strings that use only two or three 3Di states (Fig. S18). In the 1.0-bit stripe, 67% of positive and 61% of negative strings use exactly two states, almost always D and P. <!-- src: data/figSS02_stripe_bands.tsv.gz (67.43 / 60.68%; DP) --> In the 1.585-bit stripe, 42% and 31% use exactly three, mostly D, P and V. <!-- src: data/figSS02_stripe_bands.tsv.gz (41.81 / 30.74%; DPV) --> Such strings cannot exceed log₂2 or log₂3 bits and accumulate just below those values. Both classes contain them at similar rates, so low 3Di entropy alone does not mark a small ORF as non-coding. <!-- AGENT: cite ProstT5 or whatever, this is a known issue -->

A short line near 2.81 bits contains only positives. Two families of near-identical 3Di strings appear to form it, which we identified by eye as ribosomal proteins L34 and S14. <!-- src: AH smorfs/manuscript.md §3.6 --> We did not confirm these identities by homology search, and we do not interpret the stripes biologically. <!-- AGENT: I would remove this sentence, it is not important to the story -->

These results set the method's lower length limit. Gene-length ORFs separate well: in phage genomes 3Di entropy alone reaches AUC 0.947. <!-- src: Validation/mutual_information/README.md §4.6 --> Below 100 aa, entropy carries only a small signal. The GMSC negatives are hard: they are conserved and in frame, unlike the shadows and random ORFs that dominate genomic negatives. Even so, much of the weakness is probably length itself. A 3Di string of 50–99 symbols gives few observations for a 20-state distribution, and many real small proteins may be a single helix or disordered. Features that use the order of the 3Di string may do better <!-- AGENT: what does this mean? -->, but we have not tested them.

# Discussion <!-- AGENT: This section generally needs more a discussion and history of our results and how they interact with existing literature with citations - not just a restatement of our results. A good paper to cite is  https://www.pnas.org/doi/10.1073/pnas.2536965123 -->

Across bacterial and archaeal genomes, metagenomes and phages, this study shows that the entropy of a predicted 3Di structural alphabet separates annotated genes from other ORFs far better than amino-acid composition does. This is repeatedly shown over a variety of microbial genomes across different analyses. In phages, 3Di entropy alone carries 0.587 bits of information about phage annotation, and adding amino-acid entropy contributes only 0.057 bits more. Across prokaryotic genomes in the GTDB, the 3Di entropy gap between matched and unmatched ORFs is five times the protein gap, which across 50,009 metagenome assemblies from LOGAN, within fixed length bins, 3Di entropy separates genes from non-genes where length alone does not.

Overall, we found that predicted structure does most of the work, amino-acid entropy adds little, and much of protein entropy's apparent rise with length is estimator bias.

Two observations explain why it works, without yet explaining it mechanistically.
1. Most unannotated ORFs pile up against the log₂(3) ceiling, which is the the maximum entropy of a string of three 3Di states. Three 3di encoding states (D, V and P) made up 99.4% of residues below this threshold. This is not unexpected, as the Foldseek 3Di encoding space tends towards these tokens when the structure is disordered or poorly predicted <!-- AGENT: cite ProstT5 paper -->
2. The signal belongs to the translated frame. Length and amino-acid entropy are partly shared between a gene's six frames, but 3Di entropy is not. Yet the gene's own frame ranks first among six in 92.4% of genes. <!-- AGENT: this dot point is pretty vague --> This suggests that the 3Di captures properties of real protein sequences that their shadow translations lack.



**Gene callers and structural entropy can disagree in both directions, and the disagreements are informative.**

3Di entropy can be used to identify likely missed genes. In GTDB genomes, we estimate 51,000–59,000 unannotated ORFs are likely to be protein-coding genes, when tested against a comparator matched on length, 3Di entropy and truncation. A strict individual test is passed by 25,114 candidates.
Notably, a number of these examples are biologically coherent when analysed in more detail, including a Cas5 that completes a type I CRISPR–*cas* operon and a SusC that completes a polysaccharide utilisation locus.

High-scoring intergenic ORFs that Prodigal skipped in metagenomes also have full-length structural homologues 90–200 times as often as matched controls, suggesting 3Di entropy can be used to screen metagenomes for missed genes as well. For the bacterial ones, Prodigal's own scores fall just short of its decision boundary, suggesting these are mostly false negatives gene callers like Prodigal just miss calling.
Interesting, in the other direction, 17.4% of complete Prodigal genes in metagenomes score low according to structural entropy, representing likely false positive gene calls. Outside the shortest lengths, these rarely have structural homologues.

A structural score is therefore a complement to a gene caller, not a replacement. It is most useful where the two disagree as a diagnostic. Obvious limitations to this approach is that this method does not show that the ORF is expressed or functional - interesting counterexamples identified by structural entropy are best prioritised list for transcriptomic, proteomic or experimental confirmation.








**Structural entropy can be used to refine gene calls from automated annotation pipelines.**

In OM869708 the re-annotation pipeline dropped a replication initiator gene that the genome's submitters had annotated, and called a hypothetical protein in another frame instead. The structural score preferred the submitters' gene, and independent homologues support it. As per-ORF score is cheap enough <!-- AGENT: estimate resource use? --> to run an automatically annotated microbial genome, 3Di entropy can be used to flag loci where the chosen gene model looks unlike a protein, along with the reverse.


**Phage ORFs of 10–90 aa.** To tell a property of short ORFs from a property of the GMSC benchmark, we encoded every phage ORF of 10–90 aa (Methods). The run gave 102,133,250 ORFs in 36,025 genomes. It agrees with the ≥ 90-aa run where the two overlap: all 123,650 ORFs of 90 aa in that run were found again. Their labels and DNA and protein entropies are identical, and their 3Di entropy is identical for 98.7% (largest difference 0.19 bits). <!-- src: AH inphared_gadi/short_orfs/README.md (90f4fc6) check 1; tables/all_genomes/boundary_90aa.tsv --> The 260 additional ORFs of 90 aa in the new run all lack a stop codon. <!-- src: AH inphared_gadi/short_orfs/README.md (90f4fc6) check 1 (248 at a contig end; 12 internal, at ambiguity codes) -->

Annotation is sparse at these lengths. Pyrodigal-gv called 2,939 genes of 10–29 aa, 112,252 of 30–49 aa, 394,802 of 50–69 aa and 468,136 of 70–90 aa, against 2,362,723 longer genes. <!-- src: AH inphared_gadi/short_orfs/README.md (90f4fc6) check 2; data/figP05_cds_length_bins.tsv.gz --> In the 104 genomes that reassign a stop codon, ORFs called with code 11 are fragments of longer genes, and they make up 95% of the matched ORFs of 10–29 aa (9,355 of 9,813), so we excluded these genomes (Methods). <!-- src: data/figP05_matched_by_code.tsv.gz --> In the remaining 35,921 genomes, the share of ORFs that match a gene falls from 10.9% at 70–90 aa to 0.0006% at 10–29 aa (Table 4).

**Table 4. Single-feature ROC AUC for phage ORFs of 10–90 aa (35,921 genomes that use genetic code 11) and for GMSC smORFs, by length.** Matched (GMSC: positive) is the positive class.

| length (aa) | phage matched / unmatched | phage, 3Di | phage, protein | GMSC, 3Di | GMSC, protein |
|---|---|---:|---:|---:|---:|
| 10–29 | 458 / 71.0 M | 0.571 | 0.709 | 0.473 | 0.490 |
| 30–49 | 37,323 / 19.4 M | 0.575 | 0.698 | 0.608 | 0.517 |
| 50–69 | 218,070 / 7.13 M | 0.764 | 0.776 | 0.553 | 0.471 |
| 70–90 | 398,822 / 3.25 M | 0.838 | 0.829 | 0.576 | 0.468 |

<!-- src: data/figP05_auc_by_bin.tsv.gz (genomes code11 and gmsc, bins27); data/figP05_class_by_bin.tsv.gz -->

Below 50 aa, 3Di entropy separates short phage ORFs about as weakly as it separates GMSC smORFs (Fig. S19a). From 50 aa it separates them much better: AUC 0.711 at 50–59 aa, rising to 0.848 at 80–89 aa, close to the 0.874 of phage ORFs of 90–99 aa. <!-- src: data/figP05_auc_by_bin.tsv.gz (gmsc_bins); Validation/mutual_information/README.md §4.3 --> Most of that difference lies in the negatives. At 70–90 aa, annotated phage ORFs average 2.12 bits of 3Di entropy against 1.87 for GMSC positives, but unannotated phage ORFs average 1.27 bits against 1.62 for GMSC negatives (Fig. S19b, c). <!-- src: data/figP05_class_by_bin.tsv.gz; data/figP05_length_curve.tsv.gz (GMSC n-weighted means) --> Unannotated phage ORFs are mostly antisense and frameshift shadows, whereas the GMSC negatives are conserved, in-frame sequences that look coding. Amino-acid entropy shows the same thing. On phage ORFs it separates the classes as well as 3Di entropy from 50 aa, and better below 50 aa, whereas on GMSC smORFs it is near or below chance. The phage result therefore shows how much of the separation depends on the negative set; it is not evidence that the method works better on phage.

These results set the method's lower length limit. Gene-length ORFs separate well: in phage genomes 3Di entropy alone reaches AUC 0.947. <!-- src: Validation/mutual_information/README.md §4.6 --> Below 100 aa, one entropy per alphabet carries only a small signal against conserved, coding-like negatives, and below 50 aa it carries little even against phage shadows. The GMSC negatives are hard: they are conserved and in frame, unlike the shadows and random ORFs that dominate genomic negatives. Even so, much of the weakness is probably length itself. A 3Di string of 50–99 symbols gives few observations for a 20-state distribution, and many real small proteins may be a single helix or disordered. Features that use the order of the 3Di string may do better, but we have not tested them.

**Limitations of structural entropy as a diagnostic**

Structural entropy struggles on short proteins. Below about 100 amino acids the 3Di entropy signal weakens: on curated small ORFs, all features together reach AUC 0.73, and 3Di entropy is completely uninformative below 30 amino acids. <!-- AGENT: cite Phold, folding the unfoldable etc here, this isn't news -->



Another limitation is that 3Di strings are predictions, and per-ORF values differ between hardware platforms by more than 0.3 bits for 1% of ORFs (up to 2.0 bits) . Population conclusions are unaffected, but per-ORF thresholds should be reported with the configuration that produced them. <!-- AGENT: the non-determinism should be a supp, not a mention in the main text -->

We also caution using  structural entropy for vanilla ORF calls in eukaryotic sequences in metagenomes. Given the pseudo ground truth in this analysis depends on Prodigal, which is designed for prokaryotes, it is likely 3Di entropy based analyses will behave differently on eukaryotic genomes as opposed to prokaryotes.






**The method has limits we have measured and limits we have not.**
- **Short ORFs.** Below about 100 amino acids the signal weakens: on curated small ORFs, all features together reach AUC 0.73, and 3Di entropy is uninformative below 30 amino acids. On phage ORFs, 3Di entropy separates genes only weakly below 50 amino acids (AUC 0.57). From 50 amino acids it reaches 0.71–0.85, mostly because unannotated phage ORFs are less protein-like than the GMSC negatives, and amino-acid entropy does as well there.
- **The 90-aa floor.** Several of our datasets have an effective 90-amino-acid floor from a unit error in the package (#103), which hides 28.7% of annotated phage genes. A re-run of the phage genomes down to 10 amino acids covers them, and agrees with the original run at 90 amino acids.
- **Predictions, not structures.** 3Di strings are predictions, and per-ORF values differ between hardware platforms by more than 0.3 bits for 1% of ORFs (up to 2.0 bits). Population conclusions are unaffected, but per-ORF thresholds should be reported with the configuration that produced them.
- **Labels.** Labels throughout are annotations or Prodigal calls, not ground truth.
- **Eukaryotic sequence in metagenomes.** Part of what a prokaryotic caller misses there is probably eukaryotic coding sequence, which neither Prodigal nor our controls were designed for.


# Data availability

**Code.** genome_entropy is available at https://github.com/linsalrob/genome_entropy (archived at Zenodo, doi:10.5281/zenodo.18312206). Each dataset's analysis code is in its directory under `Validation/` in that repository: `gtdb/`, `logan/`, `Inphared/`, `mutual_information/` and `smorfs/`.

**Figures.** Every figure and table is regenerated from compact data committed in `manuscript/data/`, by the notebooks in `manuscript/notebooks/`, in the environment in `manuscript/environment.yml`.

**Large datasets.** These will be deposited, with checksums, as the packages listed in `manuscript/roads_deposition_candidates.md`:
- per-ORF entropies for GTDB r232;
- the missed-gene search results;
- the length references;
- small-ORF entropies;
- LOGAN selection and analysis summaries.
<!-- Rob arranges the ROADS/figshare deposition; insert DOIs when issued. Do not claim DOIs before then. -->

**Public inputs.** These are available from their providers:
- GTDB r232 and the NCBI GenBank assemblies;
- LOGAN contigs (`s3://logan-pub`);
- Inphared phage genomes;
- the Global Microbial smORF Catalogue;
- the Foldseek databases.

# References

<!-- Rendered from references.bib by pandoc --citeproc. Entries marked "verify" in references.bib need checking before submission. -->

::: {#refs}
:::

# Supplementary figures

![**Figure S1.** The rise of 3Di entropy with ORF length is mostly biological; the rise of protein entropy is plug-in estimator bias. The mean entropy of all GTDB r232 ORFs matched to a deposited CDS at each length from 90 to 999 aa (solid), relative to its value at 100 aa, beside the expected rise for i.i.d. sequences of the same length and pooled composition (dashed), which is the bias of the plug-in estimator alone. Shading marks the part of the observed rise that the bias does not explain. **(a)** Bacteria, 301.6 M ORFs. **(b)** Archaea, 9.33 M ORFs. Between 100 and 300 aa the bias (+0.11 bits) exceeds the observed rise in protein entropy (+0.08 and +0.09 bits), but explains only 12% and 14% of the rise in 3Di entropy (+0.88 and +0.76 bits).](images/figG02_estimator_bias.png){#fig:s1 width=100%}<!-- AGENT: some of the labels are unclear on the left -->

![**Figure S2.** Only the empirical length-conditioned reference is calibrated. Matched ORFs from the systematic figure sample scored against two references at their own length: the empirical distribution of real matched ORFs, and i.i.d. multinomial draws at the pooled composition. **(a)** Bacteria, n = 1,028,059. **(b)** Archaea, n = 315,638. Left and centre: the distribution of z against each reference, with N(0, 1) dashed. The empirical z has mean ≈ 0 and SD 1.00 for protein and 3Di; the i.i.d. z has an SD of 3.9–6.6. Right: the share of ORFs in each percentile decile, where 10% per decile means a calibrated reference.](images/figGS02_reference_calibration.png){#fig:s2 width=100%}<!-- AGENT: some of the labels are unclear on the left -->

![**Figure S3.** LOGAN feature distributions at fixed ORF length. Distributions of amino-acid entropy, 3Di entropy, 12-state entropy and 3Di–12-state mutual information for complete ORFs of 100–149 aa (a–d) and 200–299 aa (e–h) in 50,009 LOGAN metagenomes. The classes are complete Prodigal genes (Prodigal+) and Prodigal− ORFs that are antisense shadows, same-strand frameshift shadows or intergenic. Densities come from exact histograms over all 89.99 billion ORFs (0.01-bit bins; no sampling); n per class is in the legends.](images/figL01_feature_distributions.png){#fig:s3 width=100%}

![**Figure S4.** LOGAN discrimination within ORF-length bins. ROC AUC of out-of-fold model scores within ORF-length bins, for complete Prodigal+ ORFs against all complete Prodigal− ORFs (a) and against intergenic ORFs (b). Within a bin, length carries little information, so the length-only model falls towards 0.5 (dotted) from 75 aa while models that include 3Di entropy stay high. Bins with fewer than 200 sampled positives or negatives are omitted. The models and cross-validation are those of Figure 2.](images/figL03_models_within_length.png){#fig:s4 width=100%}

![**Figure S5.** The complete-ORF model applied to contig-end fragments. The complete-ORF model applied unchanged to contig-end fragments: Prodigal partial genes against truncated Prodigal− ORFs, open at the 5′ or the 3′ end. **(a)** ROC AUC. **(b)** Average precision. Length alone is compared with length plus all four entropy features; metrics are population-weighted and out-of-fold.](images/figL04_fragments.png){#fig:s5 width=100%}

![**Figure S6.** Length-controlled separation by biome and tranche. Exact within-length-bin ROC AUC, from all-ORF histograms, for complete Prodigal+ against all complete Prodigal− ORFs. **(a)** By biome at 100–149 aa, ordered by 3Di entropy (n = Prodigal+ ORFs in the bin). **(b)** By tranche (pilot, phaseb1, phaseb2) at 100–149 and 150–199 aa. **(c)** 3Di entropy within each length bin, by biome, with ORF length within the bin shown for reference.](images/figLS1_biome_batch_separation.png){#fig:s6 width=100%}

![**Figure S7.** An independent gene caller agrees with the candidates more often than with their matched shadows. **(a)** The percentage of each GTDB arm whose ORF coincides with a Prodigal call (same stop codon and strand). The arms are annotated CDS (positive control), candidates, the matched comparator, and low-3Di intergenic ORFs. Prodigal consults neither GenBank nor any structure database. With the matched comparator as background, the implied Prodigal share is 36.4% (95% CI 36.2–36.6) in bacteria and 42.2% (41.1–43.2) in archaea. **(b)** Coincidence for the original, unfiltered high-3Di shadow arm by frame relation to the overlapping CDS (523,346 bacterial and 22,447 archaeal shadows). A same-frame shadow is part of the annotated protein, so Prodigal almost always calls it; this is why only antisense and frameshift shadows form the comparator.](images/figG05_prodigal.png){#fig:s7 width=100%}<!-- AGENT: v7 matched comparator makes no sense, please explain it in more detail -->

![**Figure S8.** Candidate burden per genome scales with genome size, not with assembly quality. The number of candidates in each annotated GTDB r232 representative. **(a)** Bacteria: 96,875 genomes, 523,346 candidates, median 3 per genome. **(b)** Archaea: 5,492 genomes, 22,447 candidates, median 2 per genome. Each covariate is cut into ≤ 20 equal-count bins; the line is the median per bin, the bands the 25–75% and 10–90% ranges, and the dashed line the mean. Spearman ρ over all genomes is printed above each panel: burden rises with genome size (ρ = 0.55 and 0.51), falls with the fraction of ORFs that match a deposited CDS (−0.43 and −0.38), and is nearly flat against contig N50 (0.10 and 0.14).](images/figGS03_candidate_burden.png){#fig:s8 width=100%}

![**Figure S9.** What the structurally supported candidates are. **(a)** Functional class of all candidates, of those meeting C3 and of those meeting C7 (Methods), from the product name of each candidate's best hit. In bacteria (n = 523,346, 90,658 and 23,540) metabolism is 10.5%, 20.4% and 33.8%, and mobile elements 3.3%, 7.3% and 13.6%. In archaea (n = 22,447, 5,141 and 1,574) metabolism is 14.3%, 24.5% and 41.3%. C7 is enriched for well-characterised folds, so it does not represent all supported candidates. **(b)** Ten worked examples in genomic context: the candidate in blue, deposited CDS in grey, arrows giving the strand. Labels give the best-hit product (example 3 by family, RecF/Rad50-family ABC ATPase), length, the number of databases with a full-length hit, and whether the candidate is structure-only, called by Prodigal, or meets C7. All ten meet C7 and are called by Prodigal; they illustrate the finding and are not a random sample. Labels are homology-based, not evidence of expression or function.](images/figG06_functions_examples.png){#fig:s9 width=100%}

![**Figure S10.** Metagenomic candidate evidence by length, evidence class and biome. **(a)** Full-length structural hit rate by ORF length for LOGAN candidates, complete Prodigal genes and the low-score intergenic and antisense controls, which lie together near zero. **(b)** Full-length evidence classes over all five databases: sequence and structure, structure only, sequence only, neither. **(c)** Full-length structural hit rate by biome for candidates and complete Prodigal genes. **(d)** Superkingdom of the best full-length target for candidates with a hit, by biome. These shares reflect the composition of the target databases, so they are compared between arms rather than read as absolute.](images/figL07_candidate_evidence_detail.png){#fig:s10 width=100%}

![**Figure S11.** Contig taxonomy of the Foldseek queries and the eukaryotic test. Kraken2 (2.17.2; PlusPFP 2023-06-05) classification of the 79,774 contigs that carry the Foldseek queries, re-streamed from LOGAN. **(a)** Contig class for each arm. **(b)** For candidates and complete Prodigal genes whose best full-length target is eukaryotic, the class of their contig: 69.8% of such candidates (n = 2,525) lie on eukaryotic contigs. PlusPFP contains a single animal genome (human), so animal contigs are reported as one group; 15.6% of contigs are unclassified.](images/figL08_contig_taxonomy.png){#fig:s11 width=100%}

![**Figure S12.** Why Prodigal did not call the metagenomic candidates. Prodigal 2.6.3 (`-p meta`) rerun on the query contigs with `-s`, which lists every potential gene and its scores; the rerun reproduces all original calls. For each query, the best-scoring potential gene ending at its stop is shown. **(a)** Coding score. **(b)** Total score (coding + start + RBS + upstream). **(c)** ORF GC minus the GC of the Prodigal model chosen for the contig; values outside each axis are clipped to its edges. **(d)** Why the ORF was not called: no potential gene at that stop; a negative coding score; a positive coding score made negative by start penalties; or a positive total that was not chosen. Bacterial-hit candidates are those whose best full-length target is bacterial.](images/figL09_prodigal_skip.png){#fig:s12 width=100%}

![**Figure S13.** Prodigal genes the model scores low. Complete Prodigal genes that score below the 0.80-precision threshold. **(a)** The share below the threshold by gene length, from the scan's 1-in-1,000 sample of complete Prodigal genes (880,405). **(b)** Full-length structural hit rate for 10,269 low-scoring genes and 8,769 high-scoring genes (≥ the 0.90 threshold) matched to them on length from 50 aa; bins with fewer than 100 genes are not drawn. **(c)** Full-length structural hit rate of every length-matched arm over five databases, including the frameshift-shadow control.](images/figL10_prodigal_lowscore.png){#fig:s13 width=100%}

![**Figure S14.** Phage ORFs show the same structural separation as cellular ORFs, with matched phage ORFs slightly lower in 3Di entropy at every length. **(a)** Mean 3Di entropy by ORF length from 90 to 999 aa, in 40 log-spaced length bins (n-weighted means). Solid lines are ORFs matched to an annotated CDS, dashed lines unmatched ORFs. Phage: 36,025 Inphared genomes, 6.75 M ORFs. Bacteria and archaea: every ORF in CDS-bearing GTDB r232 genomes. Matched phage ORFs sit below matched bacterial ORFs at every length, by 0.24, 0.34, 0.19 and 0.09 bits at 90–149, 150–299, 300–599 and 600–999 aa. **(b)** Matched minus unmatched mean 3Di entropy in four length ranges; the gap widens with length in phage and bacteria. **(c)** Protein against 3Di entropy for every matched ORF of each population, each panel normalised to its own total (phage 2.69 M, bacteria 309 M, archaea 9.47 M ORFs). The phage and GTDB runs used genome_entropy 0.3.0 and 0.2.0 with the same model revision and encoding size; cross-run differences in class means are about 0.01 bits (Table S1).](images/figP04_phage_vs_cellular.png){#fig:s14 width=100%}

![**Figure S15.** Phage: the minus-strand label error and the shortcut it taught the model. **(a)** Six-frame ORFs ≥ 90 aa in 36,025 Inphared genomes (6,754,361 ORFs) labelled as matching an annotated CDS by the legacy matcher and by the corrected matcher, by strand. The legacy matcher compared reverse-complement ORF coordinates with genomic CDS coordinates and labelled 3 of 3,311,070 minus-strand ORFs as genes; the corrected matcher labels 1,000,839. **(b)** Mean gain (five grouped cross-validation folds; whiskers, minimum to maximum) of each feature in the original 14-feature ranking model (31,200 genomes, 5,887,690 ORFs), trained on legacy or on corrected labels. On legacy labels the model learned that the plus strand means a gene (`strand_plus` carries 0.89 of the gain); on corrected labels 3Di entropy dominates and strand is irrelevant.](images/figPS01_phage_label_correction.png){#fig:s15 width=100%} <!-- AGENT: remove a and fix b, the readers wont care about our trial and errors. Just represent b with the corrected stuff -->

![**Figure S16.** On curated small ORFs (10–99 aa), predicted-structure entropy separates real from spurious sequences only weakly, but adds more than composition. 33,647,794 GMSC positive and 10,079,115 negative smORFs. **(a)** Protein length; positives are longer (median 80 against 70 aa). **(b)** 3Di entropy. Both classes share a mode below log₂3; positives have a second mode at 2.5–3.5 bits. Spikes at log₂2 and log₂3 are two- and three-state strings (Figure S18). **(c)** Amino-acid entropy; the classes almost overlap. **(d)** Single-feature ROC AUC within length bins. AUC < 0.5 means the feature is higher in negatives. **(e)** Gradient-boosted models within length bins, trained on 2 M sequences and tested on a disjoint 2 M. Solid lines use a random split; dashed lines hold out test families at 30% identity (MMseqs2).](images/figS01_smorf_separation.png){#fig:s16 width=100%} <!-- AGENT: as discussed, this should be a main figure. Also, the 16e labels are garbled -->

![**Figure S17.** Amino-acid against 3Di entropy for all 43,726,909 GMSC smORFs. Bins are 0.011 bits on each axis. **(a, b)** The fraction of each class per bin, on a log scale. **(c)** The positive minus the negative fraction per bin: blue bins are enriched in positives, red bins in negatives. Positives are enriched above about 2.3 bits of 3Di entropy when amino-acid entropy exceeds about 3.6 bits; negatives in a band at 1.0–1.6 bits of 3Di entropy centred near 4.0 bits of amino-acid entropy.](images/figSS01_smorf_density.png){#fig:s17 width=100%}

![**Figure S18.** The sharp 3Di-entropy stripes in smORFs come from the log₂k ceiling on strings that use only k states, not from a biological threshold. **(a)** At the 1.0-bit stripe, 67% of positive and 61% of negative strings use exactly two 3Di states, almost always D and P; the control band 0.03 bits above contains none. **(b)** At the 1.585-bit stripe, 42% of positive and 31% of negative strings use exactly three states, mostly D, P and V. **(c)** At log₂7 = 2.807 bits the density equals that of the flanking bands, and every string there uses 11–15 states.](images/figSS02_smorf_stripes.png){#fig:s18 width=100%}

![**Figure S19.** Below 50 aa, 3Di entropy separates short phage ORFs as weakly as it separates GMSC smORFs; from 50 aa it separates them almost as well as gene-length phage ORFs. ORFs of 10–90 aa from the 35,921 Inphared genomes whose genes use genetic code 11 (101.4 M ORFs, 654,673 matched to an annotated CDS), and the GMSC smORFs. **(a)** Single-feature ROC AUC within length bins for 3Di entropy (solid) and protein entropy (dotted). The 10–29 aa phage bin holds only 458 annotated ORFs. **(b)** Mean 3Di entropy by ORF length: the 10–90 aa phage run joined at 90 aa (dotted line) to the ≥ 90 aa run of Figure S14. Solid lines are matched or positive, dashed lines unmatched or negative. **(c)** 3Di entropy distributions in four length bins, each normalised within its class.](images/figP05_phage_short_orfs.png){#fig:s19 width=100%}
