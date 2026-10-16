<!-- Draft Methods text, Setonix section drafter, agent-handoffs#24 checkpoint D. The integrator merges it into genome_entropy.md. -->
<!-- Sources: Validation/mutual_information/README.md (MI); Validation/Inphared/README.md (IR); AH inphared_gadi/README.md (IG); methods_provenance.md; decisions D5, D6, D7, D12. -->

## Phage genomes

### Genomes, annotation and ORFs

We used 36,025 complete phage genomes from Inphared [@Cook2021] (2.316 Gb in total; 1,761–735,411 bp). <!-- src: AH inphared_gadi/README.md §1 table; Validation/mutual_information/README.md §2 table --> The release is most likely that of 7 April 2026, which we inferred from file dates; record-level identity with that release was not checked. <!-- src: Validation/mutual_information/README.md §2 table ("Phage release", UNCONFIRMED) --> Each genome had been annotated with Pharokka [@Bouras2023pharokka] and then phold [@Bouras2025phold], whose versions were not recorded. Gene calls in these annotations came from Pyrodigal-gv 0.3.2 [@Larralde2022], which gave 3,336,744 CDS. <!-- src: Validation/mutual_information/README.md §2 table ("Gene calls in the reference"); methods_provenance.md §4 item 1 --> The re-annotated CDS were the labels for every phage analysis. The submitters' own annotations were used only for the OM869708 example.

genome_entropy found six-frame stop-to-stop ORFs. The minimum length was passed to `get_orfs` in amino acids, not nucleotides, so the effective floor was 90 aa rather than the intended 30 aa (issue #103). <!-- src: Validation/mutual_information/README.md §2 table ("ORF calling"); results_inventory.md §5 --> That gave 6,754,361 ORFs, of which 17,322 run off a contig end. <!-- src: Validation/mutual_information/README.md §2 table ("ORFs") --> 957,269 of the 3,336,744 annotated CDS (28.7%) are shorter than 90 aa and cannot appear as an ORF of their own (D6). <!-- src: Validation/mutual_information/README.md §2.1 ("Unrepresentable CDS"); IR:75 gives 28.8% for the 31,200-genome ranking subset -->

### Gene labels

An ORF was labelled a gene when it matched an annotated CDS under all of the following conditions:

- the CDS is on the same record and strand;
- the codon phase is compatible;
- the genomic overlap covers at least 90% of the shorter interval;
- at least 98% of the coordinate-aligned amino acids are identical, with `X` as a wildcard.

<!-- src: methods_provenance.md §1 (`in_genbank` label); src/genome_entropy/io/genbank.py -->

There is no stop-codon check, so a short CDS inside a longer stop-to-stop ORF labels that ORF. In 77,799 positives the CDS covers less than half of the ORF. <!-- src: Validation/mutual_information/README.md §2.1 ("Short CDS inside long ORFs") -->

The matcher used before genome_entropy commit 2cc0e1a compared reverse-complement ORF coordinates with genomic CDS coordinates. It therefore labelled almost no minus-strand ORF as a gene. We relabelled every ORF with the corrected matcher (Fig. S15a). All phage results use these corrected labels.

### Two ORF sets (D5)

Two Setonix encodings of the phage ORFs were used.

- **Mutual-information set.** All 36,025 genomes and 6,754,361 ORFs were encoded with `gbouras13/modernprost-50M` [@modernprost] at an encoding size of 20,000 residues, on AMD GPUs (ROCm, fp16). <!-- src: Validation/mutual_information/README.md §2 table ("Re-encoding") --> The package version is genome_entropy 0.2.0, inferred from the JSON schema version, and the model revision was not recorded. <!-- src: Validation/mutual_information/README.md §2 table ("Package version"); methods_provenance.md §1 --> The corrected labels give 2,685,382 genes (39.8%). <!-- src: Validation/mutual_information/README.md TL;DR; IG:14 (39.76%) -->
- **Ranking set.** The ranking used an earlier encoding (ORFs from genome_entropy 0.1.15; same model and encoding size). <!-- src: Validation/Inphared/README.md §2 table (lines 74, 76) --> 49 of the 361 annotation parts had no encoder output, every fifth part from 121 to 361, which points to a failed worker rather than to biology. <!-- src: Validation/Inphared/README.md:71 --> The set therefore covers 31,200 genomes and 5,887,690 ORFs. <!-- src: Validation/Inphared/README.md:72, :74 --> With corrected labels it has 2,338,039 genes (39.7%). <!-- src: Validation/Inphared/README.md:102 --> The two independent relabellings agree on 5,887,451 of these 5,887,690 ORFs. <!-- src: Validation/Inphared/README.md:104 -->

A third encoding of all 36,025 genomes on Gadi was used only for the comparison with cellular genomes and for the determinism test (see below).

### Mutual information

For ORF length, the four entropies and the 3Di–12-state mutual information, we estimated binned plug-in mutual information with the label, with 20 empirical-quantile bins per axis and the 2-D product grid for pairs. <!-- src: Validation/mutual_information/README.md §3 item 4 --> Estimates were made on a class-balanced sample of 200,000 ORFs per class, where H(label) = 1 bit, and on the full population. <!-- src: Validation/mutual_information/README.md §3 item 4 -->

- **Uncertainty:** 100 bootstrap resamples, giving basic 95% intervals. <!-- src: Validation/mutual_information/README.md §3 item 4 -->
- **Null:** 200 label permutations, at 10, 20 and 40 bins. <!-- src: Validation/mutual_information/README.md §3 item 4 -->
- **Length control:** I(X; label | L), with L in 20 length-quantile strata and X re-binned within each stratum, plus single-feature ROC AUC within length bins. <!-- src: Validation/mutual_information/README.md §3 item 4 -->

The random seed was 20261014 throughout. <!-- src: Validation/mutual_information/README.md §3 item 4 -->

### Ranking candidate genes

Coding probabilities came from an XGBoost random forest (`XGBRFClassifier`; 500 trees, depth 8, row and per-node column subsampling 0.8) [@Chen2016]. <!-- src: Validation/Inphared/README.md:78; scripts/rank_missing_orfs.py make_model --> We used out-of-fold predictions from five-fold cross-validation grouped by genome. <!-- src: Validation/Inphared/README.md:78 --> The features were intrinsic to each ORF: its four entropies and the lengths of the corresponding sequences, eight in all. <!-- src: Validation/Inphared/README.md:17, Table 1 line 130 --> Position, strand, frame and codon flags were excluded (Fig. S15b; Results).

Candidate missed genes were unannotated ORFs with p ≥ 0.8. <!-- src: Validation/Inphared/README.md:210; scripts/rank_missing_orfs.py defaults --> The forest's probabilities span about 0.11–0.89, so this threshold lies near the top of the scale and is not a calibrated 80%. <!-- src: Validation/Inphared/README.md:140 --> Replacement candidates were unannotated ORFs with p ≥ 0.8 that overlap an annotated ORF with p ≤ 0.5, by a margin of at least 0.4. <!-- src: scripts/rank_missing_orfs.py defaults (--minimum-probability 0.80, --maximum-annotated-probability 0.50, --minimum-probability-difference 0.40) --> The competitor was the overlapping annotated ORF with the greatest overlap. Pairs were ranked by 0.50 p + 0.35 Δp + 0.15 × the overlap fraction of the shorter ORF. <!-- src: scripts/rank_missing_orfs.py replacement_rank_score -->

### Structural validation

We searched 20,000 ORFs per arm with Foldseek 10 (10.941cd33) [@vanKempen2024]: <!-- src: Validation/Inphared/README.md §5.1 -->

- candidates: a random sample of the intergenic, antisense and other-frame candidates;
- annotated genes, as the positive control;
- low-scoring (p < 0.5) antisense, other-frame and intergenic ORFs, as negative controls;
- the candidate arm with each 3Di string shuffled, as a null.

Controls were matched to the candidate arm on length without replacement. <!-- src: Validation/Inphared/README.md:226 --> The replacement pairs were searched in full.

Queries combined the amino-acid sequence with the predicted 3Di string. We ran structure search with `--alignment-type 2 -e 1e-3` and sequence-only search at the same E-value. <!-- src: Validation/Inphared/README.md:240–243 --> The databases were PDB100 (2025-01-01) [@Berman2000], AFDB-SwissProt (v6, 2025_03) [@Varadi2024], CATH50 (v4.3.0) [@Sillitoe2021] and the viral BFVD (2025_03_v3) [@Kim2025bfvd]. <!-- src: Validation/Inphared/README.md:247–252 --> A hit was full-length when query and target coverage were both ≥0.8. <!-- src: Validation/Inphared/README.md:254 --> We give Wilson 95% intervals. The share of candidates that behave like annotated genes was estimated as (candidate − control) / (annotated − control). <!-- src: Validation/Inphared/README.md:254; scripts/07_evidence.py header -->

### Comparison with cellular genomes

For Fig. S14 we used the Gadi encoding of all 36,025 genomes: genome_entropy 0.3.0, encoding size 10,000, NVIDIA V100. <!-- src: AH inphared_gadi/README.md §1 table --> Its labels equal the corrected labels ORF for ORF. <!-- src: AH inphared_gadi/README.md TL;DR, §4 --> Bacterial and archaeal ORFs came from GTDB r232 genomes that carry annotation, encoded with genome_entropy 0.2.0 at the same encoding size and summarised by length (#101). <!-- src: notebooks/figP04_phage_vs_cellular.ipynb provenance cell --> Mean 3Di entropy of matched and unmatched ORFs was compared by length from 90 to 999 aa.

### OM869708

We took the submitters' GenBank record of OM869708 from NCBI (retrieved 2026-10-16). <!-- src: notebooks/figP01_phage.ipynb provenance, upstream (d) --> We used BLASTP 2.12.0+ [@Camacho2009] to search the stop-to-stop translations of ORF3 and ORF7 against a local copy of NCBI nr (snapshot of 1 August 2026), with E ≤ 10⁻⁵ and up to 50 target sequences. <!-- src: work/om869708/blastp_nr.slurm (Setonix job 50987603); methods_provenance.md §3 Phage -->
