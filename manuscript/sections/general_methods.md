## Overview of the genome_entropy framework

All entropy features were computed with genome_entropy (https://github.com/linsalrob/genome_entropy) [@genome_entropy]. The pipeline takes nucleotide sequences (GenBank or FASTA) and:
1. calls every open reading frame (ORF) in all six frames;
2. translates each ORF with the bacterial genetic code (NCBI table 11);
3. predicts a structural-alphabet representation of each translated ORF;
4. reports per-ORF entropy features.

When an input record carries CDS annotations, each ORF is also labelled by whether it matches one of them. Different datasets were processed with different package releases, encoding settings and hardware. The configuration for each is given in Table 1, and its effect on the values is quantified under *Reproducibility of the structural encoding*.

## ORF calling

ORFs are maximal stop-to-stop open reading frames reported by `get_orfs` [@get_orfs], so an ORF extends upstream of any start codon to the preceding in-frame stop. Each ORF records whether it is complete or runs off a contig end.

The minimum-length argument behaves differently by release. Before genome_entropy 0.4.0 it was passed to `get_orfs` in amino acids, although it was documented as nucleotides (genome_entropy #103, #104). The GTDB, phage and mutual-information datasets therefore have an effective floor of **90 amino acids**. The LOGAN analysis set the floor explicitly to 30 amino acids. The GMSC small ORFs are a curated 10–99 amino-acid set and were not ORF-called. <!-- src: methods_provenance.md §1; Validation/gtdb/scientific_report.md l.135–141; #103 -->

## Annotation labels

For datasets with deposited annotations (GTDB, phage), an ORF was labelled as matching a CDS when all of the following held:
- it lies on the same record and strand as the CDS;
- its biological start is in compatible codon phase with the CDS's, differing by a multiple of three, measured from the start on the plus strand and from the end on the minus strand;
- the two genomic intervals overlap by at least 90% of the shorter one;
- the coordinate-aligned, ungapped translations over the shared codons are at least 98% identical, with `X` treated as a wildcard.

The rule has no stop-codon requirement. A short CDS that lies wholly inside a longer stop-to-stop ORF can therefore match it. <!-- src: src/genome_entropy/io/genbank.py evaluate_orf_genbank_cds_match; MIN_CDS_OVERLAP_FRACTION=0.90, MIN_SHARED_AA_IDENTITY=0.98 -->

Earlier releases contained a matcher that compared reverse-complement ORF coordinates with genomic CDS coordinates. Under it, almost no minus-strand ORF could match. It was fixed in genome_entropy 0.2.0 (commit 2cc0e1a), and every label used here comes from the corrected matcher (Fig. S15). LOGAN has no deposited annotation, so its ORFs were labelled against Prodigal gene calls instead (*LOGAN metagenomes*).

A label records agreement with an annotation, not proof that an ORF is expressed or functional, nor that an unmatched ORF is not.

## Structural encoding

Translated ORFs were encoded with ModernProst (`gbouras13/modernprost-50M`) [@modernprost]. This multitask protein language model predicts two per-residue structural alphabets directly from the amino-acid sequence, with no structure or alignment as input:
- the 20-state Foldseek 3Di alphabet [@vanKempen2024];
- a 12-state alphabet.

The model revision is `2d68d506d4ce8d59025c985fd00243a8860363c6` for every dataset where it was recorded (Table 1). Sequences were encoded in batches of up to a fixed number of residues, the *encoding size*. Because ModernProst predicts structural states from sequence alone, it assigns confident 3Di states to any protein-like sequence. A 3Di string is therefore a prediction, not an observation of structure.

## Entropy features

For each ORF, Shannon entropy was computed in bits from the observed symbol frequencies of the sequence in a given alphabet:

$$H(X) = -\sum_{i} p_i \log_2 p_i ,$$

where $p_i$ is the fraction of positions in the ORF that carry symbol $i$. This was done separately for the nucleotide sequence (4 symbols), the amino-acid sequence (20), the 3Di string (20) and the 12-state string (12). The maxima are therefore 2, $\log_2 20 \approx 4.32$, $\log_2 20$ and $\log_2 12 \approx 3.58$ bits.

The per-ORF mutual information between the aligned 3Di and 12-state strings was computed from their joint frequencies:

$$I(\mathrm{3Di};\,\mathrm{12st}) = \sum_{a,b} p(a,b)\,\log_2 \frac{p(a,b)}{p(a)\,p(b)} .$$
<!-- src: src/genome_entropy/entropy/shannon.py shannon_entropy, mutual_information -->

Two properties of these estimators shape the analyses below.
- **Length bias.** The plug-in estimator is biased downwards by an amount that shrinks with sequence length. Part of any rise of entropy with ORF length is therefore estimation bias, not biology. We quantify this against an i.i.d. reference and an empirical length-conditioned reference (Fig. S1, S2), and control for length in every comparison.
- **The log₂k ceiling.** A string that uses only $k$ symbols has entropy of at most $\log_2 k$ bits. An ORF whose predicted 3Di uses two or three states therefore has 3Di entropy of at most 1 or $\log_2 3 \approx 1.585$ bits, whatever its length.

## Statistics

The statistics shared across datasets are:
- **Discrimination:** ROC AUC and average precision (AP).
- **Proportions:** reported with Wilson or first-order (delta-method) 95% confidence intervals.
- **Correlations:** Spearman's ρ.

Dataset-specific estimands are described under each dataset:
- the mixture estimate of real-gene share and its comparator (GTDB);
- grouped cross-validation (LOGAN, phage);
- binned mutual information with permutation nulls (phage).

Analyses used Python 3.12 with NumPy, pandas, SciPy, scikit-learn and XGBoost. The exact versions for each dataset are recorded in its analysis directory.

## Figures and data availability

Every figure and supplementary figure is drawn by a Jupyter notebook in `manuscript/notebooks/`, from compact plotting data committed in `manuscript/data/`. `manuscript/figures_manifest.tsv` maps every panel to:
- its notebook;
- its plotting data;
- its upstream source;
- its producing code.

Running `manuscript/tools/run_notebooks.py` in the pinned environment (`manuscript/environment.yml`, `environment.lock.txt`) regenerates every committed image byte for byte. Datasets too large for the repository are listed, with checksums, in `manuscript/roads_deposition_candidates.md` and `manuscript/roads/`.

**Table 1. Datasets and encoding configurations.**

| dataset | genomes / samples | ORFs analysed | ORF floor | labels | genome_entropy | encoding size | hardware |
|---|---|---:|---|---|---|---:|---|
| GTDB r232 bacteria | 189,715 species representatives (of 189,801) | 2,568,244,984 | 90 aa | deposited GenBank CDS | 0.2.0 (inferred; commit 90c8c15) | 10,000 | NVIDIA V100, NCI Gadi |
| GTDB r232 archaea | 10,122 species representatives | 54,858,398 | 90 aa | deposited GenBank CDS | 0.2.0 (inferred) | 10,000 | NVIDIA V100, NCI Gadi |
| LOGAN metagenomes | 50,009 assembled metagenomes, 13 biomes | 89,991,549,208 | 30 aa | Prodigal 2.6.3 `-p meta` | 0.3.0 (commit 750e2b92) | 10,000 | AMD MI250X, Pawsey Setonix |
| Inphared phage, mutual information | 36,025 genomes | 6,754,361 | 90 aa | Pharokka + phold CDS (corrected matcher) | ≈0.2.0 | 20,000 | AMD MI250X (ROCm, fp16), Setonix |
| Inphared phage, ORF ranking | 31,200 genomes | 5,887,690 | 90 aa | as above | ORFs from 0.1.15, relabelled | 20,000 | Setonix |
| Inphared phage, comparison with bacteria | 36,025 genomes | 6,754,361 | 90 aa | as above | 0.3.0 (commit 45dd283) | 10,000 | NVIDIA V100, Gadi |
| GMSC small ORFs | — | 43,726,909 (10–99 aa) | — | GMSC curated positive/negative | 0.3.0 | 50,000 | NVIDIA V100, Gadi |

ModernProst revision `2d68d506…` is recorded for GTDB (recovered from the offline model cache), LOGAN (every chunk manifest) and the small ORFs. It is not recorded for the Setonix phage runs. <!-- src: methods_provenance.md §2; agent-handoffs#24 (LOGAN agent, Gadi provenance comments); Validation/mutual_information/README.md §2; Validation/Inphared/README.md -->
