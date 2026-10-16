<!-- Draft Methods and Results text, Setonix section drafter, agent-handoffs#24 checkpoint D. The integrator merges it into genome_entropy.md. -->
<!-- Sources: Validation/smorfs/README.md (SR); AH smorfs/manuscript.md and tables/; manuscript/data/figS01_*, figSS02_*, tabS01–04; sections/reproducibility_methods.md (Gadi); AH inphared_gadi/README.md §5; data/tabG01_encoding_determinism.tsv.gz; decisions D3, D9, D11. -->

## Methods

### Small ORFs from the Global Microbial smORF Catalogue

To find the method's lower length limit, we used the curated benchmark sets of the Global Microbial smORF Catalogue (GMSC) [@Duan2024], as supplied by its authors.

- **Positives:** 33,647,794 high-quality smORFs supported by RNAcode coding potential [@Washietl2011] together with metatranscriptomic or metaproteomic evidence, or by homology to known small proteins. <!-- src: Validation/smorfs/README.md:4, :23; AH smorfs/manuscript.md §2.1 -->
- **Negatives:** 10,079,115 conserved, in-frame GMSC sequences from clusters that RNAcode called non-coding. Sequences with conserved-domain hits or homology to the positives were removed. <!-- src: Validation/smorfs/README.md:4, :23; AH smorfs/manuscript.md §2.1 -->

The negatives are therefore not random ORFs. They are assembled, conserved, in-frame sequences that failed tests for coding, and they look protein-like by construction.

Proteins are 10–99 aa long: medians 80 aa (positives) and 70 aa (negatives). No 90-aa floor applies (Table S13). <!-- src: Validation/smorfs/README.md:24–28; data/tabS04_dataset_summary.tsv.gz -->

ORFs were translated with genetic code 11, the terminal stop was removed, and internal stops were kept. 1.71% of positives and 1.58% of negatives contain an internal stop. <!-- src: AH smorfs/manuscript.md §2.2 ("Translation") -->

Encoding used genome_entropy 0.3.0 and `gbouras13/modernprost-50M` (revision 2d68d506) at an encoding size of 50,000 residues, on NVIDIA V100 GPUs (Gadi). <!-- src: Validation/smorfs/README.md:34–40; notebooks/figS01_smorf_separation.ipynb provenance --> We computed the nucleotide, amino-acid, 3Di and 12-state entropies of each sequence.

We measured separation by ROC AUC, over all sequences and within seven length bins (10–29, 30–49, then 10-aa bins to 99 aa; Table S10). <!-- src: AH smorfs/manuscript.md §2.3; data/tabS01_length_bins.tsv.gz --> Gradient-boosted classifiers (scikit-learn `HistGradientBoostingClassifier`, early stopping, seed 42) [@Pedregosa2011] were trained on nine nested feature sets. <!-- src: AH smorfs/manuscript.md §2.3 ("Models") --> Each was trained on 2 million sequences and tested on a disjoint 2 million. <!-- src: Validation/smorfs/README.md:83 -->

To control for homology, all 43,726,909 proteins were clustered together with MMseqs2 14.7e284 [@Steinegger2017], at 50% identity (`linclust`) and at 30% identity (`cluster`), both at 80% coverage (Table S11). <!-- src: AH smorfs/manuscript.md §2.3 ("Homology clusters"); data/tabS02_cluster_stats.tsv.gz --> We repeated the models with train and test sets drawn from disjoint halves of the clusters. We also report a cluster-weighted AUC in which each test sequence has weight 1/(size of its 50%-identity cluster).

To explain sharp stripes in the 3Di-entropy distribution, we took the 3Di strings within ±0.0055 bits of log₂k (k = 2, 3, 7). <!-- src: notebooks/figSS02_smorf_stripes.ipynb provenance ("Bands") --> We compared them with control bands of the same width 0.03 bits below and above. <!-- src: notebooks/figSS02_smorf_stripes.ipynb provenance ("Bands") -->

### Reproducibility of the structural encoding

The datasets in this paper were encoded with different package versions, encoding sizes and hardware (Table 1; D9). To separate run-to-run variation from the effects of configuration, we re-encoded the first 50 genomes of the Gadi phage encoding (6,284 ORFs). <!-- src: AH inphared_gadi/README.md §5; data/tabG01_encoding_determinism.tsv.gz (n_orfs) --> We used genome_entropy 0.3.0 and `gbouras13/modernprost-50M` on one NVIDIA V100, at encoding sizes of 10,000 and 20,000 residues (PBS job 181181761). <!-- src: AH inphared_gadi/README.md §5; sections/reproducibility_methods.md -->

We compared these re-encodings with two earlier encodings (Table S1):

- the production run they repeat (V100, encoding size 10,000);
- an independent encoding of the same ORFs on AMD GPUs on Pawsey Setonix (ROCm, fp16, genome_entropy 0.2.0, encoding size 20,000). <!-- src: data/tabG01_encoding_determinism.tsv.gz (run_a, run_b) -->

The comparisons gave three results.

- **Repeat run.** Repeating the production configuration reproduced every 3Di entropy, 12-state entropy and 3Di–12-state mutual information value bit for bit. <!-- src: data/tabG01_encoding_determinism.tsv.gz rows "det es10k vs production" (100%) -->
- **Encoding size.** Changing the encoding size changed 0.3–0.7% of ORFs, by at most 0.076 bits. <!-- src: data/tabG01_encoding_determinism.tsv.gz rows "det es20k vs production" (99.68/99.60/99.35% identical; max 0.0758) -->
- **Hardware, precision and package version.** Between the two systems only 37.8% of 3Di entropies were bit-identical. <!-- src: data/tabG01_encoding_determinism.tsv.gz row "det es10k vs Setonix", three_di_entropy --> The median absolute difference was 0.007 bits, the 99th percentile 0.31 bits and the maximum 0.92 bits (Pearson r = 0.997). <!-- src: data/tabG01_encoding_determinism.tsv.gz (0.0066, 0.310, 0.920, 0.99719) --> Matching the encoding size left this spread unchanged. <!-- src: data/tabG01_encoding_determinism.tsv.gz row "det es20k vs Setonix" (37.7%, 0.310, 0.920) -->

We could not re-run the Setonix side in the same configuration. The design therefore cannot apportion the residual between hardware, numerical precision and package version. The 50 genomes reproduce the whole-population agreement between the two systems (99th percentile 0.310 against 0.317 bits; r = 0.99719 against 0.99698). <!-- src: AH inphared_gadi/README.md §5 point 3 -->

Population-level quantities are unaffected: class means move by about 0.01 bits, against a 1.46-bit difference between classes. <!-- src: AH inphared_gadi/README.md §5 ("Consequences") --> Per-ORF entropies are not reproducible bit for bit across platforms. For 99% of ORFs they agree to within 0.31 bits (3Di) and 0.35 bits (12-state), but single ORFs differ by up to 0.92 and 1.05 bits. <!-- src: data/tabG01_encoding_determinism.tsv.gz (p99 0.310 / 0.348; max 0.920 / 1.050) --> An ORF near a decision threshold can therefore change class between platforms. We report per-ORF thresholds and candidate rankings with the hardware, numerical precision, encoding size and software version that produced them.

## Results

### Short ORFs mark the lower limit of the method

On the GMSC benchmark, structural entropy separates real from spurious small ORFs only weakly (Fig. S16). Pooled over all 43.7 million smORFs, length is the best single feature (AUC 0.640). 3Di entropy (0.605) and 12-state entropy (0.602) come next. <!-- src: Validation/smorfs/README.md:74–80; data/figS01_single_feature_auc.tsv.gz --> Nucleotide and amino-acid entropy barely separate the classes (0.504 and 0.525). <!-- src: data/figS01_single_feature_auc.tsv.gz (0.5037, 0.5254) --> Positives are longer (median 80 against 70 aa; Table S13). <!-- src: data/tabS04_dataset_summary.tsv.gz -->

At fixed length, 3Di entropy keeps a small signal that grows with length: AUC rises from 0.526 at 50–59 aa to 0.626 at 90–99 aa (Fig. S16d). <!-- src: AH smorfs/manuscript.md Table 2; data/figS01_single_feature_auc.tsv.gz --> Amino-acid entropy falls to or below chance in the same bins (0.46–0.50). <!-- src: AH smorfs/manuscript.md Table 2 --> At 10–29 aa, 3Di entropy is uninformative (AUC 0.47). This bin holds only 11,382 positives against 80,578 negatives (Table S10). <!-- src: data/figS01_single_feature_auc.tsv.gz (0.4727); data/tabS01_length_bins.tsv.gz -->

Combining features helps, but the ceiling is low (Fig. S16e):

- length plus nucleotide and amino-acid entropy reaches held-out AUC 0.669;
- length plus 3Di and 12-state entropy reaches 0.699;
- all five features reach 0.730.

<!-- src: AH smorfs/manuscript.md Table 3; data/figS01_models_by_split.tsv.gz (0.6692, 0.6993, 0.7300) -->

Holding out homology clusters barely changes this: 0.728 at 50% identity and 0.721 at 30%. <!-- src: Validation/smorfs/README.md:86–90 --> Weighting each 50%-identity cluster once lowers the all-feature AUC to 0.695 under the 30% split. <!-- src: Validation/smorfs/README.md:90 --> The positives are far more redundant than the negatives: 14.5 against 6.0 sequences per 50%-identity cluster, and the 20 largest clusters at each threshold are all positive (Tables S11, S12). <!-- src: data/tabS02_cluster_stats.tsv.gz; data/tabS03_cluster_largest.tsv.gz; AH smorfs/manuscript.md §3.7 --> Cluster weighting lowers 3Di AUC more than the other features (0.612 to 0.571, random split). Part of the structural signal therefore sits in large positive families. <!-- src: data/figS01_models_by_split.tsv.gz (3Di, random: 0.6122 / 0.5710) -->

The amino-acid × 3Di entropy plane shows where the classes differ (Fig. S17). Positives are enriched above about 2.3 bits of 3Di entropy when amino-acid entropy exceeds about 3.6 bits. Negatives are enriched in a band at 1.0–1.6 bits of 3Di entropy. <!-- src: notebooks/figSS01_smorf_density.ipynb caption --> Sharp stripes at 1.0 and 1.585 bits are strings that use only two or three 3Di states (Fig. S18). In the 1.0-bit stripe, 67% of positive and 61% of negative strings use exactly two states, almost always D and P. <!-- src: data/figSS02_stripe_bands.tsv.gz (67.43 / 60.68%; DP) --> In the 1.585-bit stripe, 42% and 31% use exactly three, mostly D, P and V. <!-- src: data/figSS02_stripe_bands.tsv.gz (41.81 / 30.74%; DPV) --> Such strings cannot exceed log₂2 or log₂3 bits and accumulate just below those values. Both classes contain them at similar rates, so low 3Di entropy alone does not mark a small ORF as non-coding.

A short line near 2.81 bits contains only positives. Two families of near-identical 3Di strings appear to form it, which we identified by eye as ribosomal proteins L34 and S14 (D11). <!-- src: AH smorfs/manuscript.md §3.6 --> We did not confirm these identities by homology search, and we do not interpret the stripes biologically.

These results set the method's lower length limit. Gene-length ORFs separate well: in phage genomes 3Di entropy alone reaches AUC 0.947. <!-- src: Validation/mutual_information/README.md §4.6 --> Below 100 aa, one entropy per alphabet carries only a small signal. The GMSC negatives are hard: they are conserved and in frame, unlike the shadows and random ORFs that dominate genomic negatives. Even so, much of the weakness is probably length itself. A 3Di string of 50–99 symbols gives few observations for a 20-state distribution, and many real small proteins are a single helix or disordered. ModernProst has not been validated against experimental or predicted structures below about 50 aa. Features that use the order of the 3Di string may do better, but we have not tested them.
