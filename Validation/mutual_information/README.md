# Phage ORFs: how much each entropy variable tells us about annotation status

Analysis date: 2026-10-14. Label: an ORF is positive if it matches a Pharokka/phold CDS in the Inphared phage genomes. This is an analysis record, not paper text.

## TL;DR

- **The archived `in_genbank` label behind the earlier result (0.383 bits) is wrong on the minus strand.** Only 3 of the 3,311,070 minus-strand ORFs are labelled positive. The labels came from a legacy PHOLD JSON written by the pre-2026-07-31 matcher, which compared reverse-complement ORF coordinates with genomic CDS coordinates. As a result, about 1.0 M real minus-strand phold genes sat in the "negative" class.
- **Every ORF was relabelled** with the current matcher (`genome_entropy.io.genbank`). The new labels give 2,685,382 positives out of 6,754,361 ORFs. They contain 1,000,839 minus-strand positives, against 3 before. All results below use these corrected labels (`v2`). For reference, the legacy-label tables are kept in `tables/legacy_label/`.
- **3Di entropy is the strongest single variable.**
  - I(3Di; label) = **0.587 bits** on a class-balanced sample (H = 1 bit).
  - On the full population it is 0.572 bits, which is 59% of H(label) = 0.970 bits.
  - Its single-feature ROC AUC is **0.947** (AP 0.936).
- **The best pair is again protein + 3Di entropy, at 0.644 bits.** It adds only **+0.057 bits** (bootstrap 95% CI 0.055–0.058) over 3Di alone, and every other 3Di pair adds ≤ 0.026 bits. 3Di entropy therefore carries most of the information, and the pair ranking in the original stub mainly reflected which partner adds a little extra.
- **The result is not driven by length.**
  - Length alone is weak here: I = 0.125 bits, AUC 0.733. The reason is that every ORF is ≥ 90 aa (see Provenance).
  - Conditioning on length (20 quantile strata) keeps 82% of the 3Di information: I(3Di; label | length) = 0.472 bits.
  - Within every length bin, 3Di entropy has AUC 0.874–0.991, while length has 0.53–0.63.
- **The binned MI estimates are far above chance.**
  - The 99th percentile of the permutation null is ≤ 0.0009 bits (20 bins), against signals of 0.13–0.64 bits.
  - The ranking is stable across 10, 20 and 40 bins per axis; every value moves by < 0.011 bits.

## 1. Objective

The question is how much each per-ORF variable tells us about whether a six-frame ORF is an annotated phage gene, and whether pairs of variables add information beyond the best single one. The variables are the DNA, protein, 3Di and 12-state Shannon entropies, the per-ORF 3Di–12-state mutual information, and ORF length. The analysis also checks that the signal survives control for length and is clearly above the estimator's null, and makes the numbers comparable with the GTDB, LOGAN and smORF analyses.

## 2. Data and provenance

| item | value | evidence |
|---|---|---|
| Genomes | 36,025 GenBank records in `phold.gbk` (5.47 GB; `.gz` sha256 `8399e68c…3219c`) | `tables/phold_gbk_provenance.txt` |
| Phage release | **Most likely Inphared `7Apr2026_genomes`**, annotated by Pharokka and then phold in 361 parts (`Validation/Inphared/Pharokka/7Apr2026_genomes_pharokka_phold_out/` on the remote; 361 parts of ≤ 100 genomes is consistent with 36,025 records) | The LOCUS dates in `phold.gbk` (26-MAY: 13,500; 27-MAY: 22,425; 28-MAY: 100) match the part outputs (for example part_001 `phold.gbk` is dated 27-MAY-2026). `GenomesDB_Jan_2026.tar.gz` is **not** the source. **UNCONFIRMED:** the concatenation order differs, so record-level identity with the parts was not checked. |
| Gene calls in the reference | Pyrodigal-gv 0.3.2 (3,336,744 CDS), tRNAscan-SE 2.0.12, Aragorn 1.2.41, MinCED 0.4.2. Function annotation by phold. | `/inference` qualifiers |
| phold / Pharokka versions | **Not recorded.** Neither the part outputs nor `phold.gbk` contain a main log. | — |
| Re-encoding | `genome_entropy run` on FASTA converted from 145 chunks of 250 records (`slurm/rocm/run_mutual_information_chunks.slurm` → `rerun_mutual_information.slurm`): `--model gbouras13/modernprost-50M --encoding-size 20000 --multi-gpu`, on ROCm GPUs (`inference_device = cuda`, fp16). The model revision was not pinned. | Repository Slurm templates; per-feature `model_name` in the JSON |
| Package version | **genome_entropy 0.2.0 (inferred).** The JSON schema is 2.2.0, which was first added on 2026-08-17 (commit 1e40e40) and released in v0.2.0 the same day; the venv was installed from PyPI, and the outputs are dated 2026-08-17. | `schema_version` in all 145 JSON chunks |
| ORF calling | Default `--min-aa 30` → `get_orfs -l 90`, so the **effective minimum is 90 aa** (issue #103). The shortest protein is exactly 90 aa in both classes. Re-running `get_orfs -l 90` with the current code reproduces the archived ORF set exactly (6,754,361/6,754,361; 0 extra, 0 missing). | `tables/dataset_summary.json`, `tables/label_check.tsv`, `tables/relabel_chunk_summary.tsv` |
| ORFs | 6,754,361 six-frame stop-to-stop ORFs. 18,322 of them run off a contig end. | — |
| Archived label | `in_genbank` in `phold_modernprost_50m_mutual_information.tsv` (sha256 `e7aff2e7…9028`). It was copied by `aggregate_mutual_information.slurm --labels LEGACY.json.gz` from an older PHOLD JSON, keyed on (input_id, start, end, strand, frame). The exact legacy file is UNCONFIRMED; it is most likely `Validation/Inphared/phold.json.gz` on the remote. | `scripts/extract_structural_entropy_tsv.py` |

The JSON and TSV values are identical (maximum absolute difference 9e-16). The (input_id, orf_id) join is 1:1, with no rows missing on either side.

### 2.1 The archived label is broken on the minus strand

The archived label disagrees with the current matcher in a systematic way (`tables/label_legacy_vs_v2.tsv`):

| strand | contig edge | legacy + / v2 + | legacy − / v2 + | legacy + / v2 − | both − |
|---|---|---|---|---|---|
| + | no | 1,674,541 | 5,127 | 3,626 | 1,750,428 |
| + | yes | 0 | 4,875 | 0 | 4,694 |
| − | no | 3 | **1,000,836** | 0 | 2,301,478 |
| − | yes | 0 | 0 | 0 | 8,753 |

**Cause.** The matcher in use before commit 2cc0e1a (2026-07-31) matched on the stop coordinate plus the C-terminal sequence. It compared `orf.start` with the genomic CDS start on the minus strand, but `get_orfs` reports minus-strand coordinates on the reverse complement, so minus-strand ORFs (almost) never matched. It also required a stop, so plus-strand ORFs that run off a contig end never matched. In effect, the legacy label is "plus-strand phold gene".

**Consequence for the original analysis.** About a third of the real genes (1.0 M) sat in the negative class, which deflated every MI value. On the same data with the legacy label, the best pair is again protein + 3Di at 0.382 bits, which reproduces the stub's 0.383 bits. With the corrected label it is 0.644 bits.

**Current label rule (`v2`).** An ORF is positive if there is a CDS on the same record with the same strand and codon phase, a genomic overlap ≥ 90% of the *shorter* interval, and ≥ 98% coordinate-aligned amino-acid identity (X is a wildcard). Under this rule:

- **Unrepresentable CDS:** 957,269 of the 3,336,744 phold CDS (28.7%) are shorter than 90 aa, so a whole ORF of their own cannot be called at the 90-aa floor.
- **Short CDS inside long ORFs:** because stop-to-stop ORFs extend upstream of the start, 310,360 positive ORFs match a CDS shorter than 90 aa. In 77,799 positives the CDS covers less than half the ORF; the median CDS fraction of the ORF is 0.956.
- **One CDS per ORF:** no ORF matched more than one CDS.

## 3. Methods

Scripts are in `scripts/`. Each has a Slurm wrapper; all ran on Setonix CPU nodes.

1. `fetch_phage_mi_data.slurm` and `fetch_phage_mi_chunks.slurm` copy the TSV, the 145 `results/phold_*.json.gz` files and the 145 `chunks/phold_*.gbk.gz` files from the OneDrive remote to scratch.
2. `build_phage_orf_table.py` (`build_phage_orf_table.slurm`) builds one Parquet row per ORF. Each row holds:
   - coordinates, protein length and the five variables;
   - the number of distinct 3Di and 12-state states, and the fraction of `D` in the 3Di string;
   - the archived label.

   The same job runs `check_phage_labels.py` on 3 chunks and summarises the provenance of `phold.gbk`.
3. `relabel_phage_orfs.py` (`relabel_phage_orfs.slurm`) re-derives the ORFs of all 145 chunks with `get_orfs -l 90` and labels them with the current matcher. It uses `src/` from this worktree, which is identical to `main` at 750e2b9. `compare_labels.py` tabulates legacy vs v2.
4. `analyze_phage_mi.py` (`analyze_phage_mi.slurm`) runs the analysis. The seed is 20261014 throughout.
   - **Binned plug-in MI** I(X; label) is computed with empirical-quantile bins per axis (10, 20 or 40; tied quantiles merge bins), and for pairs with the 2-D product grid. It is reported on:
     - (a) a class-balanced sample of 200,000 ORFs per class, where H(label) = 1 bit; and
     - (b) the full population of 6.75 M ORFs at natural prevalence 0.398, where H(label) = 0.970 bits.
   - **Gain of a pair** is defined as I(pair) − max(I(each member)), using the same bins and the same data.
   - **Permutation null:** 200 label shuffles on the balanced sample, for every single variable and pair, at every bin count.
   - **Uncertainty:** 100 bootstrap resamples of the balanced sample give basic (reverse-percentile) 95% intervals for each MI value and for each pair's gain.
   - **Length control:**
     - Length bins are 30–49, 50–74, 75–89, 90–99, 100–149, 150–199, 200–299 and 300+ aa. The first three are empty because of the 90-aa floor.
     - Within each bin, the analysis computes MI with the label (20 within-bin quantile bins), ROC AUC and AP.
     - The conditional MI I(X; label | L) = Σ_l p(l) I(X; label | L = l) is computed for L = those bins and for L = 20 length-quantile strata, with X re-binned within each stratum. By the chain rule, I((X, L); label) = I(L; label) + I(X; label | L).
   - **Single-feature ROC AUC / AP** are computed on all 6.75 M ORFs. The score is the raw variable and the positive class is in_genbank, so AUC < 0.5 means the variable is lower in positives.
   - **Models:** genome-grouped 5-fold cross-validation (`GroupKFold` by genome) of `HistGradientBoostingClassifier` (200 iterations, learning rate 0.1, early stopping). It is trained on all ORFs of a random 25% of genomes (9,006 genomes, 1,705,008 ORFs) and gives out-of-fold AUC and AP for each feature set.

Figures use Seaborn/Matplotlib and are written as PNG (300 dpi) and PDF. The class colours are blue (in_genbank) and vermillion (not in_genbank).

## 4. Results (corrected label `v2`)

### 4.1 Dataset

| | in_genbank | not in_genbank |
|---|---|---|
| ORFs | 2,685,382 | 4,068,979 |
| median length (aa) | 184 | 120 |
| mean / median 3Di entropy (bits) | 2.663 / 2.720 | 1.188 / 1.248 |
| mean protein entropy (bits) | 4.034 | 3.713 |
| mean 3Di–12-state MI (bits) | 0.868 | 0.390 |
| mean DNA entropy (bits) | 1.946 | 1.943 |
| 3Di entropy ≤ log₂3 = 1.585 bits | 7.3% | 80.6% |
| ≤ 3 distinct 3Di states | 1.2% | 44.0% |
| median distinct 3Di states | 15 | 4 |
| `D` is the majority 3Di state | 8.0% | 65.7% |

These figures come from `tables/class_descriptives.tsv`. The class gap in 3Di entropy (1.47 bits in means) is smaller than in GTDB (2.1 bits; 3.12 vs 1.04). The GTDB and phage pipelines differ in labels and in the extent of length filtering, so the two gaps should not be compared directly.

### 4.2 Single variables vs pairs (Fig. 1; `tables/mi_single_and_pairs.tsv`)

The MI columns use 20 quantile bins per axis. Gain is the pair's MI minus the MI of its best member, on the balanced sample.

| features | I balanced (bits) | 95% CI | I full (bits) | fraction of H (full) | gain over best single | best single |
|---|---|---|---|---|---|---|
| **3Di H** | **0.5865** | 0.584–0.589 | 0.5725 | 0.590 | — | — |
| 3Di–12-state MI | 0.4794 | 0.477–0.482 | 0.4625 | 0.477 | — | — |
| protein H | 0.4299 | 0.427–0.432 | 0.4132 | 0.426 | — | — |
| 12-state H | 0.4246 | 0.422–0.427 | 0.4159 | 0.429 | — | — |
| length | 0.1298 | 0.128–0.131 | 0.1251 | 0.129 | — | — |
| DNA H | 0.0029 | 0.003–0.003 | 0.0027 | 0.003 | — | — |
| **protein H + 3Di H** | **0.6437** | 0.640–0.645 | 0.6253 | 0.645 | **+0.057** | 3Di |
| protein H + 3Di–12-state MI | 0.6321 | 0.628–0.634 | 0.6101 | 0.629 | +0.153 | MI |
| 3Di H + length | 0.6126 | 0.609–0.614 | 0.5983 | 0.617 | +0.026 | 3Di |
| 3Di H + 3Di–12-state MI | 0.6077 | 0.604–0.610 | 0.5910 | 0.610 | +0.021 | 3Di |
| 3Di H + 12-state H | 0.6024 | 0.599–0.604 | 0.5879 | 0.606 | +0.016 | 3Di |
| DNA H + 3Di H | 0.5966 | 0.593–0.598 | 0.5820 | 0.600 | +0.010 | 3Di |
| 12-state H + 3Di–12-state MI | 0.5594 | | 0.5438 | | +0.080 | MI |
| 3Di–12-state MI + length | 0.5509 | | 0.5362 | | +0.072 | MI |
| DNA H + protein H | 0.5305 | | 0.5128 | | +0.101 | protein |
| protein H + 12-state H | 0.5305 | | 0.5162 | | +0.101 | protein |
| DNA H + 3Di–12-state MI | 0.4835 | | 0.4660 | | +0.004 | MI |
| protein H + length | 0.4686 | | 0.4523 | | +0.039 | protein |
| 12-state H + length | 0.4422 | | 0.4323 | | +0.018 | 12-state |
| DNA H + 12-state H | 0.4418 | | 0.4316 | | +0.017 | 12-state |
| DNA H + length | 0.1472 | | 0.1418 | | +0.017 | length |

**Reading the table.**

- **No pair beats 3Di alone by much.** The best pair, protein + 3Di, adds 0.057 bits (≈ 6% of the label entropy). Every other partner of 3Di adds 0.010–0.026 bits.
- **Large gains come only from pairs without 3Di.** Pairs such as protein + 3Di–12-state MI (+0.153) or protein + DNA (+0.101) gain a lot, but they still stay below 3Di + protein.
- **DNA entropy carries essentially nothing** (0.003 bits; AUC 0.524).
- **Bias does not explain the gains.** The plug-in bias of a 400-cell pair grid on 400,000 ORFs is about 0.0007 bits; the null mean is 0.0006–0.0007 bits for pairs and ≤ 0.00007 bits for singles. This is two orders of magnitude below the reported gains.

**Estimator checks.**

- **Permutation null.** Across the 21 feature sets, the 99th percentile of the null is at most 0.00027, 0.00087 and 0.0032 bits at 10, 20 and 40 bins. Every observed value except DNA H is more than 100× larger. Every feature set, DNA H included, has permutation p = 1/201, the smallest value attainable with 200 permutations.
- **Bin-count sensitivity.**
  - Going from 10 to 40 bins raises every value by 0.001–0.014 bits (e.g. 3Di 0.580 → 0.588; protein + 3Di 0.638 → 0.649). Part of that rise is the larger plug-in bias of the finer grid (up to 0.003 bits for pairs at 40 bins).
  - The ranking of the top six feature sets is identical at 10, 20 and 40 bins.
  - The balanced-sample and full-population estimates agree to within 0.02 bits, and the full-population values are slightly lower because H(label) is 0.970 rather than 1 bit.

### 4.3 Length control (Fig. 2; `tables/by_length_bin_auc_ap_mi.tsv`, `tables/conditional_mi_given_length.tsv`)

The 30–49, 50–74 and 75–89 aa bins are **empty**: the shortest ORF is 90 aa. Prevalence of in_genbank rises with length, from 0.175 at 90–99 aa to 0.776 at 300+ aa.

Single-feature ROC AUC within each length bin:

| length bin (aa) | n | prevalence | 3Di H | 12-state H | protein H | 3Di–12 MI | DNA H | length |
|---|---|---|---|---|---|---|---|---|
| 90–99 | 1,084,215 | 0.175 | **0.874** | 0.805 | 0.846 | 0.874 | 0.441 | 0.531 |
| 100–149 | 2,770,291 | 0.279 | **0.905** | 0.817 | 0.867 | 0.894 | 0.475 | 0.579 |
| 150–199 | 1,122,332 | 0.461 | **0.953** | 0.872 | 0.894 | 0.933 | 0.547 | 0.539 |
| 200–299 | 876,634 | 0.578 | **0.978** | 0.924 | 0.916 | 0.958 | 0.591 | 0.548 |
| 300+ | 900,889 | 0.776 | **0.991** | 0.980 | 0.956 | 0.959 | 0.682 | 0.626 |
| all | 6,754,361 | 0.398 | **0.947** | 0.892 | 0.902 | 0.919 | 0.524 | 0.733 |

Average precision for 3Di within bins is 0.719, 0.833, 0.948, 0.983 and 0.996, against bin prevalences of 0.175–0.776.

Conditional MI given length, on the full population (bits):

| X | I(X; label) | I(X; label \| 5 length bins) | I(X; label \| 20 length quantiles) | retained | I((X, L₂₀); label) |
|---|---|---|---|---|---|
| 3Di H | 0.572 | 0.477 | 0.472 | 82% | 0.597 |
| 3Di–12-state MI | 0.463 | 0.411 | 0.410 | 89% | 0.535 |
| protein H | 0.413 | 0.331 | 0.327 | 79% | 0.452 |
| 12-state H | 0.416 | 0.312 | 0.306 | 74% | 0.431 |
| DNA H | 0.003 | 0.015 | 0.017 | (rises) | 0.142 |
| *length alone* | | I(L₅; label) = 0.116 | I(L₂₀; label) = 0.125 | | |

"5 length bins" are the five non-empty bins of the table above; the script stores this column as `I_X_label_given_lenbins`. "Retained" is I(X; label | L₂₀) / I(X; label). In the last column, L₂₀ is the 20-quantile length stratification.

**Interpretation.**

- **3Di entropy keeps most of its information within length strata.** Length explains little of it because, at the ≥ 90 aa floor, length is only weakly informative here (AUC 0.733).
- **3Di separation increases with length.** Its within-bin AUC goes from 0.874 at 90–99 aa to 0.991 at 300+ aa. This matches the GTDB and LOGAN pattern of strong separation for ORFs of about 100 aa and longer. The within-bin MI values in Fig. 2B are not directly comparable across bins, because H(label | bin) changes from 0.67 to 1.0 bits.
- **DNA entropy has a weak, length-dependent signal.** Its AUC flips from 0.44 (90–99 aa) to 0.68 (300+ aa), so its marginal near-zero MI hides opposite within-bin effects. Its conditional MI is larger than its marginal MI.

### 4.4 Discrete log₂k ceilings (Fig. 4)

- **3Di entropy:** the not-in_genbank distribution piles up just below log₂3 = 1.585 bits, with a sharp drop above it, and has a spike at 0 bits (1.33% of negatives vs 0.0008% of positives). 44% of negative ORFs use ≤ 3 distinct 3Di states, against 1.2% of positives.
- **12-state entropy:** negatives show spikes at about 0.08 bits and at log₂2 = 1 bit.
- **The `D` state** is the majority state in 66% of negative ORFs and 8% of positives. This is the same "D/P/V low-complexity" pattern seen in GTDB and in smORFs. What `D` represents structurally (disorder? coil?) remains untested.

### 4.5 Joint protein × 3Di density (Fig. 3)

The figure uses full-population 2-D histograms (cells of 0.01 × 0.02 bits) and a class-normalised log₂ enrichment panel.

- **Positives** form a column at protein entropy ≈ 3.8–4.25 bits, spanning 3Di ≈ 1.5–4.0 bits.
- **Negatives** concentrate at 3Di ≈ 1.0–1.6 bits across a wider protein-entropy range (≈ 2.8–4.2 bits), with a floor at 3Di = 0.
- **How protein entropy helps:** protein entropy separates positives from negatives only in the region where 3Di entropy is ambiguous (≈ 1.6–2.4 bits), which is the source of its modest +0.057-bit gain over 3Di alone.

### 4.6 Single-feature AUC and grouped-CV models (Fig. 5; `tables/grouped_cv_models.tsv`)

Single-feature ROC AUC on all ORFs:

| feature | AUC |
|---|---|
| 3Di | 0.947 |
| 3Di–12-state MI | 0.919 |
| protein | 0.902 |
| 12-state | 0.892 |
| length | 0.733 |
| DNA | 0.524 |

Genome-grouped 5-fold cross-validation (9,006 genomes, 1,705,008 ORFs, prevalence 0.396):

| model | out-of-fold AUC | out-of-fold AP |
|---|---|---|
| length | 0.733 | 0.659 |
| 3Di | 0.947 | 0.936 |
| length + 3Di | 0.955 | 0.947 |
| length + protein + 3Di | 0.968 | 0.958 |
| 5 entropies (no length) | 0.980 | 0.972 |
| length + 5 entropies | **0.983** | **0.977** |

The gain over length alone is ΔAP = +0.318 (0.659 → 0.977), similar in size to LOGAN's ΔAP ≈ +0.30 (0.58 → 0.88). Here, however, most of the gain comes from 3Di entropy by itself (AP 0.936).

## 5. Comparison with the other datasets

| dataset | label | ORF lengths | 3Di AUC | length AUC | notes |
|---|---|---|---|---|---|
| **Phage (this work)** | phold/Pharokka CDS (Pyrodigal-gv) | ≥ 90 aa | **0.947** | 0.733 | within-bin 3Di AUC 0.874–0.991; 82% of 3Di MI retained given length |
| LOGAN metagenomes | Prodigal `-p meta` | ≥ 30 aa | 0.935 | 0.964 | length-stratified 3Di AUC 0.705; 200–299 aa 0.99 |
| GMSC smORFs | curated positive/negative | 10–99 aa | 0.605 | 0.640 | 3Di 0.47 at 10–29 aa |
| GTDB r232 | deposited GenBank CDS | ~90 aa effective (v0.2.0, #103; to be confirmed) | not computed as AUC | — | 3Di class gap 2.1 bits |

Points of comparison:

- **Overall 3Di AUC is similar to LOGAN** (0.947 vs 0.935).
- **Length behaves very differently.** In LOGAN, ORFs down to 30 aa are included, and short ORFs are overwhelmingly non-coding, so length alone reaches 0.964. In the phage set, every ORF is ≥ 90 aa and phage genomes are gene-dense, so length is much weaker (0.733) and 3Di entropy beats it outright.
- **Matched-length comparison.** In the 200–299 aa bin both datasets give 3Di AUC ≈ 0.98–0.99. At 90–99 aa the phage value (0.874) is far above smORFs at 90–99 aa (0.626) and LOGAN at 30–49 aa (0.52). The phage set has no ORFs below 90 aa, so it cannot address the short-ORF regime.
- **Before quoting the LOGAN length-stratified AUC (0.705) against these numbers,** compute the same pooled statistic. That has not been done here; only the per-bin AUCs above are directly comparable.

## 6. Caveats

1. **The label is phold/Pharokka annotation, not ground truth.**
   - Positives are Pyrodigal-gv gene calls. The 3Di/structure signal is partly circular, since Pyrodigal-gv also favours long, protein-like ORFs.
   - Negatives include real unannotated genes. The problem is positive–unlabelled, so AUC and AP measure agreement with Pyrodigal-gv, not true precision.
2. **Upstream extension of stop-to-stop ORFs.** The ≥ 90%-of-shorter-interval rule lets a short CDS label a much longer ORF: 77,799 positives have a CDS covering less than half the ORF, and 310,360 match a CDS shorter than 90 aa. The ORF's entropy includes the untranslated upstream extension.
3. **Minimum ORF length.** It is 90 aa (issue #103), not the documented 30 aa. 28.7% of phold CDS (shorter than 90 aa) cannot appear as positives except when nested in longer ORFs, so the analysis says nothing about short phage genes.
4. **Plug-in entropy length bias.** Order-0 plug-in entropy is biased low for short strings, by roughly (K−1)/(2N ln 2) bits, so ~0.1 bits for 20 states at N = 90. This bias is shared by both classes within a length bin, which is why the within-bin AUC and conditional-MI analyses are the length-controlled results. The per-ORF 3Di–12-state MI is plug-in on a 20 × 12 joint and is biased upward for short ORFs.
5. **The binned MI is a plug-in estimate on quantile bins.** It is a lower bound on the continuous-variable MI, and it depends weakly on bin count (§4.2). The permutation null and the bootstraps cover the balanced sample only. The full-population values have negligible sampling error at N = 6.75 M, but they share the same discretisation.
6. **Non-independence.** ORFs from the same genome and overlapping ORFs in different frames are not independent. The grouped-CV models split by genome; the MI and AUC figures do not account for clustering. No homology split was done (contrast the smORF cluster-split).
7. **Unpinned model.** ModernProst-50M remote code is not pinned to a revision, and the model ran in fp16 on ROCm.
8. **Version uncertainty.** The genome_entropy version used for re-encoding (0.2.0) is inferred, not logged. The phold and Pharokka versions are unknown.

## 7. Figures

Every figure is provided as PNG and PDF.

| figure | content |
|---|---|
| `figures/fig1_mi_single_vs_pairs` | **A:** MI with the label for the 6 single variables (blue) and 15 pairs (grey), at 20 quantile bins, on the balanced sample. Whiskers are bootstrap 95% CIs, the ticks mark each pair's best single member, and the grey band is the permutation-null 99th percentile. **B:** sensitivity to bin count (10, 20, 40). |
| `figures/fig2_length_control` | **A:** single-feature AUC within length bins. **B:** within-bin MI. **C:** ORF counts per bin and class. |
| `figures/fig3_protein_vs_3di_density` | Full-population 2-D histograms of protein vs 3Di entropy for each class, plus a log₂ enrichment panel. |
| `figures/fig4_entropy_distributions_log2k` | Full-population 3Di and 12-state entropy distributions by class, with log₂k reference lines. |
| `figures/fig5_auc_single_and_models` | **A:** single-feature AUC. **B:** genome-grouped cross-validated gradient-boosting AUC and AP. |

![Fig 1](figures/fig1_mi_single_vs_pairs.png)
![Fig 2](figures/fig2_length_control.png)
![Fig 3](figures/fig3_protein_vs_3di_density.png)
![Fig 4](figures/fig4_entropy_distributions_log2k.png)
![Fig 5](figures/fig5_auc_single_and_models.png)

## 8. Reproduction

The large data stays on scratch at `/scratch/pawsey1018/edwa0468/genome_entropy/work/phage_mi/` and is not committed. The commands below are run from `Validation/mutual_information/scripts/` with `W=<work dir>` and `R=<repo checkout>`:

```bash
j1=$(sbatch --parsable fetch_phage_mi_data.slurm $W)
j2=$(sbatch --parsable fetch_phage_mi_chunks.slurm $W)
# build_phage_orf_table.slurm also needs W/provenance/phold_{0000,0072,0144}.gbk.gz
j3=$(sbatch --parsable --dependency=afterok:$j1 build_phage_orf_table.slurm $W $R $R/Validation/Inphared/phold.gbk.gz)
j4=$(sbatch --parsable --dependency=afterok:$j2 relabel_phage_orfs.slurm $W $R)
sbatch --dependency=afterok:$j3:$j4 analyze_phage_mi.slurm $W/phage_orfs.parquet $R \
    --label v2 --relabel $W/phage_orf_labels_v2.parquet
sbatch --dependency=afterok:$j3 analyze_phage_mi.slurm $W/phage_orfs.parquet $R \
    --label legacy --skip-models --tables $R/Validation/mutual_information/tables/legacy_label --figures $W/figures_legacy_label
python compare_labels.py $W/phage_orfs.parquet $W/phage_orf_labels_v2.parquet $W/relabel_chunk_summary.tsv --tables ../tables
```

**Environment:** `/scratch/pawsey1018/edwa0468/software/envs/logan` (Python 3.12, numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, seaborn 0.13.2, matplotlib 3.11.2, pyarrow 25.0.0), with `get_orfs` 0.1 from `~/software/bin`.

**Jobs (2026-10-14):**

| job ID | step |
|---|---|
| 50802161 | fetch |
| 50802565 | build (debug partition) |
| 50802606 | fetch chunks |
| 50802607 | relabel (2.5 min) |
| 50802855 | v2 analysis (4 min) |
| 50802856 | legacy analysis |

## 9. Open items

- Rerun or regenerate the downstream phage products that inherited the minus-strand label bug:
  - the `rank_orfs` candidate list (52,057 high-probability unannotated ORFs); check whether its 312 legacy JSON inputs carry the same labels;
  - the original stub figures.
- Record phold and Pharokka versions, and confirm the Inphared release with a record-level accession comparison against the `7Apr2026` parts.
- Consider restricting positives to ORFs whose matched CDS covers ≥ 90% of the ORF, or entropy computed on the CDS (start-to-stop) part only, to remove upstream-extension contamination.
- Add a homology-grouped split (e.g. by vConTACT/genus cluster) to the CV models.
