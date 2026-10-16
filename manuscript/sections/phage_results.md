<!-- Draft Results text, Setonix section drafter, agent-handoffs#24 checkpoint D. The integrator merges it into genome_entropy.md. -->
<!-- Sources: Validation/mutual_information/README.md (MI); Validation/Inphared/README.md (IR); AH inphared_gadi/README.md (IG); manuscript/data/figP01_*, figP04_*; decisions D5–D8, D11, D12. -->

## Phage genomes

### Structural entropy carries most of the information about phage annotation

Phage genomes are densely coding: 39.8% of the 6,754,361 ORFs of at least 90 aa in 36,025 Inphared genomes match an annotated CDS. <!-- src: Validation/mutual_information/README.md TL;DR (2,685,382 / 6,754,361); IG:14 --> These labels are corrected ones. The matcher used before genome_entropy 0.2.0 labelled 3 of 3,311,070 minus-strand ORFs as genes, against 1,000,839 after correction (Fig. S15a). <!-- src: Validation/mutual_information/README.md §2.1 table; figPS01 caption -->

3Di entropy alone carries 0.587 bits of information about the label (95% CI 0.584–0.589), out of 1 bit on a class-balanced sample (Fig. 6a). <!-- src: Validation/mutual_information/README.md §4.2 table; data/figP01_mutual_information.tsv.gz --> As a single score it gives a ROC AUC of 0.947. <!-- src: Validation/mutual_information/README.md §4.6 --> The best pair, protein plus 3Di entropy, reaches 0.644 bits, only 0.057 bits (95% CI 0.055–0.058) more than 3Di entropy alone. <!-- src: Validation/mutual_information/README.md TL;DR, §4.2 table --> Every other partner adds at most 0.026 bits to 3Di entropy. <!-- src: Validation/mutual_information/README.md TL;DR --> DNA entropy carries essentially nothing (0.003 bits). <!-- src: Validation/mutual_information/README.md §4.2 table --> Every estimate is far above the permutation null, whose 99th percentile is at most 0.0009 bits at 20 bins. <!-- src: Validation/mutual_information/README.md TL;DR -->

The signal is not a length effect. Every ORF here is at least 90 aa, and length alone is weak (0.130 bits; AUC 0.733). <!-- src: Validation/mutual_information/README.md §4.2 table, §4.6 --> Conditioning on length keeps 82% of the information in 3Di entropy (0.472 of 0.572 bits, full population). <!-- src: Validation/mutual_information/README.md §4.3 conditional-MI table --> Within length bins, 3Di entropy gives AUC 0.874 at 90–99 aa, rising to 0.991 at 300 aa and above, against 0.53–0.63 for length. <!-- src: Validation/mutual_information/README.md §4.3 AUC table -->

Amino-acid composition therefore adds little once predicted structure is known (D8). The labels are Pyrodigal-gv gene calls, not ground truth, so these values measure agreement with that gene caller.

### Ranking missing genes with corrected labels

Fig. S15b shows what the label error cost. The original ranking model included strand among its features. Trained on the old labels, it learned that a plus-strand ORF is a gene: the strand feature carried 0.89 of its gain. <!-- src: Validation/Inphared/README.md:10, §4.2 table; figPS01 caption --> Scored against corrected labels, that model reaches AUC 0.740 only. <!-- src: Validation/Inphared/README.md Table 1 (0.7404) -->

We therefore retrained on corrected labels using intrinsic features only: entropies and lengths, with no strand, frame or position. On the 31,200-genome ranking set this model gives AUC 0.976 and average precision (AP) 0.970, against 0.733 and 0.660 for length alone (Fig. 6b). <!-- src: Validation/Inphared/README.md Table 1 (intrinsic 0.9764 / 0.9696; length only 0.7328 / 0.6598); data/figP01_ranking_models.tsv.gz --> Length plus 3Di entropy already reaches 0.953 and 0.945. <!-- src: Validation/Inphared/README.md Table 1 --> 3Di entropy takes 0.745 of the model's gain. <!-- src: Validation/Inphared/README.md §4.2 intrinsic table --> The ranking set's positives make up 40% of ORFs, so AP here is not comparable with AP on metagenomes. <!-- src: Validation/Inphared/README.md:139, :192 -->

The model flags 46,132 unannotated ORFs at p ≥ 0.8, in 19,560 genomes (median 136 aa). <!-- src: Validation/Inphared/README.md:210 --> Most overlap an annotated gene:

- 22,913 are antisense to a CDS;
- 15,005 lie in another frame on the same strand;
- 4,169 are intergenic.

<!-- src: Validation/Inphared/README.md §5.1 candidate context table -->

The remaining 4,045 are pieces of annotated genes, mostly partial CDS at genome ends, and were not searched. <!-- src: Validation/Inphared/README.md:224 -->

Structural searches support the candidates (Fig. 6c). Full-length Foldseek hits in any database were found for:

- 33.8% of candidates (95% CI 33.1–34.5%);
- 64.6% of annotated genes;
- 2.2%, 0.24% and 0.10% of low-scoring intergenic, other-frame and antisense ORFs;
- 0.015% of candidates with shuffled 3Di strings.

<!-- src: Validation/Inphared/README.md §5.2 rate table; data/figP01_foldseek_rates.tsv.gz (fullcov_ci) -->

Scaled between the controls and the annotated genes, this places 51–52% of candidates with the genes. <!-- src: Validation/Inphared/README.md §5.2 excess table (50.6–52.3%); tables/evidence_excess.tsv --> Most of that evidence comes from BFVD, a database of predicted structures of viral proteins. Its hits are mostly near-identical (median identity 0.98 for candidates), so they mostly show that another annotation of a related genome called the same ORF. <!-- src: Validation/Inphared/README.md §5.2 identity table --> The independent test is the non-viral databases. There candidates reach 6.1% (95% CI 5.8–6.5%), against 18.0% for annotated genes and at most 0.09% for every control. <!-- src: Validation/Inphared/README.md §5.2; data/figP01_foldseek_rates.tsv.gz --> That gives a share of 34%. <!-- src: Validation/Inphared/README.md §5.2 excess table (33.7–34.0%) -->

Antisense candidates validate poorly: 9.9% have a full-length hit, against 66.5% of intergenic and 61.6% of other-frame candidates. <!-- src: Validation/Inphared/README.md §5.2 by-context table --> Many are probably shadows of the gene on the opposite strand.

Of the 4,102 replacement pairs, only the replacement ORF has a full-length hit in 46.6% of pairs, and only the annotated gene in 5.9%. <!-- src: Validation/Inphared/README.md §6 table --> Replacement ORFs hit 53.8% of the time, the genes they would replace 13.2%. <!-- src: Validation/Inphared/README.md §6 hit-rate table --> This test shows only which ORF of a pair has better external evidence. Phages do carry overlapping genes, so each case needs checking before it is called an annotation error.

Two limits apply to every phage result:

- **Short genes.** The 90-aa floor hides 28.7% of the annotated CDS, so these results say nothing about short phage genes (D6). <!-- src: Validation/mutual_information/README.md §2.1 (957,269 / 3,336,744) -->
- **Near-duplicate genomes.** Inphared contains near-duplicate genomes, and cross-validation was grouped by genome, not by homology cluster. Out-of-fold metrics may therefore be slightly optimistic.

### Phage genes sit lower in 3Di entropy than cellular genes

Annotated phage genes show the same separation as bacterial and archaeal genes, and the gap widens with length in every population (Fig. S14). Annotated phage ORFs have lower mean 3Di entropy than annotated bacterial ORFs at every length:

| length (aa) | difference (bits) |
|---|---|
| 90–149 | 0.24 |
| 150–299 | 0.34 |
| 300–599 | 0.20 |
| 600–999 | 0.09 |

<!-- src: notebooks/figP04_phage_vs_cellular.ipynb caption (a); recomputed from data/figP04_{phage,gtdb}_length_3di.tsv.gz as n-weighted means: 0.242, 0.339, 0.194, 0.090 -->

We have not tested the cause (D11). Phage proteomes may be richer in small or poorly folded proteins. Alternatively, the labels may contribute: 310,360 phage positives match a CDS shorter than 90 aa inside a longer stop-to-stop ORF. <!-- src: AH inphared_gadi/README.md §3.2; Validation/mutual_information/README.md §2.1 --> Restricting positives to CDS that cover most of their ORF would separate the two.

### An annotation error in OM869708

OM869708 is the 5,753-bp genome of the microvirus *Dipodfec virus* RodF1_38 (Fig. 6d). <!-- src: work/om869708/OM869708_ncbi.gb (DEFINITION, ORGANISM, REFERENCE bases 1 to 5753) --> The submitters' GenBank record annotates a replication initiator protein at 4753..5730 (UPW42009.1). <!-- src: data/figP01_om869708_ncbi_cds.tsv.gz (0-based 4752–5730); decisions.md D7 --> The Pharokka and phold re-annotation used for our labels does not contain this gene. Instead it calls a hypothetical protein in another frame, CDS_0007, which matches ORF7. <!-- src: data/figP01_om869708_cds.tsv.gz; data/figP01_om869708_orfs.tsv.gz -->

The corrected model disagrees with the re-annotation. ORF3 is the 335-aa stop-to-stop ORF that contains the submitters' gene and shares its stop codon. It has 3Di entropy 2.98 bits and scores p = 0.889. <!-- src: data/figP01_om869708_orfs.tsv.gz (orf3); decisions.md D7 --> The annotated ORF7 (198 aa) has 3Di entropy 1.20 bits and scores p = 0.175. <!-- src: data/figP01_om869708_orfs.tsv.gz (orf7); decisions.md D7 --> The model ranks ORF3 → ORF7 179th of the 4,102 replacement candidates. <!-- src: data/figP01_om869708_orfs.tsv.gz (replacement_rank); decisions.md D7 -->

BLASTP against nr agrees with the model. ORF7 has no hit at E ≤ 10⁻⁵. <!-- src: data/figP01_om869708_blastp.tsv.gz (no OM869708_orf7 rows) --> ORF3's top hit is the genome's own submitted protein (UPW42009.1, 100% identity), which is not independent support. <!-- src: data/figP01_om869708_blastp.tsv.gz row 1 --> The next five hits are replication initiator proteins of other viruses, one of them a named microvirus, at 43.7–55.4% identity. The best of these aligns over 323 aa with E = 2 × 10⁻¹²⁷. <!-- src: data/figP01_om869708_blastp.tsv.gz rows 2–6 (XHH71331.1 55.4%, E 2.4e-127 … QXN75198.1 43.7%) -->

genome_entropy therefore recovers a replication gene that the submitters annotated and the re-annotation pipeline lost. We did not determine why the pipeline lost it. Its functional annotation step can only describe proteins that its gene caller reports, so a wrong gene model is not corrected downstream. This example shows how one score per ORF can flag such a locus for review. It does not by itself show that the replacement is correct; here that rests on the submitters' annotation and the independent homologues.
