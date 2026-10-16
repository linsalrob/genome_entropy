# Figure and table plan

This is a proposal awaiting Rob's decision (D4 in [`decisions.md`](decisions.md)).
Package IDs are the owners' provisional ones:
- `L` = LOGAN agent (#117);
- `G` = Gadi (#118 and later);
- `S` = Gadi smORFs;
- `P` = Setonix phage;
- `I` = integrator.

Final figure numbers are assigned here, and the IDs are renamed once the plan
is agreed. Every panel must have a notebook and committed plotting data (see
[`notebooks/README.md`](notebooks/README.md)). Status is tracked in
[`figures_manifest.tsv`](figures_manifest.tsv).

## Storyline

1. **Real proteins and spurious ORFs occupy different structural-entropy
   regimes.** Shown at population scale on 2.62 billion GTDB ORFs. Spurious
   ORFs sit under the log₂k ceilings that a 2–3-state 3Di alphabet allows.
2. **The signal is structural, not length.** Plug-in entropy rises with
   length mostly as estimator bias (#101). Within fixed length bins, 3Di
   separates genes from non-genes in 90 billion LOGAN ORFs.
3. **It is specific to the reading frame.** In six-frame comparisons 3Di
   ranks the true frame first 92.4% of the time, and alternative frames
   do not inherit it.
4. **It finds genes the annotations missed.** In GTDB, ~51–59k
   (floor ~5,400), checked against matched shadows and an independent gene
   caller. In LOGAN, 3.43 M candidates, 41.9% with full-length structural
   homologues against ≤0.45% for controls.
5. **It corrects annotation in phage**, a mobile, gene-dense use case:
   mutual information, corrected ranking with Foldseek validation, and the
   OM869708 example.
6. **Limits:** short ORFs (smORFs, AUC 0.73), the 90-aa floor, Prodigal
   disagreement, and hardware non-determinism. These go in the
   supplementary material and Discussion.

## Main figures

| fig | title (working) | panels → package | owner | status |
|---|---|---|---|---|
| **1** | Joint sequence and structural entropy separates coding ORFs | a: schematic of ORF → aa, 3Di and 12-state entropy (I01); b–c: GTDB protein vs 3Di, bacteria and archaea, with log₂k ceilings (G01) | I, Gadi | G01 in progress |
| **2** | Structural entropy is informative beyond length | a: estimator bias vs biology (G02, #101); b: nested models, what 3Di/12st/MI add beyond length (L02); c: within-length-bin AUC (L03 panel) | Gadi, LOGAN | L02/L03 in #117 |
| **3** | Structural entropy is specific to the reading frame | a: six-frame inheritance; b: true-frame discrimination (L05 a–b) | LOGAN | in #117 |
| **4** | Missed genes in GTDB | a: candidate funnel; b: full-length coverage by arm; c: evidence ladder (G04); d: Prodigal agreement by arm (G05) | Gadi | in #118 |
| **5** | Missed-gene candidates in 50,000 metagenomes | a: Foldseek evidence by arm, five databases, with controls (L06); b: low-scoring Prodigal genes (L10) | LOGAN | in #117 |
| **6** | Phage: mutual information, ranking and an annotation error | a: MI of each entropy with the label (P01); b: ranking performance and Foldseek validation of candidates (P02); c: OM869708 locus (P03) | Setonix | to do |

## Supplementary figures

| package | content | owner |
|---|---|---|
| L01 | LOGAN feature distributions at fixed length | LOGAN |
| L04 | contig-end fragments | LOGAN |
| L05c | six-frame Δ | LOGAN |
| L07 | candidate evidence by length, classes, biome × superkingdom | LOGAN |
| L08 | contig taxonomy (Kraken2) and the eukaryotic test | LOGAN |
| L09 | why Prodigal skipped candidates | LOGAN |
| LS1 | length-controlled separation by biome and tranche | LOGAN |
| G02 | if not main: bias decomposition and z-score calibration | Gadi |
| G03 | whole-population entropy summary | Gadi |
| G06 | exemplar loci and functional composition | Gadi |
| GS* | truncation, assembly strata, hit quality, evidence classes, unmatched-comparator sensitivity | Gadi |
| S01+ | GMSC smORFs: density, clusters, stripes | Gadi |
| P04 | phage vs bacterial entropy by length (Gadi phage run) | Gadi |
| P05 | phage ranking detail: feature importance before and after the label correction | Setonix |

## Tables

| table | content | owner |
|---|---|---|
| 1 | Datasets and encoding configurations: n genomes/ORFs, labels, ORF floor, package, model revision, encoding size, hardware (D9) | integrator |
| 2 | LOGAN ORF classes (tabL*) | LOGAN |
| 3 | GTDB direct-evidence ladder, bacteria and archaea (G04 data) | Gadi |
| S* | LOGAN tables (tabL01–09), GTDB supplementary tables, smORF tables, determinism test (Gadi), phage tables (Setonix) | owners |

## Removed from the current manuscript

| current item | why | replacement |
|---|---|---|
| Fig. 1 (`images/media/image1.png`) | old test-case scatter, not GTDB | Fig. 1b–c (G01) |
| Figs "8", "10", "11" (`images/223ec3…`, `4ca389…`, `2c9587…`) | four-database LOGAN results; 560 px | Fig. 5 (L06), L07 |
| Table 2 (LOGAN ORF classes) | partial-shard numbers | Table 2 from final S3 |
| Table 7 (LOGAN Foldseek) | four databases | five-database table from L06 data |
| Figs 2, 3, 6, 8 (`images/media/image2/4/6/8.png`) | correct but orphan PNGs | L01, L03, L05 notebooks |
| OM869708 locus (`images/media/image9.png`) | orphan; numbered "Fig. 1" | Fig. 6c (P03) |
