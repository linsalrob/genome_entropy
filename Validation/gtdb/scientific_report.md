# Structural-state entropy across GTDB representative genomes

**Dataset:** GTDB release 232 (R11-RS232, 15 April 2026)
**Model:** `gbouras13/modernprost-50M` — dual-head ModernProst, 3Di + 12-state
**Tool:** `genome_entropy` 0.2.0, output schema 2.2.0
**Platform:** NCI Gadi (PBS Pro), project `ob80`
**Report date:** 2026-08-31 — bacteria complete, archaea in progress

---

## 1. Summary

3Di and 12-state structural entropy was computed for every bacterial species
representative in GTDB r232. The headline analytical result is that the
**boundary at 3Di entropy ≈ 1.585 bits, visible as a sharp horizontal line
separating GenBank-matched from unmatched ORFs, is an information-theoretic
ceiling rather than a biological threshold**: it is exactly log₂(3), and it
arises because the encoder assigns a large ORF population to only three
distinct 3Di states. What those states correspond to structurally is not
established here — see the note under "Why encodings collapse to three
states" in §5. Section 5 establishes the ceiling quantitatively.

A secondary result (section 6) is that the population of unmatched ORFs with
high 3Di entropy — a candidate pool for genes missed by annotation — is 95.5%
attributed to genomes that appear never to have been annotated. A further 64%
of the remainder was attributed to shadow ORFs overlapping real CDS, leaving
about 1.6% as candidates.

Both splits are provisional. The shadow test compared plus- and minus-strand
coordinates in different systems, and "never annotated" was inferred from
`in_genbank` rather than from the GenBank records, which can only over-count
unannotated genomes. Both defects are fixed in the scripts; the numbers here
predate the fixes. See the note in section 6.

---

## 2. Coverage

| | count |
|---|---:|
| GTDB bacterial representatives | 189,801 |
| Not served by NCBI (suppressed/withdrawn) | 47 |
| Downloaded | 189,754 |
| Failed to encode (see §7) | 39 |
| **Genomes encoded** | **189,715 (99.955%)** |
| **ORFs with entropy values** | **2,568,244,984** |
| Mean ORF calls per genome | 13,537 |

Archaea: all 10,122 representatives downloaded; encoding submitted.

`in_genbank` — whether a called ORF was matched to an annotated CDS by genomic
overlap, frame and translation:

| | count | share |
|---|---:|---:|
| `in_genbank = True` | 309,065,661 | 12.03% |
| `in_genbank = False` | 2,259,179,323 | 87.97% |

The low matched fraction is expected and is not an error rate. `get_orfs`
enumerates open reading frames in all six frames, so most calls are not genes.
Separately, **at least 46% of bacterial representatives appear to carry no CDS
annotation at all** (GCA assemblies submitted unannotated), and every ORF in
those genomes is `False` by construction. Any analysis of what `False` *means*
must exclude them; see §6. That 46% is an upper bound: it was measured from
`in_genbank`, which cannot distinguish a genome with no annotation from one
whose annotations the ORF matcher rejected. `12_genome_cds_counts.pbs` answers
this from the GenBank records; see the note in §6.

---

## 3. Methods

Downloads used NCBI `datasets` (18.36.0) via the dehydrated → rehydrate path on
`copyq`, the only Gadi queue with outbound internet. Encoding ran on `gpuvolta`
(Tesla V100-SXM2-32GB) with `genome_entropy run --genbank`, model cached offline
under `HF_HOME` on `/g/data` with `HF_HUB_OFFLINE=1`.

Genomes were processed in chunks of 250. Per-genome files were written to
node-local `$PBS_JOBFS` and only two files per chunk returned to `/g/data`: a
`zstd -3` archive of all per-genome JSON, and a gzipped per-ORF TSV of the
entropy values. This was forced by the inode quota, not by disk space —
see §8.

### Parameters, chosen by measurement

**`--encoding-size 10000` (the default; raising it is slower).** The per-batch
token budget was swept on three bacterial genomes over two rounds, with under
1% round-to-round variation:

| `--encoding-size` | s/genome | vs default |
|---|---:|---:|
| **10000** | **119.5** | **1.000 (fastest)** |
| 25000 | 120.3 | 0.993 |
| 50000 | 124.0 | 0.964 |
| 100000 | 131.5 | 0.909 |

An earlier single-genome sweep extended this: 800000 was 14% slower than the
default, and 400000 and 800000 returned byte-identical peak memory, meaning the
budget already exceeded one genome's entire protein content. `encoding_size` is
a token budget per batch, so a wider budget batches sequences of more varied
length and pads each to the longest in its batch; past a point the padding costs
more than the wider batch saves. A single `run` leaves the GPU at 12–48%
utilisation using 1.5–2.1 GB of 32 GB, which invites the opposite conclusion,
but memory was never the binding resource.

**`PARALLEL=4` (four `genome_entropy` processes per GPU).** One at a time used a
single core of the 12 that `gpuvolta` bills per GPU, and left the device idle
waiting on CPU-side work (`get_orfs`, GenBank parsing, JSON writing):

| processes | genomes/hr | speedup | GPU mem |
|---|---:|---:|---:|
| 1 | 40 | 1.00× | 3.1 GB |
| 2 | 68 | 1.71× | 4.5 GB |
| **4** | **82** | **2.04×** | **7.8 GB** |
| 6 | 83 | 2.07× | 10.5 GB |
| 12 | 82 | 2.07× | 19.2 GB |

Throughput saturates at four, taking utilisation from 12% to 79%. Beyond that
the device itself is the bottleneck and extra processes only consume VRAM. The
gain is ~2×, not the ~20× that "4% of GPU memory in use" suggests.

Choosing `PARALLEL=4` reduced the projected bacterial cost from ~227,000 SU to
~74,000 SU.

---

## 4. Entropy separation by CDS support

Restricted to genomes carrying at least one annotated CDS (3,069 of 5,996
sampled genomes), matched and unmatched ORFs occupy substantially different
regions of (protein entropy, 3Di entropy) space:

- matched ORFs peak near (4.05, 3.6)
- unmatched ORFs peak near (3.8, 1.35)

The separation is almost entirely in **3Di**, not protein, entropy: unmatched
ORFs have plausible amino-acid composition but structurally monotonous 3Di
strings. Mean 3Di entropy across 250 genomes was 3.153 for matched ORFs against
1.818 for all ORFs.

Filtering to annotated genomes matters a great deal. It removes 95.5% of the
high-3Di unmatched population, collapses panel B of the density figures from
three apparent populations to one mode, and raises the matched fraction in the
sample from 11.8% to 21.0%. The unfiltered figures substantially understate the
separation.

### Figures

In `/g/data/ob80/re3494/gtdb_entropy/figures/`:

| file | content |
|---|---|
| `protein_vs_3di_entropy.png` | scatter, 4 panels, all genomes |
| `protein_vs_3di_entropy_annotated.png` | scatter, annotated genomes only |
| `protein_vs_3di_hexbin.png` | hexbin, log counts, all genomes |
| `protein_vs_3di_hexbin_annotated.png` | hexbin, annotated genomes only |
| `protein_vs_3di_kde.png` | KDE + contour overlay, all genomes |
| `protein_vs_3di_kde_annotated.png` | KDE + contour overlay, annotated only |

The four-panel scatter exists to make one methodological point: panels C and D
hold identical data in opposite draw order, and the two readings differ
completely. With ~175k unmatched points over ~23k matched, whichever class is
drawn last wins. Neither ordering is neutral — drawing the 12% class last
overstates it just as burying it understates it — which is the argument for the
binned and smoothed views.

Palette (`#2a78d6` / `#eb6834`) was checked for colour-vision separation before
use: OKLab ΔE 33.6 normal, 46.7 deuteranopia, 24.7 protanopia, 33.5 tritanopia.

---

## 5. The line at 3Di entropy 1.585 is log₂(3)

The hexbin of unmatched ORFs shows a pronounced horizontal boundary with most
of the density below it. **It is not a biological threshold. It is the maximum
entropy attainable by a three-symbol string.**

Shannon entropy over *k* distinct symbols cannot exceed log₂(k). Observed
maxima match that bound exactly (52,947 ORFs from three genomes of chunk
`bac_000`, encodings ≥30 residues):

| distinct 3Di states used | n | max observed H | log₂(k) |
|---|---:|---:|---:|
| 2 | 7,170 | **1.0000** | 1.0000 |
| 3 | 18,286 | **1.5850** | 1.5850 |
| 4 | 6,401 | 1.753 | 2.000 |
| 5 | 2,582 | 2.053 | 2.322 |
| ≥15 | 8,593 | 4.104 | 4.322 |

`n_states = 3` is the single largest group. Every ORF in it is capped at 1.585
by arithmetic, and that population's ceiling is the line.

### Why encodings collapse to three states

| | H < 1.585 | H ≥ 1.585 |
|---|---|---|
| median distinct states | **3** | **16** |
| median top-state fraction | **0.750** | 0.338 |
| dominant state is `D` | **89.1%** of ORFs | 55.3% |
| residue usage | **D 76.1%, V 14.7%, P 8.6%** | D 26.9%, V 21.1%, P 9.6%, L 7.1%, S 6.6% |
| median length (aa) | 120–190 | 288–372 |
| fraction matched to CDS | ~0–1% | 3.6% → 66.9% rising with H |

Below the line, **three letters account for 99.4% of all residues.** A string
that is ~76% `D` with a little `V` and `P` is mathematically pinned below
1.585. Above the line the full 20-letter alphabet is in genuine use with no
single dominant state.

The distribution is therefore **bimodal by mechanism**, not a continuum with a
cut imposed on it:

- *low state diversity* — ≤3 effective states, H ≤ 1.585 by arithmetic
- *full alphabet* — most of the 20 states in use, H spreading to ~4.1

That is why the boundary is crisp: it is where a hard ceiling meets a genuinely
different regime. Note that this argument is purely about how many distinct
states the encoder emits; it does not depend on what any individual state
means.

> **What `D` means is not established here.** An earlier version of this
> report called `D` "the coil/unstructured state" and read its dominance as the
> model reporting "no resolvable structure". Foldseek's 3Di alphabet is a
> learned 20-state description of each residue's local tertiary-interaction
> geometry; the letters are borrowed from the amino-acid alphabet for tooling
> convenience and are not named secondary-structure categories, so there is no
> designated coil state. Dominance by `D` is a measured fact and is sufficient
> to explain the entropy ceiling, but reading it as disorder needs an
> independent measurement — DSSP over predicted structures, or a disorder
> predictor, on the same ORFs. Recorded as outstanding work.

### Consequences for interpretation

1. **1.585 is the principled threshold**, not the ~1.5 read off the figure by
   eye. Use the constant.
2. **3Di entropy is partly length-confounded.** Median length below the line is
   120–190 aa against 288–372 above; short sequences have fewer opportunities
   to display state variety. Some of the separation in §4 is length, not
   structure.
3. **A cleaner discriminant exists.** The *fraction of `D` residues* measures
   dominance by a single 3Di state directly and, unlike entropy, is not bounded
   by sequence length. Worth computing as an alternative axis — as a measure of
   state dominance, not of disorder, until the note above is settled.

---

## 6. Candidate genes missed by annotation

Hypothesis tested: unmatched ORFs with 3Di entropy above ~2.5 look
structurally like real proteins, so perhaps the original annotation software
failed to call them.

> **The shadow/intergenic split below is superseded and awaits recomputation.**
> Two independent defects produced it, both now fixed in the scripts and
> neither reflected in the numbers here.
>
> *Coordinates.* Overlap was tested using raw `get_orfs` coordinates. On the
> negative strand those index the reverse complement, so every
> plus-versus-minus comparison was made between two different coordinate
> systems: real cross-strand shadows were missed and unrelated spans were
> counted as shadows. Both sides are now placed on the forward genomic axis
> with `normalise_orf_interval`, the same helper the `in_genbank` matcher uses.
>
> *What counts as "annotated".* The CDS set was taken to be the spans of ORFs
> whose `in_genbank` flag was True. That misleads in both directions: a CDS the
> matcher rejected is absent from it, so an ORF sitting on a real gene reads as
> intergenic, and a matched ORF runs stop to stop and can extend past the
> deposited CDS, so its overhang manufactures shadows. The test now uses the
> deposited GenBank coordinates from `13_cds_intervals.pbs`. The example data in
> this repository contains a concrete instance: an origin-crossing CDS that the
> matcher skips entirely, and which the old method therefore could never see.
>
> Synthetic checks show both defects produce errors in opposing directions, so
> totals can look stable while individual ORFs sit in the wrong group. Treat
> the 15,338 / 8,651 split, everything derived from it, and the "1.6% survives"
> figure as provisional until the analysis is rerun.
>
> The 95.5% attributable to genomes with no annotation at all does not depend
> on coordinates and is unaffected by that fix — but it has a separate problem
> of its own, below.
>
> **"Never annotated" was inferred from `in_genbank`, which cannot establish
> it.** The flag is set only when a called ORF passes the coordinate, frame,
> and translation match, so a genome with real CDS features that all fail that
> match was counted as never annotated and had its high-3Di ORFs written off.
> The error can only run one way: it over-counts unannotated genomes, so the
> 95.5% here and the 46% quoted in §1 and §6 are upper bounds, not estimates.
> `12_genome_cds_counts.pbs` now counts CDS features in the GenBank records
> directly, and `10_missed_genes.py` requires its table instead of the proxy.
> Both figures need regenerating from it.
>
> Rerunning needs chunk TSVs carrying the `contig_length` column that
> `extract_entropy_rows.py` now emits; TSVs from the original run do not have it
> and must be regenerated from the JSON archives.

Two confounders dominate. Starting from 537,488 unmatched ORFs with 3Di ≥ 2.5
across 500 genomes (chunks `bac_000`, `bac_051`):

| | count | share |
|---|---:|---:|
| in genomes with **no annotation at all** (upper bound, see note) | 513,499 | **95.5%** |
| in annotated genomes | 23,989 | 4.5% |
| → **shadow** of an annotated CDS (overlapping, any frame/strand) | 15,338 | 63.9% of those |
| → **intergenic — candidates** | **8,651** | 36.1% of those |

`get_orfs` reads all six frames, so an unmatched ORF overlapping a real CDS
inherits real coding structure and can score high 3Di entropy without being a
distinct protein. Only intergenic ORFs in annotated genomes are candidates:
**1.6% of the original population.**

Those survivors do look protein-like:

| group | n | median aa | ≥100 aa | median 3Di |
|---|---:|---:|---:|---:|
| annotated CDS | 864,934 | 323 | 98.4% | 3.30 |
| **candidate: intergenic, 3Di ≥ 2.5** | **8,651** | **169** | **88.7%** | **2.89** |
| intergenic, 3Di < 2.5 | 741,921 | 126 | 80.8% | 1.18 |
| shadow of CDS, 3Di ≥ 2.5 | 15,338 | 172 | 87.7% | 2.85 |

Candidates sit between annotated CDS and the low-3Di background on every axis —
shorter than real genes but longer than background, with 3Di entropy much
closer to real CDS than to noise. That is consistent with missed genes, which
should be biased toward short proteins precisely because short ORFs are what
annotation pipelines under-call. Median **24 candidates per annotated genome,
≈0.9% of its CDS count** — a plausible miss rate rather than an implausible one.

**This does not discriminate missed genes from other intergenic coding-like
sequence**: pseudogenes, phage or IS remnants, or ORFs annotation deliberately
suppressed. Two decisive next steps:

1. **Homology search** the candidates against UniProt/Pfam. A genuinely missed
   gene usually has homologs. `copyq` has internet access.
2. **Stratify by annotation provenance.** GTDB records which pipeline annotated
   each genome; if candidates concentrate in older Prokka versions rather than
   recent PGAP, that is direct evidence of pipeline misses.

---

## 7. Encoding failures

39 of 189,754 genomes (**0.021%**) failed, all with the same cause: CUDA
out-of-memory inside `_make_sliding_mask` in the model's remote code, which
materialises a full L×L matrix to build what is by definition a *banded*
sliding-window mask. Memory is quadratic in protein length:

| chunk | attempted allocation | implied protein length |
|---|---:|---:|
| bac_050 | 17.4 GiB | 48,298 aa |
| bac_005 | 46.0 GiB | 78,532 aa |
| **bac_064** | **4,373.8 GiB** | **766,188 aa** |

A typical bacterial protein is ~300 aa; the largest known is ~35,000. These are
pathological ORF calls, and three of the affected genomes carry no CDS
annotation at all. Not fixable by configuration — it is per-sequence, so
`--encoding-size` is irrelevant, and 4,374 GiB exceeds an 80 GB A100 by 50×.

Each failure is named in a per-chunk `<tag>.failures` file, with the traceback
in `<tag>.failure_logs`. A `--max-aa` guard in `genome_entropy`, refusing to
encode absurdly long ORFs, would eliminate the class.

---

## 8. Inodes were the binding constraint, not bytes

`gdata/ob80` had ~520,000 free inodes against 9 TB of free space. NCBI
`datasets` writes two inodes per genome (accession directory + `genomic.gbff`)
and the pipeline one JSON per genome:

| layout | inodes |
|---|---:|
| per-genome files on `/g/data` | ~600,000 — over quota |
| **jobfs + one archive per chunk** | **~10,800** |

The naive layout would have exhausted the quota on bacterial GenBank alone,
before a single entropy JSON was written. Staging per-genome work on node-local
`$PBS_JOBFS` and returning one compressed archive per chunk reduced this by
more than 50×.

Measured compression: entropy JSON is 2.3–4.9× the size of its GenBank input;
`zstd -3` compresses it ~3.4× at roughly ten times gzip's speed; and the
per-ORF entropy values extracted to TSV are **~42× smaller** than the JSON they
came from. Downstream analysis therefore reads the TSVs and never unpacks an
archive.

Final footprint: 1.5 TB results + 350 GB GenBank archives.

---

## 9. Cost

| stage | SU |
|---|---:|
| Smoke test + three calibration jobs | ~50 |
| Bacterial download (760 chunks) | 68 |
| Bacterial encoding (760 chunks) | ~80,000 |
| Counting, diagnosis | ~120 |
| **Total to date** | **~87,500 (8.8% of the 1 MSU grant)** |

`gpuvolta` charges ~36 SU per GPU-hour (measured: 4.41 SU for 7m21s on 1 GPU +
12 CPUs). Archaeal encoding is projected at ~4,000 SU.

---

## 10. Limitations

- Figures and the analyses in §5 and §6 rest on samples: 20 of 760 chunks for
  the figures, 2 chunks for the missed-gene analysis, 3 genomes for the 3Di
  composition. The log₂(k) ceilings are exact arithmetic and hold universally;
  the D/V/P proportions and the candidate counts are sample estimates.
- Chunks are contiguous slices of the GTDB accession list, which is not random
  with respect to taxonomy. Sampling every 17th chunk mitigates but does not
  eliminate taxonomic clustering.
- The 2.5 threshold in §6 was chosen by eye from the scatter. It sits well above
  the 1.585 ceiling, so the result is not an artefact of it, but the candidate
  count is threshold-sensitive.
- `in_genbank` reflects agreement between `get_orfs` calls and deposited CDS
  annotation. It is not ground truth about whether a sequence is a protein.
- Normalised entropy is derived, not stored; `05_aggregate_results.py` computes
  it only for the alphabets whose sizes the schema fixes.

---

## 11. Outstanding work

### Figures to add when these are regenerated

1. **Reference lines at the log₂(k) ceilings.** Add dotted horizontal lines at
   log₂(1) = 0, log₂(2) = 1, log₂(3) = 1.585 and log₂(4) = 2, labelled, so the
   1.585 boundary is visibly the three-state ceiling rather than an unexplained
   feature. This is the single most important addition — the figure currently
   shows the effect without naming its cause.
2. **Bivariate plot with marginal distributions** on the top and right axes
   (`seaborn.jointplot` or a manual `GridSpec`). The marginal 3Di histogram
   makes the bimodality and the hard edge at 1.585 explicit in a way the joint
   density alone does not; the marginal protein-entropy histogram shows how
   little separation that axis carries by comparison.

### Analyses

3. Regenerate all figures sampling **all 760 bacterial chunks** rather than 20.
4. Produce the **archaeal** counterparts and a bacteria-vs-archaea comparison;
   archaeal genomes are smaller (3.76 MB mean GenBank vs 6.28 MB) and less
   often annotated (40% vs 54% in samples).
5. Compute **fraction of `D` residues** per ORF as a length-independent
   alternative to 3Di entropy (§5.3).
6. **Run `12_genome_cds_counts.pbs`** and regenerate the annotation-presence
   figures (the 46% in §1/§6 and the 95.5% in §6) from CDS features in the
   GenBank records rather than from `in_genbank`. Both are currently upper
   bounds.
7. **Rerun `10_missed_genes.py`** now that it converts negative-strand
   coordinates onto the forward genomic axis, takes annotation presence from
   item 6, and tests shadows against deposited CDS coordinates from
   `13_cds_intervals.pbs`; replace the superseded shadow/intergenic numbers in
   §6. Needs chunk TSVs regenerated from the JSON archives, because the
   originals predate the `contig_length` column, and `13` run over the same
   chunks.
8. **Test what 3Di state `D` corresponds to.** Run DSSP over predicted
   structures, or a disorder predictor, on ORFs below and above the 1.585 line.
   The entropy ceiling is established without this, but the disorder reading of
   `D` dominance is not (§5).
9. Homology-search the recomputed missed-gene candidates (§6). The 8,651 figure
   is provisional until items 6 and 7 are done.
10. Stratify candidates by GTDB annotation provenance (§6).
11. Parallelise `05_aggregate_results.py` — 87 s per chunk single-threaded is
   ~18 h over 760 chunks. Each genome lives in exactly one chunk, so chunks are
   independent and this parallelises trivially.

### Upstream reports (not yet filed — both would post publicly)

12. `genome_entropy`: add a `--max-aa` guard (§7).
13. `gbouras13/modernprost-50M`: `_make_sliding_mask` materialises an L×L matrix
    for a banded mask (§7).
14. `genome_entropy download` prints a hardcoded `~/.cache/huggingface` path
    while correctly honouring `HF_HOME`, which is misleading for exactly the
    offline-cache workflow an air-gapped GPU node needs.

---

## 12. Reproduction

Pipeline in `/g/data/ob80/re3494/Projects/genome_entropy/claude/`:

| script | role |
|---|---|
| `00_smoke_test.sh` | login-node checks, 5-genome download |
| `00b_smoke_test_gpu.pbs` | GPU smoke test on real hardware |
| `00c/00d/00e_calibrate_*.pbs` | encoding-size and parallelism sweeps |
| `01_get_gtdb_reps.sh` | GTDB metadata → per-domain accession lists |
| `01b_make_chunks.sh` | split one domain into chunks, print the `-J` range |
| `02_download_genomes.pbs` | `copyq` array, strided, one archive per chunk |
| `03/03b_*` | environment and model cache |
| `04_run_entropy.pbs` | `gpuvolta` array, `PARALLEL=4`, archive + TSV per chunk |
| `05_aggregate_results.py` | per-genome summary, one domain at a time |
| `06_diagnose_failures.pbs` | re-run failed genomes, capture the cause |
| `07_count_orfs.pbs` | exact ORF / `in_genbank` counts, cached per chunk |
| `08_plot_entropy_scatter.py` | four-panel scatter |
| `09_plot_density.py` | hexbin and KDE figures |
| `10_missed_genes.py` | §6 analysis |
| `12_genome_cds_counts.pbs` | annotation presence from the GenBank records |
| `13_cds_intervals.pbs`, `cds_intervals.py` | deposited CDS coordinates for the §6 shadow test |

Gadi-specific PBS templates and install instructions were contributed upstream
on branch `feature/pbs-gadi-templates` of `linsalrob/genome_entropy`.

### Gadi constraints worth recording

- No queue has both GPUs and outbound internet; the model must be cached from a
  login node and jobs run with `HF_HUB_OFFLINE=1`.
- `max_array_size = 10`. Far smaller than `max_queued = 1000` suggests, and a
  wider array is refused outright. The download job strides chunks across
  subjobs; the GPU job uses `CHUNK_START` blocks.
- Array jobs must be submitted `-r y`; Gadi defaults to `-r n` and PBS refuses
  non-rerunable arrays.
- `CUDA_VISIBLE_DEVICES` is *unset* inside a GPU job — PBS restricts visibility
  by cgroup — so `genome_entropy` falls through to PyTorch's device count, which
  is correct. Do not set it by hand.
- `copyq` is capped at 10 hours for a 1-CPU job.
- PBS `-o` paths containing `^array_index^` collide when several arrays share an
  array index; include a per-array prefix.
