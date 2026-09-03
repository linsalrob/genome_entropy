# Structural-state entropy across GTDB representative genomes

**Dataset:** GTDB release 232 (R11-RS232, 15 April 2026)
**Model:** `gbouras13/modernprost-50M` — dual-head ModernProst, 3Di + 12-state
**Tool:** `genome_entropy` 0.2.0, output schema 2.2.0
**Platform:** NCI Gadi (PBS Pro), project `ob80`
**Report date:** 2026-09-03 — bacteria and archaea complete; §6 rewritten on the
corrected classification and the full candidate population

---

## 1. Summary

3Di and 12-state structural entropy was computed for every bacterial and
archaeal species representative in GTDB r232 — 199,837 genomes and 2.62 billion
ORFs. The headline analytical result is that the
**boundary at 3Di entropy ≈ 1.585 bits, visible as a sharp horizontal line
separating GenBank-matched from unmatched ORFs, is an information-theoretic
ceiling rather than a biological threshold**: it is exactly log₂(3), and it
arises because the encoder assigns a large ORF population to only three
distinct 3Di states. What those states correspond to structurally is not
established here — see the note under "Why encodings collapse to three
states" in §5. Section 5 establishes the ceiling quantitatively.

A secondary result (section 6) is that the population of unmatched ORFs with
high 3Di entropy — a candidate pool for genes missed by annotation — is
overwhelmingly explained by two confounders: genomes carrying no deposited CDS
at all (**92,840 of 189,715 bacterial genomes, 49%**, counted from the GenBank
records rather than inferred), and shadow ORFs overlapping a real CDS
(**94.5%** of the high-3Di unmatched pool inside annotated genomes). After both
are removed, **523,346 bacterial and 22,447 archaeal candidates remain** — an
85% reduction on the first, defective pass.

A Foldseek structural homology search over **every** surviving candidate (§6.1,
[issue #92](https://github.com/linsalrob/genome_entropy/issues/92),
[#97](https://github.com/linsalrob/genome_entropy/issues/97)) gives the third
result. Against a comparator of length- and 3Di-matched shadows with same-frame
shadows removed, full-length mutual coverage implies that **23–27% of bacterial
and 26–36% of archaeal candidates behave like real protein-coding genes** —
roughly **128,000–149,000 ORFs**. A conservative count requiring strong
individual evidence, rather than a mixture model, puts **~20,350** candidates
beyond reasonable doubt and **~65,800** with any full-length structural hit.

An independent check agrees and disagrees informatively. GTDB's own Prodigal
annotation — which consults neither GenBank nor any structure database — calls a
gene in the same frame at **47.4%** of bacterial and **55.4%** of archaeal
candidate loci, against **9.4%** and **16.0%** for matched shadows and
**94–96%** for annotated CDS. Put through the same mixture arithmetic that gives
**~244,000**, roughly **1.7× the structural estimate**. The two bracket the
answer; §6.1 sets out why they differ and which is conservative.

The results in the first draft of this section — 3,562,431 candidates, 95.5%
never-annotated, and "candidates and shadows are indistinguishable" — are
**withdrawn**. They rested on a coordinate defect that placed plus- and
minus-strand ORFs in different systems and on `in_genbank` as a proxy for
annotation. §6 records what replaced them and why the corrections mattered.

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

Archaea, encoded in full after the bacterial run:

| | count |
|---|---:|
| GTDB archaeal representatives | 10,122 |
| **Genomes encoded** | **10,122 (100%)** |
| **ORFs with entropy values** | **54,858,398** |
| Mean ORF calls per genome | 5,420 |

Archaeal ORF calls per genome are 40% of the bacterial figure, consistent with
smaller genomes. Their matched fraction is *higher* than bacteria's on both
denominators — `in_genbank = True` for 17.26% of all archaeal ORFs against
12.03%, and 27.72% within annotated genomes against 21.05% — and 54.3% of
archaeal representatives carry CDS annotation against 51.1% of bacterial ones.
An earlier version of this report, working from a mid-run sample, stated the
opposite; the sampled estimate was wrong.

`in_genbank` — whether a called ORF was matched to an annotated CDS by genomic
overlap, frame and translation:

| | count | share |
|---|---:|---:|
| `in_genbank = True` | 309,065,661 | 12.03% |
| `in_genbank = False` | 2,259,179,323 | 87.97% |

The low matched fraction is expected and is not an error rate. `get_orfs`
enumerates open reading frames in all six frames, so most calls are not genes.
Separately, **48.9% of bacterial representatives carry no CDS annotation at
all** (GCA assemblies submitted unannotated), and every ORF in those genomes is
`False` by construction. Any analysis of what `False` *means* must exclude them;
see §6.

That figure was previously reported as an upper bound, because it came from
`in_genbank`, which cannot distinguish a genome with no annotation from one
whose annotations the ORF matcher rejected. `12_genome_cds_counts.pbs` now
counts CDS features in the GenBank records directly, and **the two agree to
100.000% at genome level over all 189,715 genomes** — 92,840 carry no deposited
CDS. The figure is exact, and the concern is discharged rather than confirmed:
the proxy was right, but that could not be known without counting.

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

Restricted to genomes carrying at least one annotated CDS (96,875 of 189,715
bacterial representatives, 51.1%), matched and unmatched ORFs occupy
substantially different regions of (protein entropy, 3Di entropy) space:

- matched ORFs peak near (4.05, 3.6)
- unmatched ORFs peak near (3.8, 1.35)

The separation is almost entirely in **3Di**, not protein, entropy: unmatched
ORFs have plausible amino-acid composition but structurally monotonous 3Di
strings. Mean 3Di entropy across 250 genomes was 3.153 for matched ORFs against
1.818 for all ORFs.

**Every figure in this report is restricted to genomes carrying at least one
annotated CDS, and that restriction is not cosmetic.** 48.9% of bacterial and
45.7% of archaeal representatives have no CDS annotation at all, so every ORF in
them is `in_genbank = False` whatever it is. Those rows sit overwhelmingly in the
low-3Di band, and including them inflates the unmatched class with ORFs that
carry no information about whether they are genes — making the separation look
cleaner than the evidence supports. Removing them discards 42.8% of the sampled
bacterial rows (3,667,250 of 8,560,442) and 37.7% of the archaeal (689,879 of
1,828,593), and raises the matched fraction from 12.01% to 21.01% in bacteria and
from 17.26% to 27.72% in archaea. Earlier drafts of these figures mixed the two
populations; they have been withdrawn to
`figures/superseded_20chunk/`. `08_plot_entropy_scatter.py` and
`09_plot_density.py` can still draw the unfiltered view with
`--include-unannotated`, which stamps the figure as a diagnostic — it is useful
for showing what the confounder does, and not a result.

What the filter does *not* remove is the low-3Di population itself. In both
domains a large mode remains below log₂(3) among unmatched ORFs of annotated
genomes, so that population is a real feature of six-frame ORF calling and not
an artefact of unannotated assemblies.

### Figures

All figures live in [`figures/`](figures/) and are committed alongside this
report. The entropy panels below were regenerated over **every** chunk of both
domains rather than the 20 bacterial chunks that existed while the run was in
flight; the §6 figures are written by `28_report_figures.py`, which reads the
TSVs the analysis stages emit rather than recomputing anything — a plotting
script that reimplements the analysis is a second implementation that will drift
from the first.

![Joint density of protein against 3Di entropy for bacteria, with marginal
histograms](figures/protein_vs_3di_joint_bac.png)

*Bacteria: joint density with marginals. The 3Di margin is a histogram rather
than a KDE, deliberately — smoothing rounds off the very edge the figure exists
to show.*

![The same joint density for archaea](figures/protein_vs_3di_joint_arc.png)

*Archaea, 47-fold fewer ORFs, same structure.*

![Bacteria against archaea on one axis](figures/protein_vs_3di_domains.png)

*Both domains on the same axis: the 1.585 edge falls in the same place, which is
what an information-theoretic ceiling predicts and a biological threshold does
not.*

| file | content |
|---|---|
| [`protein_vs_3di_entropy_bac.png`](figures/protein_vs_3di_entropy_bac.png) / [`_arc.png`](figures/protein_vs_3di_entropy_arc.png) | scatter, 4 panels |
| [`protein_vs_3di_hexbin_bac.png`](figures/protein_vs_3di_hexbin_bac.png) / [`_arc.png`](figures/protein_vs_3di_hexbin_arc.png) | hexbin, log counts |
| [`protein_vs_3di_kde_bac.png`](figures/protein_vs_3di_kde_bac.png) / [`_arc.png`](figures/protein_vs_3di_kde_arc.png) | KDE + contour overlay |
| [`protein_vs_3di_joint_bac.png`](figures/protein_vs_3di_joint_bac.png) / [`_arc.png`](figures/protein_vs_3di_joint_arc.png) | joint density with marginals |
| [`protein_vs_3di_domains.png`](figures/protein_vs_3di_domains.png) | bacteria against archaea |

Every panel with a 3Di axis carries dotted log₂(k) reference lines, so the
1.585 boundary reads as the three-state ceiling of §5 rather than as an
unexplained feature.

The sample is systematic — every 300th bacterial ORF row (8,560,442 rows,
12.01% matched against a population 12.03%) and every 30th archaeal
(1,828,593 rows, 17.26%, exact). Strides differ because the domains differ
47-fold in size; counts are therefore not comparable between domains, while
distribution shapes are. Written by `08b_sample_for_figures.pbs`; samples live
in `figure_samples/`, not in `figures/`.

The **joint figure is the one to read first.** Its marginal 3Di histogram — a
histogram rather than a KDE, deliberately, because smoothing rounds off the very
edge the figure exists to show — makes the bimodality explicit: unmatched ORFs
pile against 1.585 with almost nothing above it while matched ORFs peak near
3.6, and a distinct spike sits at log₂(1) = 0 for single-state encodings. The
marginal protein-entropy panel shows by contrast how little that axis separates
the classes.

The four-panel scatter exists to make one methodological point: panels C and D
hold identical data in opposite draw order, and the two readings differ
completely. With 220,105 unmatched points drawn over 29,895 matched, whichever
class is drawn last wins. Neither ordering is neutral — drawing the 12% class
last overstates it just as burying it understates it — which is the argument for
the binned and smoothed views. The scatter panels draw a 250,000-row random
subsample of the systematic sample, since a mark per point stops carrying
information long before 8.5 million of them; the binned and smoothed figures use
every sampled row.

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

### The archaeal data confirms it independently

`protein_vs_3di_domains.png` puts both domains on the same axis. The edge falls
at exactly the same place in archaea as in bacteria, and it must: log₂(3) is a
property of a three-symbol alphabet, not of an organism. Two clades separated by
billions of years of evolution, encoded from different genomes with different
GC content and coding density, produce a boundary in the same position to the
resolution of the histogram. No biological threshold behaves that way. Archaea
do differ in how the mass is distributed — proportionally more of it in the
high-3Di mode near 3.6, less in the low band — which is a real biological
difference sitting on top of an artefactual boundary, and a good illustration of
why the two have to be separated before either is interpreted.

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

Hypothesis tested: unmatched ORFs with 3Di entropy above ~2.5 look structurally
like real proteins, so perhaps the original annotation software failed to call
them.

> **The numbers in the first draft of this section are withdrawn.** Two defects
> produced them, both since fixed, and the corrections changed the answer by
> more than a factor of six.
>
> *Coordinates.* Overlap was tested on raw `get_orfs` coordinates. On the
> negative strand those index the reverse complement, so every plus-versus-minus
> comparison crossed two coordinate systems. Demonstrated on a fixture: a
> minus-strand ORF at 601–900 on a 1,000 bp contig normalises to [100, 400) and
> is a **shadow**; one at 100–400 normalises to [600, 901) and is a
> **candidate**. The old code returns exactly the inverse — the two arms swap.
> Both sides are now placed on the forward genomic axis with
> `normalise_orf_interval`, the helper the `in_genbank` matcher already used.
>
> *What counts as annotated.* The CDS set was taken to be the spans of ORFs
> whose `in_genbank` flag was True, which misleads in both directions: a CDS the
> matcher rejected is absent, and a matched ORF runs stop to stop and can extend
> past the deposited CDS. The test now uses deposited GenBank coordinates from
> `13_cds_intervals.pbs`, and annotation presence comes from
> `12_genome_cds_counts.pbs` counting CDS features directly.
>
> The 760-chunk rerun did not escape either defect — it applied the same
> classification everywhere, so agreement between scales showed the method was
> stable, not correct.

### What the corrected classification finds

| | bacteria | archaea |
|---|---:|---:|
| genomes | 189,715 | 10,122 |
| ORF rows | 2,568,244,984 | 54,858,398 |
| genomes with **no deposited CDS at all** | 92,840 (49%) | 4,630 (46%) |
| unmatched ORFs in annotated genomes | 1,158,990,307 | 24,692,953 |
| → 3Di ≥ 2.5 | 9,598,500 | 307,917 |
| → → **shadow** of a deposited CDS | 9,075,154 (**94.5%**) | 285,470 (**92.7%**) |
| → → **intergenic — candidates** | **523,346** (5.5%) | **22,447** (7.3%) |

![Funnel from all ORF calls to candidates, both domains](figures/candidate_funnel.png)

*Each step removes a population that cannot inform the question: ORFs in genomes
with no annotation at all, ORFs the matcher accepted as CDS, low-entropy calls,
and finally shadows of real genes.*

The bacterial candidate pool falls from **3,562,431 to 523,346**, an 85.3%
reduction, and the "99.8% of annotated genomes carry a candidate" figure falls
to 83.7% (81,043 of 96,875). ORF counts conserve exactly across the split
(bacteria 1,468,055,968 + 1,100,189,016 = 2,568,244,984; archaea 34,161,355 +
20,697,043 = 54,858,398), and the shadow/candidate split closes exactly in both
domains.

**The 48.9% never-annotated figure is now exact, not an upper bound.** The
`in_genbank` proxy and the authoritative CDS counts agree to 100.000% at genome
level over all 189,715 bacterial genomes, so the concern that the proxy
over-counted unannotated genomes is discharged — it was right all along, but
that could not be known without counting.

### A boundary sentinel that would have aborted every chunk

`get_orfs` emits `end = contig_length + 1` for an ORF running off the end of a
contig, and its `dna.length` switches to an exclusive convention for exactly
those rows. Unclamped, they place outside the contig and every derived
coordinate is nonsense. The signature is unambiguous: **100% of such ORFs have
no stop codon (28,728 of 28,728) against 0% of complete ORFs (0 of 1,189,718)**.
They are 1.55% of bacterial and 1.68% of archaeal ORFs, are 3× enriched in the
shadow arm, and hit structural databases at ~2–2.5× the rate of complete ORFs —
so they are excluded from every direct-evidence count rather than merely
flagged.

### The entropy-matched intergenic arm

The original negative control, `intergenic_lo`, is low-3Di **by construction**,
so its rate is confounded with the very axis under test. `unannot_hi` replaces
it: high-3Di ORFs from genomes carrying no CDS at all (210,053,658 bacterial,
5,851,244 archaeal). It is not a non-coding floor — those genomes are
unannotated because no pipeline ran, not because one ran and found nothing — so
it asks whether the assay recovers gene-like ORFs where annotation is simply
absent.

It answers yes, and the asymmetry replicates across domains: among unmatched
ORFs the 3Di ≥ 2.5 rate is **0.83% inside annotated genomes against 19.09% in
unannotated ones** (bacteria, 23.1×) and **1.25% against 28.27%** (archaea,
22.7×). Two independently processed domains giving the same 23-fold ratio is
about as strong a check on that reading as this dataset allows.

### An independent verification of the split

`20_orf_context.py` recomputes CDS overlap from the deposited interval tables
without reference to the classifier. Candidates are *defined* as not touching a
deposited CDS and shadows as touching one, so disagreement would mean the two
disagree about where ORFs sit — the exact failure that produced the withdrawn
results. **All 760 bacterial and all 41 archaeal chunks agree**, over all
1,502,473 and 112,235 ORFs in the query sets.

### The fragmentation confound was itself an artefact

The first draft reported that burden tracks assembly quality rather than
biology, with Spearman ρ of −0.603 against contig N50 — the strongest
correlation in the analysis, and the basis for a planned programme of
assembly-stratified sensitivity sets. Recomputed on the corrected candidates:

| covariate | bacteria, confounded | bacteria, corrected | archaea |
|---|---:|---:|---:|
| n50_contigs | **−0.603** | **+0.100** | +0.142 |
| contig_count | +0.547 | +0.117 | +0.074 |
| n_orfs | — | **0.568** | **0.526** |
| genome_size_mb | — | 0.553 | 0.514 |
| coding_density | −0.040 | −0.270 | −0.434 |
| frac_orfs_in_genbank | — | −0.429 | −0.379 |

N50 does not merely weaken, **it changes sign**. The mechanism is coherent: a
fragmented assembly has more contig ends, more contig ends means more truncated
ORFs, and under un-normalised coordinates a minus-strand ORF near a contig end
was placed wrongly and fell in the wrong arm. Burden correlated with
fragmentation because *the bug* correlated with fragmentation. What burden now
tracks is ORF count, genome size and low coding density — which is what a
missed-gene signal should look like.

![Candidate burden against genome covariates,
bacteria](figures/candidate_burden_bac.png)

![The same for archaea](figures/candidate_burden_arc.png)

### Sensitivity of the ORF caller

Falling out of the CDS-count work: a median **161 deposited CDS per archaeal
genome have no matching six-frame ORF** (1,502 matched of 1,670 deposited,
~90% recall). Compound and multi-part CDS, CDS below the length floor, and
non-standard starts account for it. This does not affect any candidate count —
candidates are ORFs by construction — but it puts a floor under any claim of
the form "we would have found it if it were there", and it belongs with the
limitations in §10.

### 6.1 Structural homology over the full candidate population

The pilot searched 8,712 bacterial candidates drawn from 13 of 760 chunks —
**1.7% of the population** — and extrapolated. Archaea used all 22,447. This
section reports the run over **every** candidate in both domains: 1,502,473
bacterial and 112,235 archaeal query ORFs, four target databases, both
structural and sequence modes, 946 SU.

**Design.** Five arms. `candidate`; `shadow_hi`, matched 1:1 on length *and* 3Di
entropy preferring a partner from the same genome (94.0% achieved), which holds
lineage, GC, assembly quality and annotation pipeline fixed at once;
`annotated_cds` and `intergenic_lo` matched on length; `unannot_hi` matched on
length and 3Di. A sixth query set — the candidates' own 3Di strings shuffled —
is the technical null, identical in amino acids, length and 3Di composition but
with residue order destroyed. Control arms are capped at 200 per chunk because a
control only has to pin down a *rate*; that also improves match quality, since
200 draws from a pool of 2,000 find closer partners than 633 do.

**The technical null holds at 60× the pilot's n.** Over all 523,346 bacterial
candidates the shuffled-3Di search returns 156 / 224 / 35 / 4 hits against
Swiss-Prot / PDB100 / CATH50 / BFVD — at most 0.04%.

#### The comparator had to be fixed first

A same-strand, same-frame shadow largely **is** the annotated protein, so it
scores like a real gene and must be excluded from the comparator. Classifying
frame correctly turned out to be harder than it looks.

A plus-strand feature is translated left to right, so its frame is
`g_start % 3`; a minus-strand feature is translated right to left, so its frame
is `g_end % 3`. Verified against eleven hand-built cases, **anchoring on
`g_start` for both strands is correct on plus and wrong on minus**, and a
3′-anchored test is exactly the reverse; only the strand-dependent anchor passes
all eleven.

The error was nearly invisible because the two anchors give identical verdicts
whenever both lengths divide by three — 100% of complete ORFs, 95.6% of
deposited CDS parts, and empirically 100.00% of such pairs. It fires on
**partial CDS**: features with fuzzy ends (`<1..500`), which are 3.9% of CDS
parts genome-wide but **36.6%** of those overlapped by same-strand shadows, a 9×
enrichment. A partial CDS has no true codon boundary, so *neither* anchor is
meaningful, and forcing such a pair into same-frame or frameshift is a coin
flip — which was scattering genuinely same-frame shadows into the clean
comparator and contaminating the background with ORFs that are the annotated
protein. Those pairs now get their own `frame undefined` class and are excluded.

The control that has to come out one way does: `annotated_cds` is **92.0%
same-frame plus 7.5% frame-undefined**, 99.5% accounted for. Independently, an
external gene caller agrees with **95.2%** of bacterial and **94.1%** of
archaeal same-frame shadows and only **4.4%** and **8.1%** of antisense ones — a
22× and 12× discrimination.

Bacterial shadows resolve to 43.5% opposite strand, 31.5% frame undefined, 12.6%
same frame and 12.4% frameshift.

#### Full-length mutual coverage is the readout that discriminates

Requiring the best hit to cover ≥80% of both query and target, against the clean
comparator (antisense and frameshift shadows only):

| database | candidate | clean shadow | annotated | real-gene share (95% CI) |
|---|---:|---:|---:|---:|
| **bacteria** | | | | |
| afdb_swissprot | 10.3% | 2.7% | 35.3% | **23.3% (23.0 – 23.7)** |
| cath50 | 8.4% | 2.7% | 23.9% | **27.0% (26.5 – 27.5)** |
| pdb100 | 8.7% | 2.2% | 30.1% | **23.4% (23.1 – 23.8)** |
| bfvd | 6.8% | 2.0% | 11.0% | 54.2% — discounted |
| **archaea** | | | | |
| afdb_swissprot | 14.2% | 6.4% | 35.9% | **26.2% (24.3 – 28.2)** |
| pdb100 | 13.3% | 5.7% | 32.6% | **28.3% (26.3 – 30.4)** |
| cath50 | 12.7% | 5.5% | 25.3% | **36.4% (33.7 – 39.2)** |
| bfvd | 9.2% | 3.6% | 11.6% | 70.2% — discounted |

![Full-length coverage by arm and database, with the implied real-gene
share](figures/candidate_coverage.png)

BFVD is excluded from estimation because its annotated ceiling is ~11%, so the
mixture denominator is small and the ratio unstable. It is retained for
interpretation: a BFVD-only hit suggests phage or prophage biology.

**23–27% (bacteria) and 26–36% (archaea)** implies roughly **122,000–141,000**
and **5,900–8,200** ORFs, or **~128,000–149,000** across both domains.

The pilot was wrong in both directions, which is instructive about 1.7% samples.
On the *uncorrected* comparator the full-scale shares come in **lower** than the
pilot's (6.6 / 10.4 / 6.5% against 10.4 / 13.0 / 7.9%); on the corrected one they
come in **higher** (23.3 / 27.0 / 23.4% against 15.8 / 19.7 / 13.9%). Confidence
intervals tighten from ±3 points to ±0.4.

Hit rate remains the wrong readout — above 3Di 3.38 bits everything hits at 99%,
real gene and unmatched ORF alike, because a sequence-only 3Di encoder emits
confident 3Di for anything with protein-like composition. That finding from the
pilot is unchanged.

#### A conservative count that needs no mixture model

Every criterion is reported against the **matched clean shadows**, scaled for
the unequal arms. Shadows overlap real genes and clear these bars too, so the
excess is the floor — the candidate column is not.

| criterion | bac candidates | bac excess | arc candidates | arc excess |
|---|---:|---:|---:|---:|
| C1 all candidates | 523,346 | — | 22,447 | — |
| C2 not contig-truncated | 483,767 | — | 19,835 | — |
| C3 C2 + any full-length structural hit | 90,658 | **62,507** | 5,141 | **3,256** |
| C4 C2 + full-length in ≥2 databases | 44,325 | 33,226 | 2,777 | 1,872 |
| C5 C2 + full-length Swiss-Prot/PDB/CATH | 76,662 | 53,383 | 4,396 | 2,749 |
| C6 C2 + structure-only, full-length | 81,257 | 54,410 | 4,439 | 2,717 |
| C7 strict combined rule | 23,540 | **19,232** | 1,574 | **1,118** |

C7 requires: not truncated, full-length in ≥2 databases including at least one
of Swiss-Prot/PDB/CATH, an interpretable product name, ≥100 aa, E < 10⁻¹⁰.

![Direct evidence: candidates against matched clean
shadows](figures/evidence_ladder.png)

**These are a set, not a nested ladder** — C6 (4,439) exceeds C4 (2,777) in
archaea, so reading down the column as increasing stringency is wrong. C3–C6 are
each C2 plus one independent requirement; C7 is the conjunction.

So: **~20,350 candidates with strong individual evidence** (19,232 + 1,118) and
**~65,800 with any full-length structural hit** (62,507 + 3,256). These are the
numbers that survive without the two-component assumption.

One structural caveat on C7: it requires a Swiss-Prot/PDB/CATH hit, so
**BFVD-only candidates are invisible to it by construction** — 13,996 bacterial
candidates (15.4% of C3) have a full-length viral hit and nothing else. The
strict floor therefore systematically excludes viral biology.

#### An independent test: does another gene caller agree?

GTDB runs Prodigal over every representative genome, consulting neither GenBank
nor any structure database. Extracting its coordinates (618,638,921 gene calls
over 32,366,333 contigs) allows an ORF-level test: does a candidate coincide,
in the same frame and strand with an exactly matching 3′ end, with a Prodigal
gene that GenBank does not carry?

The coordinate conventions were **calibrated rather than assumed** — the offset
distribution between our 3′ ends and the nearest Prodigal 3′ end peaks at
exactly **0 bp** in both domains. A silent 3 bp stop-codon difference would have
produced zero matches and the false conclusion that Prodigal disagrees.

| arm | bacteria | archaea |
|---|---:|---:|
| **annotated_cds** (positive control) | **94.21%** | **96.13%** |
| **candidate** | **47.42%** | **55.36%** |
| clean shadow (3Di-matched, occupied space) | 9.41% | 16.00% |
| `intergenic_lo` (unoccupied space) | 9.85% | 15.44% |

![Prodigal agreement by arm, and by shadow frame
class](figures/prodigal_validation.png)

Two objections fail against their own controls. First, an antisense shadow sits
inside a gene Prodigal already calls, and gene callers avoid overlapping
predictions — so its low rate might reflect a prior commitment rather than the
ORF looking non-coding. But `intergenic_lo` sits in unoccupied space with no
competing call and gives the **same rate to within half a point** in both
domains. Second, both methods might merely favour coding-like composition — but
`shadow_hi` is matched to each candidate on 3Di entropy, so at equal apparent
structure Prodigal still separates candidate from shadow **5-fold**.

Taking the larger background, the excess is **37.6%** (bacteria) and **39.4%**
(archaea), implying **196,638** and **8,836** candidates independently called.
Through the same mixture arithmetic as the structural assay:

| | share (95% CI) | implied ORFs |
|---|---:|---:|
| bacteria | **44.5% (44.3 – 44.7)** | **233,074** |
| archaea | **49.1% (48.2 – 50.0)** | **11,026** |

#### How much annotation GTDB itself adds, and a correction

A genome-level version of the same question: GTDB's Prodigal calls exceed the
deposited GenBank CDS count in **88.3% of annotated archaeal and 78.4% of
bacterial genomes**, by a median of **30 and 29 genes** respectively.

> **A correction.** That gap was first reported here and on
> [#97](https://github.com/linsalrob/genome_entropy/issues/97) as a median of
> **193 genes**, with Prodigal exceeding GenBank in 99.9% of genomes. Both were
> wrong, through a column-name collision: `candidate_burden_*.tsv` renames
> `n_orfs_in_genbank` to `n_cds`, so that column counts *ORFs that matched a
> CDS*, while `genome_cds_counts_*.tsv` — also called `n_cds` — counts
> *deposited CDS features*. Differencing Prodigal against the first gives the
> real gap (30) plus the ORF-matching loss (161). Two tables, two quantities,
> one column name.

The corrected figure is much weaker evidence than the published one, and the
genome-level argument should not be leaned on: Spearman ρ between candidate
burden and the corrected gap is **0.119 (archaea) / 0.035 (bacteria)** —
essentially nothing. Only the ORF-level test above carries weight.

#### Where the two estimates disagree

Structural homology says ~128,000–149,000; independent gene calling says
~244,000. **They differ by ~1.7× and this report does not resolve it.** Three
explanations, not separable with what is here:

1. **Prodigal over-calls**, which it is documented to do. The background is
   measured on ORF-shaped sequence, not random DNA, so 9.85% is its
   false-positive rate on non-gene ORFs and the rate on genuinely non-coding
   sequence could be higher. A 20% background would drop the bacterial share to
   ~37%.
2. **The structural mixture underestimates.** It assumes candidates are a
   mixture of annotated-CDS-like and shadow-like behaviour, and real missed
   genes are plausibly shorter, more divergent and worse represented in PDB and
   Swiss-Prot than the average annotated CDS — precisely because those
   properties are why annotation missed them. The structural assay's ceiling on
   *known* genes is only 30–35%, against Prodigal's 94–96%, so it works much
   closer to its detection floor.
3. Both.

The two should be reported as bracketing the answer, with the structural figure
the more conservative. The direct-evidence floor of ~20,350 is unaffected by
either.

**The two lines of evidence converge on the same ORFs, however.** Prodigal
confirms **47.4%** of all bacterial candidates but **91.8%** of those meeting
the strict structural rule, and **95.9%** of the archaeal strict set, against
**9.4%** for matched shadows. The strict rule selects for Prodigal agreement
without ever using Prodigal.

#### What the supported candidates are

Functional class of the strict sets, from Swiss-Prot and PDB product text (only
those two databases carry free text; CATH gives a superfamily code and BFVD a
bare accession, so `unclassified` reflects the reference database, not weak
evidence):

| class | bacteria (n = 23,540) | archaea (n = 1,574) |
|---|---:|---:|
| **metabolism** | **33.8%** | **41.3%** |
| other named | 25.4% | 20.3% |
| **mobile element** | **13.6%** | **11.8%** |
| transport | 6.2% | 3.9% |
| uncharacterized | 5.9% | 7.3% |
| translation | 5.7% | 8.6% |
| regulation | 4.6% | 1.9% |
| replication / repair | 2.0% | 2.7% |
| defence | 1.6% | 1.0% |
| cell envelope | 1.2% | 1.3% |

![Functional composition of the strict-evidence
sets](figures/functional_composition.png)

The most frequent *individual families* are mobile-element proteins —
resolvases, recombinases, integrases, homing endonucleases, intron-encoded
proteins — which is a correct result rather than an artefact, since those are
genuinely omitted from GenBank annotation. But they are only 12–14% of the
supported set, while metabolic enzymes are 34–41%. There are 8,065 distinct
Swiss-Prot products among 23,393 annotated bacterial strict candidates, with the
top ten covering 10.0%.

#### Where the separation lives

Stratifying the archaeal coverage contrast by assembly quality shows the
candidate-versus-shadow separation is **not uniform across genomes**. In the top
contig-N50 quartile candidates reach 43.5% against shadows at 32.6% — an
11-point gap — while in the bottom quartile the two are indistinguishable
(65.8% against 65.5%).

This argues *against* the separation being an assembly artefact: an artefact of
fragmentation would appear most strongly in the worst assemblies, and it appears
only in the best. It also gives a positive reason to draw manuscript examples
from high-completeness, high-N50 genomes — not merely to avoid artefacts, but
because that is where the signal demonstrably is. A candidate in a poorly
assembled genome carries little evidential weight however good its structural
hit.

#### Examples

Ten were selected by walking functional classes rather than by score, which is
what stops ten Bacteroidota TonB-dependent receptors filling every slot. Each is
non-truncated, lies entirely between two deposited CDS, is ≥100 aa, sits in a
host ≥95% complete with ≤5% contamination and ordinary candidate burden, and has
full-length support in all four databases. Dossiers are in
`missed_genes/dossiers/`.

![Genomic context of the ten manuscript examples](figures/exemplar_loci.png)

*Each panel is a window around one candidate. Grey arrows are deposited CDS;
the outlined coloured arrow is the candidate, filled by functional class.*

Two are genuinely incomplete modules:

- **Cas5 completing a type I CRISPR–*cas* operon.** The deposited annotation
  carries Cas6, Cas7, Cas8a1, Cas3′, Cas1 and Cas2, co-oriented, and **no
  Cas5** — which a type I Cascade complex requires. The candidate is Cas5, 226
  aa, 5 bp from Cas7 and 96 bp from Cas1, filling the one gap in an otherwise
  contiguous operon.
- **SusC completing a polysaccharide utilisation locus.** SusD, SusE,
  alpha-amylase and neopullulanase are annotated; SusC, the TonB-dependent
  transporter the SusC/SusD pair cannot function without, is absent. The 917 aa
  candidate fills the 3,029 bp gap.

Examples illustrate the biology; the **rate** comes from the full-population
comparisons above. Selecting examples on biological coherence and then citing
them as evidence for the rate would be circular, and they are not used that way.

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
| Archaeal encoding (41 chunks) | ~4,000 |
| Counting, diagnosis, figures | ~120 |
| Regenerating `entropy_rows` with `contig_length` | ~130 |
| CDS counts and intervals | ~65 |
| Corrected classification, both domains | ~30 |
| Foldseek pilots (bacterial, archaeal) | ~70 |
| **Full-population Foldseek search** (4 shards, 1.5 M queries) | **946** |
| Context, ranking, classification, examples, dossiers | ~40 |
| GTDB Prodigal coordinates (123 GiB streamed) | 27 |
| **Total** | **~85,550 (8.6% of the 1 MSU grant)** |

`gpuvolta` charges ~36 SU per GPU-hour (measured: 4.41 SU for 7m21s on 1 GPU +
12 CPUs). **Encoding dominates by two orders of magnitude**: everything in the
missed-gene analysis, including searching 1.5 million queries against four
structural databases, is ~1.3% of the total.

Wasted, and recorded because both were avoidable: **61.57 SU** on a staging
path that wrote plain text where gzip was expected (the smoke test exercised the
final filename, not the staging filename), and **13.35 SU** on a job that parsed
618,638,921 rows successfully and then destroyed its own output when a
verification step exceeded the jobfs quota and the cleanup trap fired.

---

## 10. Limitations

- Figures still rest on samples — 20 of 760 chunks, and 3 genomes for the 3Di
  composition. **§6 no longer does**: the classification covers all 760
  bacterial and 41 archaeal chunks, and the Foldseek search covers every
  candidate in both domains. The log₂(k) ceilings are exact arithmetic; the
  D/V/P proportions remain sample estimates.
- Chunks are contiguous slices of the GTDB accession list, which is not random
  with respect to taxonomy. Sampling every 17th chunk mitigates but does not
  eliminate taxonomic clustering in the figures.
- The 2.5 threshold in §6 was chosen by eye from the scatter. It sits well above
  the 1.585 ceiling, so the result is not an artefact of it, but the candidate
  count is threshold-sensitive.
- `in_genbank` reflects agreement between `get_orfs` calls and deposited CDS
  annotation. It is not ground truth about whether a sequence is a protein.
- **The ORF caller misses ~10% of deposited CDS** (median 161 per archaeal
  genome), so any statement of the form "we would have found it if it were
  there" carries that floor.
- **The mixture model assumes two components.** Candidates are treated as a
  mixture of annotated-CDS-like and shadow-like behaviour. A real missed gene
  need not be either, and the confidence intervals cover binomial noise only —
  not the two-component assumption. The ~1.7× disagreement between the
  structural and Prodigal estimates (§6.1) is the visible consequence.
- **Prodigal agreement is independent of GenBank and of structural homology,
  but not of coding-like composition.** The 3Di-matched shadow arm bounds that
  shared prior; it does not eliminate it.
- Manuscript examples are chosen for biological diversity and are illustrations,
  not evidence for the rate.
- Normalised entropy is derived, not stored; `05_aggregate_results.py` computes
  it only for the alphabets whose sizes the schema fixes.

---

## 11. Outstanding work

### Done since the first draft

- Dotted **log₂(k) reference lines** on every 3Di axis, labelled (§4).
- **Joint figure with marginal distributions**, the 3Di margin as a histogram
  so the 1.585 edge is not smoothed away (§4).
- All figures regenerated over **all 760 bacterial chunks**, plus **archaeal
  counterparts and a bacteria-vs-archaea comparison** (§4, §5). Archaeal genomes
  are smaller, as expected — 5,420 ORF calls per genome against 13,537 — but the
  claim that they are *less often annotated* was an artefact of a mid-run sample
  and is corrected in §2: 54.3% against 51.1%.
- **`12_genome_cds_counts.pbs` run over both domains.** The 48.9% never-annotated
  figure is now exact rather than an upper bound, and agrees with the
  `in_genbank` proxy to 100.000% at genome level (§2).
- **`10_missed_genes.py` rerun on the corrected coordinates and deposited CDS
  intervals**, over all 760 bacterial and 41 archaeal chunks. Bacterial
  candidates fall from 3,562,431 to 523,346 (§6).
- **The Foldseek search rerun over the full candidate population**, not a 1.7%
  sample, in both domains — with the same-frame-shadow correction, the
  direct-evidence ladder, the independent Prodigal comparison, functional
  composition, ten manuscript examples and their dossiers (§6.1).
- **The reading-frame definition settled** against hand-built ground truth, and
  the partial-CDS case that made it wrong identified (§6.1).
- The fragmentation confound **retired**: ρ(N50) moves from −0.603 to +0.100 on
  the corrected candidates, so the assembly-stratified sensitivity sets that
  were planned to control it are no longer needed (§6).

### Analyses

5. Compute **fraction of `D` residues** per ORF as a length-independent
   alternative to 3Di entropy (§5.3).
6. **Resolve the structural-versus-Prodigal disagreement** (§6.1). The two
   estimates differ by ~1.7× and this report brackets rather than resolves them.
   The missing measurement is Prodigal's false-positive rate on sequence that is
   confidently non-coding, which our arms cannot supply — every one of them is
   ORF-shaped by construction.
7. **Test what 3Di state `D` corresponds to.** Run DSSP over predicted
   structures, or a disorder predictor, on ORFs below and above the 1.585 line.
   The entropy ceiling is established without this, but the disorder reading of
   `D` dominance is not (§5).
8. **Measure neighbourhood coherence as a statistic**, not only as an example
   selection criterion. `cds_intervals/` plus the candidates' genomic
   coordinates allow gap occupancy and strand coherence to be compared against
   matched controls — a line of evidence independent of Foldseek, the structural
   databases and the mixture model. Note the trap found while scoping it: raw
   neighbour *distance* cannot be compared between candidates and shadows,
   because a shadow overlaps a CDS by definition and its nearest
   non-overlapping neighbour is measured past the far end of that gene.
   **Regenerate the context tables first.** The copies on `/g/data` predate a
   fix to `up_strand`/`up_cds_id`, which for a CDS nested inside an earlier
   longer one were read from a different gene than the one `dist_up` measures
   to. Incidence is 10 of 2,080,688 archaeal CDS parts and none of the ten
   exemplars, so nothing published depends on it — but strand coherence is
   computed from exactly those columns.
9. Stratify candidates by **GTDB annotation provenance** (§6). If candidates
   concentrate in older Prokka versions rather than recent PGAP, that is direct
   evidence of pipeline misses.
10. Parallelise `05_aggregate_results.py` — 87 s per chunk single-threaded is
    ~18 h over 760 chunks. Each genome lives in exactly one chunk, so this
    parallelises trivially. Consequently the **bacterial** per-genome summary has
    never been produced; only the archaeal one exists
    (`summary_per_genome_arc.tsv`, 41 chunks in 6.5 min).
11. **modernprost-50M versus ProstT5** 3Di comparison. Framed in the first draft
    as the validation that decides everything; the shuffled-3Di null has since
    done that job directly, so this is characterisation rather than a gate.

### Upstream reports (not yet filed — both would post publicly)

13. `genome_entropy`: add a `--max-aa` guard (§7).
14. `gbouras13/modernprost-50M`: `_make_sliding_mask` materialises an L×L matrix
    for a banded mask (§7).
15. `genome_entropy download` prints a hardcoded `~/.cache/huggingface` path
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
| `05b_aggregate_results.pbs` | 05 under the scheduler, an hour of CPU per domain |
| `08b_sample_for_figures.pbs` | systematic entropy sample over every chunk |
| `08_plot_entropy_scatter.py` | four-panel scatter |
| `09_plot_density.py` | hexbin, KDE, joint-with-marginals, domain comparison |
| `figstyle.py` | shared sample loader and log₂(k) ceiling lines |
| `10_missed_genes.py` | §6 classification, per chunk and aggregate |
| `12_genome_cds_counts.pbs` | annotation presence from the GenBank records |
| `13_cds_intervals.pbs`, `cds_intervals.py` | deposited CDS coordinates for the §6 shadow test |
| `13_missed_gene_candidates.pbs` | §6 over every chunk, 48 cores |
| `12_foldseek_databases.pbs` | Foldseek target databases (issue #92) |
| `14_pilot_queryset.py` | candidate/shadow/CDS arms, length-matched |
| `14b_extract_orf_seqs.{py,pbs}` | amino acids and 3Di back out of the archives |
| `15_build_query_db.py` | Foldseek query database from precomputed 3Di |
| `16_candidate_burden.py` | §6.1 burden against GTDB assembly metadata |
| `17_pilot_search.pbs` | the searches plus the shuffled-3Di null; also writes best-hit-per-query into `best/` |
| `18_pilot_analysis.py`, `18b_analysis.pbs` | §6.1 readout: hit rates, coverage, stratification |
| `04b_regenerate_entropy_rows.pbs` | re-emits per-ORF TSVs carrying `contig_length`, which the originals lack |
| `13b_cds_intervals_all.pbs` | `13` parallelised across chunks — the serial loop is unusable at 760 |
| `19_full_queryset.pbs` | full-population query set; per-chunk matching, sharded output |
| `20_orf_context.{py,pbs}` | genomic context, remaining entropy axes, reading-frame class; recomputes CDS overlap independently of the classifier as a fatal cross-check |
| `21_target_descriptions.py` | target id → product, from the Foldseek header databases |
| `22_rank_candidates.{py,pbs}` | ranked candidate table and the direct-evidence ladder |
| `23_gtdb_prodigal_coords.pbs` | GTDB Prodigal coordinates, streamed from the 123 GiB archive |
| `24_prodigal_overlap.{py,pbs}` | ORF-level agreement with an independent gene caller |
| `25_functional_classes.{py,pbs}` | functional classes, mobile-element and BFVD-only flags |
| `26_select_examples.{py,pbs}` | manuscript examples, chosen by class rather than score |
| `27_build_dossiers.{py,pbs}` | dossiers, with neighbour products read back from the GenBank archives; also emits `exemplar_neighbours.tsv` |
| `28_report_figures.py` | every figure in §6 and §6.1, read from the machine-readable artefacts the analysis stages emit rather than re-derived |

The same tree is committed under `Validation/gtdb/scripts/` in this repository;
see its `README.md` for the traps each stage encodes. Gadi-specific PBS
templates and install instructions were contributed upstream on branch
`feature/pbs-gadi-templates` of `linsalrob/genome_entropy`.

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
