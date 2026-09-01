# Structural-state entropy across GTDB representative genomes

**Dataset:** GTDB release 232 (R11-RS232, 15 April 2026)
**Model:** `gbouras13/modernprost-50M` — dual-head ModernProst, 3Di + 12-state
**Tool:** `genome_entropy` 0.2.0, output schema 2.2.0
**Platform:** NCI Gadi (PBS Pro), project `ob80`
**Report date:** 2026-08-31 — bacteria and archaea complete

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
high 3Di entropy — a candidate pool for genes missed by annotation — is largely
explained by two confounders: genomes that appear never to have been annotated
(95.5% of the pool in the two chunks where the full pool was counted), and
shadow ORFs overlapping real CDS (62.9% of what remains, measured across all
760 chunks). Over the whole bacterial set, **3,562,431 candidates survive in
96,701 genomes**.

Both splits are provisional. The shadow test compared plus- and minus-strand
coordinates in different systems, and "never annotated" was inferred from
`in_genbank` rather than from the GenBank records, which can only over-count
unannotated genomes. Both defects are fixed in the scripts; the numbers here
predate the fixes, at both two-chunk and 760-chunk scale. See the note in
section 6.

A Foldseek structural homology search of the candidate pool (§6.1, [issue #92](https://github.com/linsalrob/genome_entropy/issues/92))
adds a third result: on full-length alignment coverage neither candidates nor
shadow ORFs resemble annotated CDS, and the two are indistinguishable from each
other. The first half of that stands on its own. The second half is weakened by
the classification defects above, which move ORFs between the two arms and so
bias the comparison toward finding no difference.

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
Separately, **at least 48.9% of bacterial representatives carry no CDS
annotation at all** (GCA assemblies submitted unannotated), and every ORF in
those genomes is `False` by construction. Any analysis of what `False` *means*
must exclude them; see §6. That figure is an upper bound, and so was the 46%
measured mid-run that it replaces: both come from `in_genbank`, which cannot
distinguish a genome with no annotation from one whose annotations the ORF
matcher rejected. `12_genome_cds_counts.pbs` answers this from the GenBank
records; see the note in §6.

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

In `/g/data/ob80/re3494/gtdb_entropy/figures/`, regenerated over **every**
chunk of both domains rather than the 20 bacterial chunks that existed while the
run was in flight:

| file | content |
|---|---|
| `protein_vs_3di_entropy_bac.png` / `_arc.png` | scatter, 4 panels |
| `protein_vs_3di_hexbin_bac.png` / `_arc.png` | hexbin, log counts |
| `protein_vs_3di_kde_bac.png` / `_arc.png` | KDE + contour overlay |
| `protein_vs_3di_joint_bac.png` / `_arc.png` | joint density with marginals |
| `protein_vs_3di_domains.png` | bacteria against archaea |

Every panel with a 3Di axis now carries dotted log₂(k) reference lines, so the
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
> *The miss-rate denominator.* The per-genome "≈0.9% of its CDS count" used the
> number of matched ORFs as the genome's CDS count, which undercounts wherever
> the matcher rejected a real CDS and therefore inflates the rate. It now comes
> from the deposited counts.
>
> Synthetic checks show the first two defects produce errors in opposing
> directions, so totals can look stable while individual ORFs sit in the wrong
> group; the third inflates in one direction only (5× on a fixture built to
> isolate it). Treat the 15,338 / 8,651 split, everything derived from it, the
> "1.6% survives" figure, and the per-genome miss rate as provisional until the
> analysis is rerun.
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
> 95.5% here and the 48.9% quoted in §1 and §6 are upper bounds, not estimates.
> `12_genome_cds_counts.pbs` now counts CDS features in the GenBank records
> directly, and `10_missed_genes.py` requires its table instead of the proxy.
> Both figures need regenerating from it.
>
> Rerunning needs chunk TSVs carrying the `contig_length` column that
> `extract_entropy_rows.py` now emits; TSVs from the original run do not have it
> and must be regenerated from the JSON archives.
>
> **The 760-chunk rerun does not escape any of this.** It applied the same
> classification to every chunk, so the agreement between scales (63.9% shadows
> in two chunks against 62.9% across 760) demonstrates that the method is
> stable, not that it is correct. Every figure in the full-scale table below
> inherits both defects.

Two confounders dominate. The pool was first characterised on two chunks
(`bac_000`, `bac_051`, 500 genomes, 537,488 unmatched ORFs with 3Di ≥ 2.5) and
the classification has since been run over all 760 bacterial chunks.

Two-chunk pool, where the never-annotated fraction was counted:

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

Over all 760 bacterial chunks:

| | count | share |
|---|---:|---:|
| unmatched ORFs in annotated genomes with 3Di ≥ 2.5 | 9,598,500 | |
| → **shadow** of an annotated CDS | 6,036,069 | 62.9% |
| → **intergenic — candidates** | **3,562,431** | 37.1% |

**96,701 of the 96,875 annotated genomes (99.8%) carry at least one candidate**,
median 26 per genome, maximum 5,763. That near-universality needs explaining in
either direction: annotation misses genes almost everywhere in GTDB
representatives, or the candidate definition is still too permissive. The
burden analysis in §6.1 argues for the second.

Candidates sit between annotated CDS and the low-3Di background on every axis —
shorter than real genes but longer than background, with 3Di entropy much
closer to real CDS than to noise. That is consistent with missed genes, which
should be biased toward short proteins precisely because short ORFs are what
annotation pipelines under-call. The earlier reading of median **24 candidates
per annotated genome, ≈0.9% of its CDS count** as "a plausible miss rate" is
superseded twice over: its denominator was the number of ORFs the matcher
accepted rather than the deposited CDS count, so every genome with a rejected
CDS contributed too small a denominator and the rate is inflated; and §6.1
shows the burden tracks assembly fragmentation rather than biology. The script
now divides by `n_cds` from `12_genome_cds_counts.pbs` and indexes over every
annotated genome, so genomes yielding no candidates count as zero rather than
dropping out of the median.

Those survivors do look protein-like, and this is where the analysis runs out of
discriminating power:

| group | n | median aa | ≥100 aa | median 3Di | median protein |
|---|---:|---:|---:|---:|---:|
| annotated CDS (sampled) | 1,520,000 | 323 | 98.4% | 3.29 | 4.01 |
| **candidate: intergenic, 3Di ≥ 2.5** | **3,562,431** | **175** | **89.5%** | **2.94** | **4.00** |
| **shadow of CDS, 3Di ≥ 2.5 (sampled)** | **1,519,199** | **180** | **89.1%** | **2.88** | **3.98** |
| intergenic, 3Di < 2.5 (sampled) | 1,520,000 | 126 | 81.0% | 1.17 | 3.73 |

Candidates sit between annotated CDS and the low-3Di background — shorter than
real genes but longer than background, with 3Di entropy much closer to real CDS
than to noise. Median 26 candidates per annotated genome, ≈0.9% of its CDS
count, is a plausible miss rate rather than an implausible one.

But **candidates and high-3Di shadows are indistinguishable on every axis
available without a search** — length, the fraction over 100 aa, 3Di entropy and
protein entropy all agree to within a few percent. Since a shadow is by
construction *not* a distinct protein, the resemblance is either evidence that
the shadow set contains real genes too, or that the candidate set is mostly
shadow-like artefacts the overlap test failed to catch. Nothing in this dataset
can tell those apart.

**This therefore does not discriminate missed genes from other intergenic
coding-like sequence**: pseudogenes, phage or IS remnants, or ORFs annotation
deliberately suppressed. The decisive next steps, tracked in
[issue #92](https://github.com/linsalrob/genome_entropy/issues/92):

1. **Structural homology search with Foldseek**, not a sequence search. If these
   ORFs were findable by sequence similarity the original pipeline would likely
   have called them, so sequence homology is the weaker instrument. The 3Di
   strings needed already exist in the archives from this run, so Foldseek's
   expensive step is already paid for; a query database can be built from
   precomputed 3Di and amino acids with `foldseek tsv2db`. Shadows are searched
   as a first-class control arm alongside the candidates, since separating those
   two is the actual question. An amino-acid search against the same target
   databases runs as an orthogonal comparator, making **structure-only
   homology** — a Foldseek hit with no sequence hit — a category the analysis can
   isolate.
2. **Stratify by annotation provenance.** GTDB records which pipeline annotated
   each genome; if candidates concentrate in older Prokka versions rather than
   recent PGAP, that is direct evidence of pipeline misses.

### 6.1 The Foldseek pilot: what the structural search found

Item 1 above has been run ([issue #92](https://github.com/linsalrob/genome_entropy/issues/92)).
Four arms of 8,651 ORFs each — candidate, high-3Di shadow, annotated CDS,
low-3Di intergenic — were searched against PDB100, AFDB/Swiss-Prot, CATH50 and
BFVD twice over: once with `foldseek search --alignment-type 2` (3Di + amino
acid) and once with `foldseek base:search` (amino acid only). Same queries,
same targets, one difference, so **structure-only homology is a set difference
on identical targets** rather than a comparison across two differently built
databases. A fifth query set, the candidates' own 3Di strings shuffled,
provides a technical null with identical amino acids, length and 3Di
composition but no residue order.

Arms are length-matched exactly (25th/50th/75th percentile 116/169/296 aa), and
shadows were matched to candidates on 3Di entropy as well, preferring a partner
from the same genome — achieved for 98.5% of pairs, which holds N50,
completeness, contamination, GC and annotation pipeline fixed simultaneously.

**The assay has range and the controls hold.** Union over the four databases:

| arm | struct | seq | struct-only | neither |
|---|---:|---:|---:|---:|
| annotated CDS | 73.8% | 74.1% | 12.2% | 13.6% |
| candidate | 58.0% | 50.0% | 15.0% | 35.0% |
| shadow | 55.8% | 47.1% | 14.9% | 38.0% |
| low-3Di intergenic | 1.5% | 9.3% | 1.1% | 89.5% |
| null (shuffled 3Di) | 0.2% | — | — | — |

Shuffling the 3Di takes candidates from 58.0% to 0.2%, and to zero against
BFVD, so `--alignment-type 2` really is aligning on the structural alphabet
rather than leaking amino-acid signal. Structure-only homology is substantial —
15.0% of candidates have a structural hit with no sequence hit on the same
target database — and it is equally substantial in the shadow arm.

**Hit rate is a measure of encoder confidence, not of foldedness.** Stratified
by the 3Di entropy that defined the arms:

| 3Di quintile | candidate | shadow | annotated CDS |
|---|---:|---:|---:|
| Q2 (1.46–2.58) | 14.4% | 13.0% | 31.8% |
| Q3 (2.58–2.88) | 30.8% | 28.5% | 74.2% |
| Q4 (2.88–3.38) | 80.1% | 80.7% | 92.5% |
| Q5 (3.38–4.12) | 99.7% | 99.3% | 99.5% |

Above 3.38 bits everything hits at 99%, real gene and unmatched ORF alike.
Because the null preserves 3Di composition and therefore entropy exactly while
collapsing to 0.2%, entropy *per se* does not manufacture alignments; what the
table shows is that a sequence-only 3Di encoder emits confident, well-ordered
3Di for anything with protein-like composition, and Foldseek aligns it. This is
the same phenomenon as the log₂(k) ceiling of §5 seen from the other end, and
it is the reason a raw hit rate is the wrong readout for a predicted-3Di
search. Note also that the low-3Di intergenic arm is low-entropy *by
construction*, so its floor-level rate is confounded with entropy and adds
nothing this table does not already say; the entropy-matched intergenic control
it was meant to be is still missing.

**Full-length mutual coverage is the readout that discriminates.** Requiring
the best hit to cover ≥80% of both query and target:

| database | annotated CDS | candidate | shadow | implied real-gene share (95% CI) |
|---|---:|---:|---:|---:|
| PDB100 | 38.5% | 11.6% | 11.1% | 1.8% (−1.7 – 5.2) |
| AFDB/Swiss-Prot | 44.3% | 12.0% | 11.8% | 0.5% (−2.5 – 3.5) |
| CATH50 | 26.4% | 9.9% | 8.8% | 6.2% (1.3 – 10.9) |

This criterion separates gene from non-gene 3–4× where the loose hit rate
managed 1.3×, and it is not saturated by the encoder artefact: restricted to
3Di ≥ 3.0 it still gives annotated CDS 60.3% against candidate 23.8% and shadow
24.1%. Median fractional identity is the same (~0.29–0.32) in all three arms, so
what separates them is full-length correspondence with a real protein, not hit
quality in general. The last column treats candidates as a mixture of real genes
behaving like annotated CDS and non-genes behaving like shadows; two of three
intervals include zero.

**Burden tracks assembly quality, not biology.** Over all 96,875 annotated
bacterial representatives (`16_candidate_burden.py`), Spearman ρ of candidates
per Mb against contig N50 is −0.603, contig count +0.547, CheckM2 completeness
−0.491, contamination +0.356 — while coding density, the covariate that would
matter if this were about gene content, is flat at −0.040. Assembly level
reproduces it without a coefficient: 8.97 candidates/Mb for contig-level
assemblies against 5.44 for chromosome-level. Partial-annotation genomes that
passed the "≥1 CDS" filter explain only the tail (137 genomes, 1.9% of the
pool); a ≥500 CDS/Mb floor removes them.

**What this does and does not establish.** That neither candidates nor shadows
resemble annotated CDS on full-length coverage is robust: it does not depend on
which of the two arms a given ORF belongs to. That candidates and shadows are
indistinguishable *from each other* is the weaker claim, because both arms were
drawn from the superseded classification described at the head of this section —
misassignment between them biases exactly toward the null that was observed.
The comparison must be repeated once items 6 and 7 of §11 are done. The
shuffled-3Di null and the full-length-coverage criterion are, on the evidence
here, the two controls a predicted-3Di Foldseek search cannot do without.

Caveats: two chunks (`bac_000`, `bac_051`), bacteria only, E < 10⁻³, best hit
per query. AFDB50 was planned and has been dropped — a 54M-entry search would
re-measure a difference already bounded near zero.

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

### Done since the first draft

- Dotted **log₂(k) reference lines** on every 3Di axis, labelled (§4).
- **Joint figure with marginal distributions**, the 3Di margin as a histogram
  so the 1.585 edge is not smoothed away (§4).
- All figures regenerated over **all 760 bacterial chunks**, plus **archaeal
  counterparts and a bacteria-vs-archaea comparison** (§4, §5). Archaeal genomes
  are smaller, as expected — 5,420 ORF calls per genome against 13,537 — but the
  claim that they are *less often annotated* was an artefact of a mid-run sample
  and is corrected in §2: 54.3% against 51.1%.
- The candidate classification of §6 run over **all 760 chunks** rather than two.

### Analyses

5. Compute **fraction of `D` residues** per ORF as a length-independent
   alternative to 3Di entropy (§5.3).
6. **Run `12_genome_cds_counts.pbs`** and regenerate the annotation-presence
   figures (the 48.9% in §1/§6 and the 95.5% in §6) from CDS features in the
   GenBank records rather than from `in_genbank`. Both are currently upper
   bounds.
7. **Rerun `10_missed_genes.py`** now that it converts negative-strand
   coordinates onto the forward genomic axis, takes annotation presence from
   item 6, and tests shadows against deposited CDS coordinates from
   `13_cds_intervals.pbs`; replace the superseded shadow/intergenic numbers in
   §6 at both scales. Needs chunk TSVs regenerated from the JSON archives,
   because the originals predate the `contig_length` column, and `13` run over
   the same chunks.
8. **Test what 3Di state `D` corresponds to.** Run DSSP over predicted
   structures, or a disorder predictor, on ORFs below and above the 1.585 line.
   The entropy ceiling is established without this, but the disorder reading of
   `D` dominance is not (§5).
9. **Re-run the Foldseek pilot on the recomputed candidate set** (§6.1,
   [issue #92](https://github.com/linsalrob/genome_entropy/issues/92)). The first run is
   complete and its controls are sound, but its candidate and shadow arms were
   drawn from the superseded classification, so the candidate-versus-shadow
   comparison must be repeated after items 6 and 7. The full-length-coverage
   finding does not depend on that split.
10. Stratify candidates by GTDB annotation provenance (§6).
11. Parallelise `05_aggregate_results.py` — 87 s per chunk single-threaded is
   ~18 h over 760 chunks. Each genome lives in exactly one chunk, so chunks are
   independent and this parallelises trivially. Consequently the **bacterial**
   per-genome summary has never been produced; only the archaeal one exists
   (`summary_per_genome_arc.tsv`, 41 chunks in 6.5 min).
12. Count the **never-annotated fraction of the high-3Di pool at full scale.**
    The 95.5% in §1 and §6 is measured on two chunks; the 760-chunk run
   classified only ORFs inside annotated genomes, so the full-scale equivalent
   of that denominator is not yet known. One further pass over the TSVs would
   settle it.

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
| `17_pilot_search.pbs` | the four-arm search plus the shuffled-3Di null |
| `18_pilot_analysis.py` | §6.1 readout: hit rates, coverage, stratification |

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
