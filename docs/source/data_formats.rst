Data formats, coordinates, and entropy
======================================

Input formats
-------------

``run`` accepts DNA FASTA through ``--input``, GenBank through ``--genbank``, or
both (FASTA supplies sequences while GenBank supplies CDS annotations). GenBank
``.gz`` files are supported. ``orf`` accepts DNA FASTA. ``fasta-to-protein`` and
``encode3di`` accept protein FASTA extensions ``.fasta``, ``.fa``, and ``.faa``;
``encode3di`` also accepts protein-record JSON from ``translate`` or
``fasta-to-protein``.

``read_json`` and ``write_json`` transparently use gzip when a path ends in
``.gz``. ML directory readers discover both ``*.json`` and ``*.json.gz``.

Unified pipeline JSON
---------------------

``run`` serialises schema ``2.1.0``. The top level is a list because one input
file can contain multiple sequence records. A compact representative record is:

.. code-block:: json

   [{
     "schema_version": "2.1.0",
     "input_id": "contig_1",
     "input_dna_length": 900,
     "dna_entropy_global": 1.97,
     "alphabet_sizes": {
       "dna": 4, "protein": 20, "three_di": 20, "twelve_state": 12
     },
     "features": {
       "orf1": {
         "orf_id": "orf1",
         "location": {"start": 1, "end": 300, "strand": "+", "frame": 1},
         "dna": {"nt_sequence": "ATG...TAA", "length": 300},
         "protein": {"aa_sequence": "M...*", "length": 100},
         "three_di": {
           "encoding": "ACD...",
           "length": 100,
           "method": "modernprost_aa2fold",
           "model_name": "gbouras13/modernprost-50M",
           "inference_device": "cuda"
         },
         "twelve_state": {"encoding": "ABC...", "length": 100},
         "metadata": {
           "parent_id": "contig_1",
           "table_id": 11,
           "has_start_codon": true,
           "has_stop_codon": true,
           "in_genbank": false
         },
         "entropy": {
           "dna_entropy": 1.8,
           "protein_entropy": 3.4,
           "three_di_entropy": 3.1,
           "twelve_state_entropy": 2.7
         }
       }
     }
   }]

Coordinates and identifiers
---------------------------

ORF coordinates originate in ``get_orfs`` output and are stored as **one-based,
inclusive** ``start`` and ``end`` values. The sequence extractor converts them
to Python slicing internally. On the reverse strand, coordinates refer to the
reverse-complement sequence used by the implementation. ``strand`` is ``+`` or
``-`` and ``frame`` is the absolute frame number reported by ``get_orfs``.
This differs from the stale zero-based/half-open convention previously stated
in the dataclass docstring and older format guide.

``orf_id`` is the identifier emitted by ``get_orfs``; ``parent_id`` and the
top-level ``input_id`` identify the source sequence. ``has_start_codon`` and
``has_stop_codon`` currently indicate whether the ORF amino-acid string contains
``M`` or ``*`` respectively; they are not restricted to terminal positions.

GenBank matching and ``in_genbank``
-----------------------------------

When GenBank annotations are supplied, CDS features are extracted with Biopython
and compared to called ORFs. Public ORF coordinates remain one-based and
inclusive. Matching converts positive-strand ORFs and the reverse-complement
coordinates reported for negative-strand ORFs to a single zero-based,
half-open genomic convention. Biopython locations already use that internal
convention. ``codon_start`` offsets are applied at the biological CDS start.

A match requires the same parent record, strand, and codon phase. The genomic
intersection must cover at least 90% of the shorter coding interval. Amino-acid
slices are then selected from the shared genomic codons; at least 98% of those
aligned residues must be compatible. This recognises alternative start codons,
N-terminal extensions, contained gene models, and small terminal differences
without accepting an arbitrary conserved domain or unrestricted local protein
similarity.

Whitespace and case are ignored in translations, as is one terminal ``*``.
``X`` in either aligned translation is an unknown residue compatible with any
valid amino-acid symbol. Other unequal pairs, internal stops, internal
insertions, and internal deletions remain mismatches.

The annotation's ``/translation`` is preferred. If it is absent, the CDS
nucleotides are extracted in biological orientation, adjusted by
``codon_start``, and translated with the CDS ``transl_table``. The fallback
order is an explicit record/source translation table, the pipeline table, then
the project default. A CDS table therefore need not equal the ORF caller's
global table.

Simple partial CDS locations can match when their represented interval and
translation satisfy the same thresholds. Joined, compound, and
origin-crossing locations cannot be mapped safely to the current contiguous
ORF representation and are skipped with a debug message rather than flattened.
Translation exceptions, pseudogenes, and frameshifts are not independently
reconstructed. Debug logging reports identifiers, normalised coordinates,
phase, overlap, aligned length, wildcard count, identity, decision, and reason,
but never full sequences.

``in_genbank=True`` means this coordinate-anchored heuristic matched an
annotated CDS; it does not establish biological function. Missing GenBank
annotation input leaves the field false.

.. _raw-and-normalised-entropy:

Raw and normalised entropy
--------------------------

Stored entropy is Shannon entropy in bits:

.. code-block:: text

   H = -sum(p_i * log2(p_i))

Short sequences can have a low observed maximum and unstable symbol frequencies,
so compare entropy across lengths cautiously. Ambiguous or non-standard symbols
that occur in the sequence contribute observed categories to raw entropy.

Normalised entropy is entirely derived downstream:

.. code-block:: python

   normalised_entropy = raw_entropy / math.log2(alphabet_size)

Use theoretical sizes 4 for DNA, 20 for protein, 20 for 3Di, and 12 for
12-state. Standard JSON never contains normalised entropy. Public helpers are:

.. code-block:: python

   from genome_entropy.entropy import (
       normalise_entropy,
       normalise_dna_entropy,
       normalise_protein_entropy,
       normalise_three_di_entropy,
       normalise_twelve_state_entropy,
   )

   generic = normalise_entropy(raw_entropy, alphabet_size=20)
   protein = normalise_protein_entropy(raw_entropy)

All helpers return ``None`` for missing entropy. The generic helper raises
``ValueError`` when a non-missing value is supplied with an alphabet size of one
or less.

Intermediate and backward-compatible formats
--------------------------------------------

``orf`` writes a list of ``OrfRecord`` objects. ``translate`` writes a list of
``ProteinRecord`` objects embedding the source ORF. ``encode3di`` writes a list
of structural records embedding the protein and ORF, with nullable
``twelve_state``. ``entropy`` reads that structural-record list and writes an
``EntropyReport`` containing raw per-ORF mappings plus theoretical alphabet
sizes; its global DNA entropy is ``0.0`` because the original full contig is not
available at that stage.

The JSON writer converts in-memory legacy ``PipelineResult`` objects to unified
schema 2.1. It does not provide a general command that migrates arbitrary legacy
JSON dictionaries on disk. ML feature extraction accepts both unified and older
pipeline layouts and treats absent 12-state features as missing values.
