User guide
==========

Pipeline concepts
-----------------

``genome_entropy`` connects several distinct operations:

#. **ORF discovery** calls candidate coding regions in six reading frames using
   ``get_orfs``. This is gene calling, not functional annotation.
#. **Translation** uses the selected NCBI genetic code (table 11 by default).
#. **Structural-state prediction** converts protein sequence into 3Di and,
   with multitask ModernProst, a 12-state encoding.
#. **Entropy calculation** measures observed symbol diversity at each available
   representation.
#. **GenBank matching**, when annotations are supplied, assigns ``in_genbank``
   through a strand- and C-terminus-aware heuristic.

ORFs and genetic codes
----------------------

``--min-nt`` controls the minimum nucleotide length for ``orf``; ``run`` exposes
the equivalent ``--min-aa`` and multiplies it by three. Table 11 is the bacterial,
archaeal, and plant-plastid code. Select a different table only when appropriate
for the source organism.

Coordinates and start/stop flags follow the exact implementation conventions in
:doc:`data_formats`; do not reinterpret them as BED coordinates.

Structural-state encodings
--------------------------

3Di is a 20-state structural alphabet introduced by Foldseek. It describes
local tertiary interactions inferred here from protein sequence; it is not a
set of atomic coordinates and does not replace full structure prediction.

The multitask ModernProst head also emits 12 classes serialised as ``A`` through
``L``. Legacy encoders expose missing 12-state values as ``None``/JSON ``null``.
See :doc:`models` for model selection, provenance, and security.

Shannon entropy
---------------

For observed symbol frequencies ``p_i``, raw Shannon entropy in bits is:

.. code-block:: text

   H = -Σ p_i log₂(p_i)

Zero describes a sequence containing one observed symbol. Larger values reflect
more even use of more symbols, but do not by themselves establish sequence
quality, function, or biological complexity. Short sequences cannot realise the
theoretical maximum reliably, and comparisons across lengths or alphabets need
caution.

Standard output stores raw entropy only. Normalised entropy divides by the
theoretical maximum and should be derived downstream. The formula, alphabet
sizes, helpers, and missing-value semantics are documented in
:ref:`Raw and normalised entropy <raw-and-normalised-entropy>`.

Choosing an input workflow
--------------------------

Use DNA FASTA when only sequence-derived features are required. Use GenBank when
``in_genbank`` labels are required. Use protein FASTA with ``encode3di`` when ORF
calling and nucleotide entropy are outside the analysis. Gzip support and each
intermediate JSON format are listed in :doc:`data_formats`.

Performance and reproducibility
-------------------------------

The encoding budget is an approximate sum of amino-acid lengths per batch, not
a tokenizer token count guaranteed across models. Measure it on representative
hardware. Pre-cache model artefacts for offline jobs and record package, model,
PyTorch, Transformers, device, and parameter versions in an analysis provenance
record.

Multi-GPU mode parallelises batches across one encoder per visible accelerator.
It does not split an individual sequence across devices. See :doc:`hpc` for
scheduler and ROCm details.

Interpretation limits
---------------------

``in_genbank=False`` means the current matching heuristic did not find a CDS;
it does not prove the ORF is non-coding. Conversely, ``True`` is a heuristic
annotation match, not a functional assignment. ML predictions learn this label
and inherit its biases. Use genome- or file-level splits to estimate transfer to
unseen records and read :doc:`ml` before reporting results.

The GenBank heuristic requires the same parent, strand, and biological stop,
then compares C-terminal suffixes after ignoring each protein's first residue.
An aligned ``X`` is compatible with a specific residue, but unrelated mismatches
and internal or N-terminal-only similarity remain disallowed. For nucleotide
translation, multiply-resolvable IUPAC codons such as ``AAN`` and ``NNN`` become
``X``; ambiguities with one translation, such as ``GCN`` (alanine), retain that
specific amino acid.
