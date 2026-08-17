Changelog
=========

0.2.0 — 17 August 2026
----------------------

Structural-state analysis
^^^^^^^^^^^^^^^^^^^^^^^^^

* Added raw per-ORF mutual information in bits between aligned 3Di and
  12-state ModernProst representations. The schema is now version 2.2; legacy
  3Di-only output remains compatible and reports the new field as ``null``.

Fixes
^^^^^

* Replaced stop-coordinate/C-terminal CDS matching with coordinate-normalised,
  strand- and phase-aware genomic overlap plus aligned translation comparison.
  Alternative starts, ambiguous ``X`` residues, small terminal differences,
  CDS-specific translation tables, ``codon_start``, and missing translation
  qualifiers are now handled without unrestricted local matching.

Documentation
^^^^^^^^^^^^^

* Audited repository documentation, public docstrings, CLI help, examples, and
  HPC templates against the current implementation.
* Reconciled model identifiers with the live George Bouras Hugging Face
  namespace and documented current multitask versus deprecated 3Di-only models.
* Documented 3Di and nullable 12-state output consistently, including the
  deterministic ``A``--``L`` serialisation.
* Replaced automatic-normalisation guidance with raw JSON entropy and explicit
  downstream normalisation helpers.
* Corrected coordinate, GenBank matching, ML splitting, missing-value,
  probability, device, and persistence descriptions.
* Corrected the Read the Docs project name and aligned Python/tool version
  claims with package metadata and CI.
* Made Sphinx warnings fatal in documentation CI and documented the local
  warnings-as-errors build command.

0.1.0 — 19 January 2026
-----------------------

Initial release with ORF discovery, translation, ProstT5 3Di encoding, raw
Shannon entropy, modular CLI commands, JSON I/O, logging, batching, and device
selection. Later releases added the unified schema, GenBank matching, ML,
multi-GPU operation, and ModernProst/12-state support; those intermediate
historical entries were not recorded in this file.
