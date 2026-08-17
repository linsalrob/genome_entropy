# Unified JSON refactoring: historical implementation note

This file records the migration from duplicated `orfs`, `proteins`, and
`three_dis` collections to the unified `features` mapping. It is not the
current schema reference.

The maintained [data-format documentation](docs/source/data_formats.rst)
describes schema version `2.2.0`, 1-based inclusive coordinates, nullable
12-state fields, compression support, metadata, entropy units, and legacy-input
compatibility. In particular, current output includes an alphabet size of 12
and, for multitask ModernProst models, `twelve_state`,
`twelve_state_entropy`, and `three_di_twelve_state_mutual_information`.
Standard JSON stores raw Shannon entropy and raw structural mutual information.

The refactoring's stable user-visible result is one record per ORF under
`features`, with location, DNA, protein, structural encoding, metadata, and
entropy grouped together without nesting duplicate ORF objects.
