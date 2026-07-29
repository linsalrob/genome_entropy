# Unified JSON output format

The maintained schema reference is [`docs/source/data_formats.rst`](source/data_formats.rst) and is published as [Data formats, coordinates, and entropy](https://genome-entropy.readthedocs.io/en/latest/data_formats.html).

Current pipeline output uses schema `2.1.0` and stores each ORF once under `features`. Coordinates are one-based and inclusive, matching `get_orfs` output. Each feature includes DNA, protein, 3Di, optional 12-state, metadata, and raw entropy. Legacy encoders serialise `twelve_state` and `twelve_state_entropy` as JSON `null`; missing fields in older JSON are treated as unavailable.

Normalised entropy is downstream derived data and is not written to standard JSON. The maintained reference includes a complete compact example, intermediate record formats, gzip support, GenBank matching semantics, and compatibility limitations.
