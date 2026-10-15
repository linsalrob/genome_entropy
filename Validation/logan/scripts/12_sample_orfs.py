#!/usr/bin/env python3
"""Map step: a class-stratified, weighted sample of ORFs from one analysis shard.

Each ORF is kept with a probability set by its stratum (see RATES), and the
sample carries weight = 1 / rate, so population-level AUC / AP can be
recovered from the sample with sample_weight. Rare classes (complete
Prodigal+, intergenic) are kept at higher rates than the billions of
shadows. Truncation state is kept on every row so models can be fitted on
complete ORFs and then checked on contig-end-truncated ones.

Sampling is deterministic: the generator is seeded by the shard number and
row group.

  12_sample_orfs.py --shard 000
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import logan_common as lc  # noqa: E402

_spec = importlib.util.spec_from_file_location("h", HERE / "10_feature_histograms.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

# Keep-probabilities by (label, truncation). Chosen for ~10 M rows over the
# ~90 B ORFs, with every contrast class represented by >= ~1 M.
RATES = {
    ("prodigal_complete", "complete"): 2e-3,
    ("prodigal_partial", "complete"): 2e-3,
    ("prodigal_partial", "open5"): 1e-4,
    ("prodigal_partial", "open3"): 2e-5,
    ("intergenic", "complete"): 7e-4,
    ("intergenic", "open5"): 2e-4,
    ("intergenic", "open3"): 1e-4,
    ("antisense_overlap", "complete"): 1e-4,
    ("antisense_overlap", "open5"): 2e-5,
    ("antisense_overlap", "open3"): 1e-5,
    ("frameshift_overlap", "complete"): 1.5e-4,
    ("frameshift_overlap", "open5"): 3e-5,
    ("frameshift_overlap", "open3"): 1.5e-5,
    ("marginal_overlap", "complete"): 1e-3,
    ("marginal_overlap", "open5"): 2e-4,
    ("marginal_overlap", "open3"): 2e-4,
    ("same_frame_unmatched", "complete"): 2e-2,
    ("same_frame_unmatched", "open5"): 2e-2,
    # Prodigal- like the rest; sampled heavily because it is rare (0 ORFs in
    # the ~50 k set), so that it reaches the "all Prodigal-" contrast if present.
    ("frame_undefined_overlap", "complete"): 2e-2,
    ("frame_undefined_overlap", "open5"): 2e-2,
    ("frame_undefined_overlap", "open3"): 2e-2,
}
KEEP = ["accession", "contig_n", "start", "strand", "contig_len", "stop", "five_open", "aa_len",
        "prodigal_class", "match_gene_partial", "best_overlap_bp", "biome", "batch", "chunk",
        "aa_entropy_q4", "three_di_entropy_q4", "twelve_state_entropy_q4", "mi_3di_12st_q4"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shard", required=True)
    args = ap.parse_args()
    src = lc.root() / "analysis" / lc.CACHE_VERSION / "orfs" / f"{args.shard}.parquet"
    out = lc.root() / "analysis" / lc.CACHE_VERSION / "sample" / f"{args.shard}.parquet"
    if out.exists():
        print(f"{out} exists")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    rate = np.zeros(int(np.prod(H.SHAPE[:2])))
    for (lab, tr), r in RATES.items():
        rate[H.LABELS.index(lab) * len(H.TRUNC) + H.TRUNC.index(tr)] = r
    pf = pq.ParquetFile(src)
    parts, kept = [], 0
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g, columns=KEEP)
        s = H.strata(t)
        lab, tr = np.unravel_index(s, H.SHAPE)[:2]
        r = rate[lab * len(H.TRUNC) + tr]
        rng = np.random.default_rng([int(args.shard), g])
        m = rng.random(len(r)) < r
        if not m.any():
            continue
        sub = t.filter(pa.array(m))
        # Plain strings: each row group carries its own small int8-coded
        # dictionaries, and merging them overflowed int8 on read.
        sub = pa.table({c: (sub[c].cast(pa.string()) if pa.types.is_dictionary(sub[c].type) else sub[c])
                        for c in sub.column_names})
        sub = sub.append_column("label", pa.array(np.array(H.LABELS)[lab[m]]))
        sub = sub.append_column("trunc", pa.array(np.array(H.TRUNC)[tr[m]]))
        sub = sub.append_column("weight", pa.array(1.0 / r[m]))
        parts.append(sub)
        kept += int(m.sum())
    tbl = pa.concat_tables(parts, promote_options="default")
    tmp = out.with_name(out.name + ".partial")
    pq.write_table(tbl, tmp, compression="zstd")
    tmp.rename(out)
    print(f"shard {args.shard}: {kept:,} sampled of {pf.metadata.num_rows:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
