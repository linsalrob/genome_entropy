#!/bin/bash
#SBATCH --job-name=logan_qdb
#SBATCH --account=pawsey1018
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# Foldseek query databases for the missed-gene search, from the extracted
# AA + ModernProst 3Di strings (22_extract_sequences.py), using #92's
# tsv2db builder (Validation/gtdb/scripts/15_build_query_db.py):
#   qdb    all four arms (candidate, intergenic_lo, antisense, prodigal)
#   qnull  candidate arm with each 3Di string shuffled (same AA, same 3Di
#          composition, order destroyed): the technical null
#
#   sbatch 23a_build_queries.slurm

set -euo pipefail
source "${LOGAN_SCRIPTS:-$HOME/GitHubs/genome_entropy/Validation/logan/scripts}/env.sh"
FSENV="${FSENV:-/scratch/$PAWSEY_PROJECT/$USER/software/envs/foldseek}"
D="$LOGAN_ROOT/qc/candidates"
mkdir -p "$D/search"
python - "$D" <<'PY'
import sys, glob, pandas as pd
d = sys.argv[1]
s = pd.concat([pd.read_csv(f, sep="\t", dtype=str) for f in sorted(glob.glob(f"{d}/seqs/*.tsv"))])
sel = pd.read_parquet(f"{d}/selection.parquet", columns=["query_id"])
assert set(s.query_id) == set(sel.query_id), f"{len(set(sel.query_id) - set(s.query_id))} queries missing"
assert (s.aa.str.len() == s.three_di.str.len()).all(), "AA / 3Di length mismatch"
for kind, col in (("aa", "aa"), ("3di", "three_di")):
    with open(f"{d}/search/queries.{kind}.fasta", "w") as fa, open(f"{d}/search/cand.{kind}.fasta", "w") as fc:
        for q, x in zip(s.query_id, s[col]):
            fa.write(f">{q}\n{x}\n")
            if q.startswith("candidate|"):
                fc.write(f">{q}\n{x}\n")
print(len(s), "queries;", s.query_id.str.split("|").str[0].value_counts().to_dict())
PY
B="$HOME/GitHubs/genome_entropy/Validation/gtdb/scripts/15_build_query_db.py"
python "$B" --aa "$D/search/queries.aa.fasta" --3di "$D/search/queries.3di.fasta" \
    --db-dir "$D/search" --prefix qdb --foldseek "$FSENV/bin/foldseek"
python "$B" --aa "$D/search/cand.aa.fasta" --3di "$D/search/cand.3di.fasta" \
    --db-dir "$D/search" --prefix qnull --shuffle-3di 0 --foldseek "$FSENV/bin/foldseek"
ls -la "$D/search"
