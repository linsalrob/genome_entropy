#!/bin/bash
#SBATCH --job-name=phage_qdb
#SBATCH --account=pawsey1018
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=58G
#SBATCH --time=01:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Draw the Foldseek arms (05_select_queries.py) and build the query databases
# with #92's tsv2db builder (Validation/gtdb/scripts/15_build_query_db.py):
#   qdb    every arm;  qnull  candidate arm with each 3Di string shuffled
#   sbatch 05_select_queries.slurm intrinsic
set -euo pipefail
source "${PHAGE_REPO:?set PHAGE_REPO to the checkout}/Validation/Inphared/scripts/env.sh"
fs="${1:-intrinsic}"
D="$PHAGE_WORK/foldseek_$fs"
mkdir -p "$D/search"
orig=()   # the original run's comparison arms exist only for the 31,200-genome ORF set
[[ -d "$PHAGE_WORK/rank_orfs_orig" ]] && orig=(--orig "$PHAGE_WORK/rank_orfs_orig")
python "$PHAGE_SCRIPTS/05_select_queries.py" --tables "$PHAGE_WORK/tables" \
    --rank "$PHAGE_WORK/rank_orfs_relabelled_$fs" "${orig[@]}" --out "$D"
B="$PHAGE_REPO/Validation/gtdb/scripts/15_build_query_db.py"
python "$B" --aa "$D/queries.aa.fasta" --3di "$D/queries.3di.fasta" --db-dir "$D/search" --prefix qdb \
    --foldseek "$FSENV/bin/foldseek"
python "$B" --aa "$D/cand.aa.fasta" --3di "$D/cand.3di.fasta" --db-dir "$D/search" --prefix qnull \
    --shuffle-3di 0 --foldseek "$FSENV/bin/foldseek"
ls -la "$D/search"
