#!/bin/bash
#SBATCH --job-name=logan_fs
#SBATCH --account=pawsey1018
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=115G
#SBATCH --time=24:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#
# Missed-gene structure search, one array element per target database
# (line i of the DB list), after #92's 17_pilot_search.pbs:
#   struct  foldseek search, 3Di + AA (--alignment-type 2), qdb and qnull
#   seq     foldseek base:search, AA only, qdb
# against the SAME target, so structure-only homology is a set difference.
# Queries carry no C-alpha, hence --sort-by-structure-bits 0 (AFDB50 then
# needs ~35 GB rather than ~151 GB).
#
# Reductions per query (written next to the raw m8):
#   best/     best hit by bits (one row per query)
#   fullcov/  best hit among those with qcov >= 0.8 AND tcov >= 0.8, the
#             primary structural-homology readout (#92)
#
#   sbatch --array=1-5 23_foldseek_search.slurm dbs.txt

set -euo pipefail
FSENV="${FSENV:-/scratch/$PAWSEY_PROJECT/$USER/software/envs/foldseek}"
LOGAN_ROOT="${LOGAN_ROOT:-/scratch/$PAWSEY_PROJECT/$USER/Logan}"
DBROOT="${DBROOT:-$LOGAN_ROOT/foldseek_db}"
EVALUE="${EVALUE:-0.001}"
db=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
S="$LOGAN_ROOT/qc/candidates/search"
TMP="$LOGAN_ROOT/tmp/fs_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMP" "$S/best" "$S/fullcov"
trap 'rm -rf "$TMP"' EXIT
eval "$(conda shell.bash hook)"; conda activate "$FSENV"
FIELDS="query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue,bits,taxid,taxname"
run() {   # qdb tag mode
    local q="$S/$1" tag="$2" mode="$3" out="$S/$2.$db.$3.m8"
    [[ -s "$out" && -s "$S/best/$tag.$db.$mode.m8" ]] && { echo "$tag $db $mode done"; return; }
    local res="$TMP/res_${tag}_${mode}"
    if [[ "$mode" == struct ]]; then
        foldseek search "$q" "$DBROOT/$db" "$res" "$TMP/w" --alignment-type 2 --sort-by-structure-bits 0 \
            -e "$EVALUE" --threads "$SLURM_CPUS_PER_TASK" -v 1
    else
        foldseek base:search "$q" "$DBROOT/$db" "$res" "$TMP/w" -e "$EVALUE" --threads "$SLURM_CPUS_PER_TASK" -v 1
    fi
    foldseek convertalis "$q" "$DBROOT/$db" "$res" "$out.partial" --format-output "$FIELDS" \
        --threads "$SLURM_CPUS_PER_TASK" -v 1
    mv "$out.partial" "$out"
    awk -F'\t' '{ if (!($1 in b) || $10+0 > b[$1]) { b[$1]=$10+0; l[$1]=$0 } } END { for (q in l) print l[q] }' \
        "$out" > "$S/best/$tag.$db.$mode.m8"
    awk -F'\t' '$7+0 >= 0.8 && $8+0 >= 0.8 { if (!($1 in b) || $10+0 > b[$1]) { b[$1]=$10+0; l[$1]=$0 } } END { for (q in l) print l[q] }' \
        "$out" > "$S/fullcov/$tag.$db.$mode.m8"
    echo "$(date -Is) $tag $db $mode: $(wc -l < "$S/best/$tag.$db.$mode.m8") queries with a hit, $(wc -l < "$S/fullcov/$tag.$db.$mode.m8") full-length"
    rm -rf "$res"*
}
# MODES selects the searches (default: all three, in sequence). For a large
# target they can run as separate jobs, e.g. MODES="qdb:struct", since each
# writes its own outputs:
#   sbatch --array=5 --export=ALL,MODES="qdb:seq qnull:struct" 23_foldseek_search.slurm dbs.txt
echo "$(date -Is) $db on $(hostname)"
for m in ${MODES:-qdb:struct qdb:seq qnull:struct}; do
    run "${m%%:*}" "${m%%:*}" "${m##*:}"
done
