#!/bin/bash
#SBATCH --job-name=phage_fs
#SBATCH --account=pawsey1018
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=115G
#SBATCH --time=12:00:00
#SBATCH -o %x-%A_%a.out
#SBATCH -e %x-%A_%a.err
#
# Foldseek (CPU, as Validation/logan/scripts/23_foldseek_search.slurm), one
# array element per target database (line i of the DB list):
#   struct  foldseek search, 3Di + AA (--alignment-type 2), qdb and qnull
#   seq     foldseek base:search, AA only, qdb
# Queries carry no C-alpha, hence --sort-by-structure-bits 0.
# Reductions per query: best/ (best by bits), fullcov/ (best with qcov>=0.8
# and tcov>=0.8). theader is kept for exemplar descriptions.
#   sbatch --array=1-5 06_foldseek_search.slurm dbs.txt intrinsic
set -euo pipefail
source "${PHAGE_REPO:?set PHAGE_REPO to the checkout}/Validation/Inphared/scripts/env.sh"
conda activate "$FSENV"
EVALUE="${EVALUE:-0.001}"
# A DB-list line is a name under $FSDB, or name=/path/to/db for a database kept elsewhere (phold).
line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
db="${line%%=*}"
dbpath="$FSDB/$db"
[[ "$line" == *=* ]] && dbpath="${line#*=}"
fs="${2:-intrinsic}"
S="$PHAGE_WORK/foldseek_$fs/search"
TMP="$PHAGE_WORK/tmp/fs_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMP" "$S/best" "$S/fullcov" "$S/raw"
trap 'rm -rf "$TMP"' EXIT
FIELDS="query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue,bits,taxid,taxname,theader"
run() {   # qdb tag mode
    local q="$S/$1" tag="$2" mode="$3" out="$S/raw/$2.$db.$3.m8"
    [[ -s "$out" && -s "$S/best/$tag.$db.$mode.m8" ]] && { echo "$tag $db $mode done"; return; }
    local res="$TMP/res_${tag}_${mode}"
    if [[ "$mode" == struct ]]; then
        foldseek search "$q" "$dbpath" "$res" "$TMP/w" --alignment-type 2 --sort-by-structure-bits 0 \
            -e "$EVALUE" --threads "$SLURM_CPUS_PER_TASK" -v 1
    else
        foldseek base:search "$q" "$dbpath" "$res" "$TMP/w" -e "$EVALUE" --threads "$SLURM_CPUS_PER_TASK" -v 1
    fi
    if [[ -e "${dbpath}_taxonomy" ]]; then
        foldseek convertalis "$q" "$dbpath" "$res" "$out.partial" --format-output "$FIELDS" \
            --threads "$SLURM_CPUS_PER_TASK" -v 1
    else   # no taxonomy (phold): convertalis refuses taxid/taxname, so fill them as 0 / unknown
        foldseek convertalis "$q" "$dbpath" "$res" "$out.notax" --format-output "${FIELDS/,taxid,taxname/}" \
            --threads "$SLURM_CPUS_PER_TASK" -v 1
        awk 'BEGIN{FS=OFS="\t"} {th=$11; $11="0"; $12="unknown"; $13=th; print}' "$out.notax" > "$out.partial"
        rm -f "$out.notax"
    fi
    mv "$out.partial" "$out"
    awk -F'\t' '{ if (!($1 in b) || $10+0 > b[$1]) { b[$1]=$10+0; l[$1]=$0 } } END { for (q in l) print l[q] }' \
        "$out" > "$S/best/$tag.$db.$mode.m8"
    awk -F'\t' '$7+0 >= 0.8 && $8+0 >= 0.8 { if (!($1 in b) || $10+0 > b[$1]) { b[$1]=$10+0; l[$1]=$0 } } END { for (q in l) print l[q] }' \
        "$out" > "$S/fullcov/$tag.$db.$mode.m8"
    echo "$(date -Is) $tag $db $mode: $(wc -l < "$S/best/$tag.$db.$mode.m8") queries with a hit, $(wc -l < "$S/fullcov/$tag.$db.$mode.m8") full-length"
    rm -rf "$res"*
}
echo "$(date -Is) $db on $(hostname); foldseek $(foldseek version)"
run qdb   qdb   struct
run qdb   qdb   seq
run qnull qnull struct
