#!/bin/bash
#SBATCH --job-name=phage_case
#SBATCH --account=pawsey1018
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=58G
#SBATCH --time=04:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Case-study genome: extract (11_case_study.py), then Foldseek every candidate
# (3Di + AA, as 06_foldseek_search.slurm) against every database in the list.
#   sbatch 11_case_study.slurm NC_042013 dbs.txt
set -euo pipefail
source "${PHAGE_REPO:?set PHAGE_REPO to the checkout}/Validation/Inphared/scripts/env.sh"
G="${1:?genome accession}"
D="$PHAGE_WORK/case_study"
python "$PHAGE_SCRIPTS/11_case_study.py" --work "$PHAGE_WORK" --genome "$G" --out "$D"
python "$PHAGE_REPO/Validation/gtdb/scripts/15_build_query_db.py" --aa "$D/${G}_cand.aa.fasta" \
    --3di "$D/${G}_cand.3di.fasta" --db-dir "$D/qdb_$G" --prefix q --foldseek "$FSENV/bin/foldseek"
conda activate "$FSENV"
TMP="$PHAGE_WORK/tmp/case_$SLURM_JOB_ID"
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT
FIELDS="query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue,bits,taxid,taxname,theader"
out="$D/${G}_foldseek.m8"
: > "$out.partial"
while read -r line; do
    db="${line%%=*}"; dbpath="$FSDB/$db"; [[ "$line" == *=* ]] && dbpath="${line#*=}"
    foldseek search "$D/qdb_$G/q" "$dbpath" "$TMP/res_$db" "$TMP/w" --alignment-type 2 --sort-by-structure-bits 0 \
        -e 0.001 --threads "$SLURM_CPUS_PER_TASK" -v 1
    if [[ -e "${dbpath}_taxonomy" ]]; then
        foldseek convertalis "$D/qdb_$G/q" "$dbpath" "$TMP/res_$db" "$TMP/$db.m8" --format-output "$FIELDS" -v 1
    else   # no taxonomy (phold): fill taxid / taxname as 0 / unknown
        foldseek convertalis "$D/qdb_$G/q" "$dbpath" "$TMP/res_$db" "$TMP/$db.notax" \
            --format-output "${FIELDS/,taxid,taxname/}" -v 1
        awk 'BEGIN{FS=OFS="\t"} {th=$11; $11="0"; $12="unknown"; $13=th; print}' "$TMP/$db.notax" > "$TMP/$db.m8"
    fi
    awk -v db="$db" 'BEGIN{OFS="\t"} {print db, $0}' "$TMP/$db.m8" >> "$out.partial"
    echo "$(date -Is) $db: $(cut -f1 "$TMP/$db.m8" | sort -u | wc -l) queries with a hit"
done < "${2:?DB list}"
mv "$out.partial" "$out"
