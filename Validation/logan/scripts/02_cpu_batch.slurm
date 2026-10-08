#!/bin/bash
#SBATCH --job-name=logan_cpub
#SBATCH --account=pawsey1018
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=115G
#SBATCH --time=12:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#
# Phase B CPU stage: one array element runs PER_JOB accessions from the list,
# PARALLEL at a time (02_cpu_stage.py each). Setonix caps arrays at 1,000
# elements and `work` at 256 running jobs, so one element per accession does
# not scale to tens of thousands.
#
# Each accession is independent: a failure is logged to
# logs/cpu/<acc>.log, recorded in the element's .failed list, and does not stop
# the others. Rerunning the same element retries only accessions whose state
# is short of orfs_called. Every 100th accession of the list also gets the
# Prodigal byte-reproducibility check (the pilot ran it on all 91).
#
# Peak memory in the pilot was 7 GB for a 287 Mbp accession, so 12 x 7 GB
# fits in 115 GB with room; get_orfs threads share the 64 cores.
#
#   sbatch --array=1-250 02_cpu_batch.slurm manifests/phaseb1_accessions.tsv 100 12

set -euo pipefail
LIST="$1"; PER_JOB="${2:-100}"; PARALLEL="${3:-12}"
source "${LOGAN_SCRIPTS:-$HOME/GitHubs/genome_entropy/Validation/logan/scripts}/env.sh"
export TMPDIR="${LOGAN_ROOT}/tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR" "$LOGAN_ROOT/logs/cpu"
trap 'rm -rf "$TMPDIR"' EXIT

col=$(head -1 "$LIST" | tr '\t' '\n' | grep -nx accession | cut -d: -f1)
first=$(( (SLURM_ARRAY_TASK_ID - 1) * PER_JOB + 1 ))
# Data row r is line r+1; emit "row accession".
awk -F'\t' -v c="$col" -v a="$first" -v n="$PER_JOB" \
    'NR > 1 && NR - 1 >= a && NR - 1 < a + n {print NR - 1, $c}' "$LIST" > "$TMPDIR/todo"
echo "$(date -Is) element $SLURM_ARRAY_TASK_ID: $(wc -l < "$TMPDIR/todo") accessions, $PARALLEL at a time on $(hostname)"

run_one() {
    local row="$1" acc="$2" extra=()
    (( row % 100 == 0 )) && extra=(--repro-check)
    if python "$LOGAN_SCRIPTS/02_cpu_stage.py" "$acc" "${extra[@]}" \
            > "$LOGAN_ROOT/logs/cpu/$acc.log" 2>&1; then
        echo "ok $acc"
    else
        echo "FAILED $acc"
    fi
}
export -f run_one

xargs -P "$PARALLEL" -L 1 bash -c 'run_one "$0" "$1"' < "$TMPDIR/todo" | tee "$TMPDIR/results"
failed="logs/cpu_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.failed"
grep '^FAILED' "$TMPDIR/results" | cut -d' ' -f2 > "$failed" || true
n_fail=$(wc -l < "$failed")
echo "$(date -Is) done: $(grep -c '^ok' "$TMPDIR/results" || true) ok, $n_fail failed"
[[ "$n_fail" -eq 0 ]] && rm -f "$failed"
exit $(( n_fail > 0 ))
