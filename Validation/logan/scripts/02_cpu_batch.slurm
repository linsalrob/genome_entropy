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
# The allocation is billed whole (billing=128 for 64 cores / 115 GB whatever
# runs in it), so it is packed: 02_cpu_pack.py keeps up to PARALLEL (default
# 40) accessions running while their estimated peak memory (pilot: 7 GB at
# 287 Mbp) stays under 90% of the allocation. phaseb1 ran a fixed 12, ~10.7
# SU per process-hour.
#
#   sbatch --array=1-250 02_cpu_batch.slurm manifests/phaseb2_accessions.tsv 100 40
# (array size x PER_JOB must cover the list; rows are dealt round-robin)
#   sbatch --array=37 02_cpu_batch.slurm manifests/phaseb2_accessions.tsv 100 40 250

set -euo pipefail
LIST="$1"; PER_JOB="${2:-100}"; PARALLEL="${3:-40}"   # optional $4: stride
source "${LOGAN_SCRIPTS:-$HOME/GitHubs/genome_entropy/Validation/logan/scripts}/env.sh"
export TMPDIR="${LOGAN_ROOT}/tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR" "$LOGAN_ROOT/logs/cpu"
trap 'rm -rf "$TMPDIR"' EXIT

col=$(head -1 "$LIST" | tr '\t' '\n' | grep -nx accession | cut -d: -f1)
# STRIDED, not contiguous: element i of k takes data rows i, i+k, i+2k, ...
# Selection manifests are sorted by size within biome, so contiguous blocks
# put all the largest assemblies in the same elements; in phaseb1, 19 such
# elements hit the 12 h limit while the same accessions took 30-40 min each
# when run alone. k, the stride, defaults to the highest array index, so a
# --array=1-250 submission strides by 250; to rerun one element alone
# (--array=37) pass the original stride as the 4th argument.
k="${4:-${SLURM_ARRAY_TASK_MAX:-1}}"
awk -F'\t' -v c="$col" -v i="$SLURM_ARRAY_TASK_ID" -v k="$k" -v n="$PER_JOB" \
    'NR > 1 && (NR - 2) % k == i - 1 && taken < n {print NR - 1, $c; taken++}' "$LIST" > "$TMPDIR/todo"
echo "$(date -Is) element $SLURM_ARRAY_TASK_ID: $(wc -l < "$TMPDIR/todo") accessions, $PARALLEL at a time on $(hostname)"

# Packed by memory (02_cpu_pack.py): up to PARALLEL accessions at once,
# with their estimated peak memory kept under 90% of the allocation.
mem_gb=$(( ${SLURM_MEM_PER_NODE:-115000} * 9 / 10 / 1024 ))
python "$LOGAN_SCRIPTS/02_cpu_pack.py" "$TMPDIR/todo" --max-procs "$PARALLEL" \
    --mem-gb "$mem_gb" | tee "$TMPDIR/results" || true
failed="logs/cpu_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.failed"
grep '^FAILED' "$TMPDIR/results" | cut -d' ' -f2 | sort -u > "$failed" || true
n_fail=$(wc -l < "$failed")
echo "$(date -Is) done: $(grep -c '^ok' "$TMPDIR/results" || true) ok, $n_fail failed"
[[ "$n_fail" -eq 0 ]] && rm -f "$failed"
exit $(( n_fail > 0 ))
