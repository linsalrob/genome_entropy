#!/bin/bash
#SBATCH --job-name=logan_gpun
#SBATCH --account=pawsey1018-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#
# Phase B encoding: one whole GPU node per array element, eight independent
# streams, one per GCD. Each stream is one 03_encode_chunk.py process pinned
# to its GCD with ROCR_VISIBLE_DEVICES, running --workers encoder processes
# for its whole share of chunks. The node's 64 cores give each stream its
# 8-core chiplet.
#
# Array element i takes lines (i-1)*PER_NODE+1 .. i*PER_NODE of the chunk
# list and deals them round-robin to the eight streams. A failed chunk does
# not stop its stream; rerunning an element skips chunks already
# cache_validated.
#
# At ~192k aa/s per GCD and ~59 M aa per 1 M-ORF chunk (~5 min each), 96
# chunks per node (12 per stream) is ~1 h of encoding; PER_NODE can go to
# ~800 inside the 12 h limit.
#
#   sbatch --array=1-N 03_encode_node.slurm manifests/phaseb1_chunks.txt 400 4

set -euo pipefail
LIST="$1"; PER_NODE="${2:-400}"; WORKERS="${3:-4}"
source "${LOGAN_SCRIPTS:-$HOME/GitHubs/genome_entropy/Validation/logan/scripts}/env.sh"
module load rocm/6.4.1
work="$LOGAN_ROOT/tmp/gpu_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$work"
first=$(( (SLURM_ARRAY_TASK_ID - 1) * PER_NODE + 1 ))
last=$(( first + PER_NODE - 1 ))
sed -n "${first},${last}p" "$LIST" > "$work/all"
n=$(wc -l < "$work/all")
[[ "$n" -gt 0 ]] || { echo "no chunks for element $SLURM_ARRAY_TASK_ID"; exit 0; }
ngcd=$(rocm-smi --showid --csv 2>/dev/null | grep -c '^card' || echo 8)
echo "$(date -Is) element $SLURM_ARRAY_TASK_ID: $n chunks on $(hostname), $ngcd GCDs, $WORKERS workers each"

( while sleep 120; do rocm-smi --showuse --showmemuse --csv 2>/dev/null | tail -n +2 | sed "s/^/$(date +%T),/"; done ) \
    > "logs/gpu_util_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv" &
mon=$!

pids=()
for g in $(seq 0 $((ngcd - 1))); do
    awk -v g="$g" -v k="$ngcd" '(NR - 1) % k == g' "$work/all" > "$work/stream_$g"
    [[ -s "$work/stream_$g" ]] || continue
    ROCR_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 \
        python "$LOGAN_SCRIPTS/03_encode_chunk.py" --chunks-file "$work/stream_$g" \
            --workers "$WORKERS" --zstd-threads 4 \
        > "logs/gpun_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_gcd$g.log" 2>&1 &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
kill "$mon" 2>/dev/null || true
grep -h 'FAILED' logs/gpun_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_gcd*.log || true
echo "$(date -Is) done rc=$rc"
rm -rf "$work"
exit $rc
