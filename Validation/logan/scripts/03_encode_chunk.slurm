#!/bin/bash
#SBATCH --job-name=logan_gpu
#SBATCH --account=pawsey1018-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#
# One array element per cache chunk (03_encode_chunk.py). One GCD per chunk:
# the Setonix allocation pack is 1 GCD + its 8-core chiplet + ~29 GB.
# The chunk list is a plain file of chunk ids, one per line; element i takes
# line i.
#
#   sbatch --array=1-N 03_encode_chunk.slurm manifests/chunks_to_encode.txt --workers 4

set -euo pipefail
LIST="$1"; shift
source "${LOGAN_SCRIPTS:-$HOME/GitHubs/genome_entropy/Validation/logan/scripts}/env.sh"
module load rocm/6.4.1
chunk=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
[[ -n "$chunk" ]] || { echo "no chunk at line $SLURM_ARRAY_TASK_ID" >&2; exit 1; }
echo "$(date -Is) $chunk on $(hostname)"
( while sleep 60; do rocm-smi --showuse --showmemuse --csv 2>/dev/null | tail -n +2 | sed "s/^/$(date +%T) /"; done ) > "logs/gpu_util_${SLURM_JOB_ID}.csv" &
mon=$!
python "$LOGAN_SCRIPTS/03_encode_chunk.py" "$chunk" "$@"
kill $mon 2>/dev/null || true
echo "$(date -Is) done"
