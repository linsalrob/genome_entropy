#!/bin/bash
#SBATCH --job-name=ge_mi_chunk
#SBATCH --account=pawsey1018-gpu
#SBATCH --partition=gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --time=0-4:00
#SBATCH --ntasks=1
#SBATCH -o %x-%A_%a.out
#SBATCH -e %x-%A_%a.err
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 CHUNK_DIRECTORY OUTPUT_DIRECTORY" >&2
    exit 2
fi

mkdir -p "$2"

for offset in 0 1 2 3; do
    index=$((SLURM_ARRAY_TASK_ID * 4 + offset))
    chunk=$(printf '%s/phold_%04d.gbk.gz' "$1" "$index")
    output=$(printf '%s/phold_%04d.json.gz' "$2" "$index")
    [[ -f "$chunk" ]] || continue
    bash slurm/rocm/rerun_mutual_information.slurm "$chunk" "$output"
done
