#!/bin/bash
#SBATCH --job-name=logan_tables
#SBATCH --account=pawsey1018
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# Build analysis tables (09_build_analysis_tables.py). As an array, element
# i builds the i-th shard of the list file; otherwise pass shard ids.
#
#   sbatch --array=1-N 09_build_analysis_tables.slurm shards.txt
#   sbatch 09_build_analysis_tables.slurm --shards 000

set -euo pipefail
source "${LOGAN_SCRIPTS:-$HOME/GitHubs/genome_entropy/Validation/logan/scripts}/env.sh"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    shard=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1"); shift
    python "$LOGAN_SCRIPTS/09_build_analysis_tables.py" --shards "$shard" --procs "${TABLE_PROCS:-12}" "$@"
else
    python "$LOGAN_SCRIPTS/09_build_analysis_tables.py" --procs "${TABLE_PROCS:-12}" "$@"
fi
