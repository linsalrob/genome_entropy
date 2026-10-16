#!/bin/bash
#SBATCH --job-name=logan_seqs
#SBATCH --account=pawsey1018
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#
# Fetch AA + 3Di for the Foldseek query set from the cache, one array element per shard.
#
#   sbatch --array=1-N 22_extract_sequences.slurm shards.txt

set -euo pipefail
source "${LOGAN_SCRIPTS:-$HOME/GitHubs/genome_entropy/Validation/logan/scripts}/env.sh"
shard=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
python "$LOGAN_SCRIPTS/22_extract_sequences.py" --shard "$shard" --selection "${CAND_DIR:-$LOGAN_ROOT/qc/candidates}/selection.parquet"
