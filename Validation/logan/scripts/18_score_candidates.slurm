#!/bin/bash
#SBATCH --job-name=logan_cand
#SBATCH --account=pawsey1018
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#
# Missed-gene scan: score intergenic ORFs >= 100 aa and controls, one array element per shard.
#
#   sbatch --array=1-N 18_score_candidates.slurm shards.txt

set -euo pipefail
source "${LOGAN_SCRIPTS:-$HOME/GitHubs/genome_entropy/Validation/logan/scripts}/env.sh"
shard=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
python "$LOGAN_SCRIPTS/18_score_candidates.py" --shard "$shard" --model "$LOGAN_ROOT/qc/analysis_models_final/final_model.joblib"
