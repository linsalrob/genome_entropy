#!/bin/bash
#SBATCH --job-name=logan_select_cand
#SBATCH --account=pawsey1018
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=115G
#SBATCH --time=04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Summarise the missed-gene scan and draw the Foldseek query set.
set -euo pipefail
source "${LOGAN_SCRIPTS:-$HOME/GitHubs/genome_entropy/Validation/logan/scripts}/env.sh"
cd "$LOGAN_ROOT"
python "$LOGAN_SCRIPTS/21_candidates_select.py" \
    --thresholds qc/analysis_models_final/final_model_thresholds.json --out qc/candidates --n 20000
