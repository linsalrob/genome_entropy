#!/bin/bash
#SBATCH --job-name=logan_models
#SBATCH --account=pawsey1018
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=115G
#SBATCH --time=12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# Grouped-CV models: does structure add information beyond ORF length?
#   sbatch 13_models.slurm --out qc/analysis_models

set -euo pipefail
source "${LOGAN_SCRIPTS:-$HOME/GitHubs/genome_entropy/Validation/logan/scripts}/env.sh"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
python "$LOGAN_SCRIPTS/13_models.py" "$@"
