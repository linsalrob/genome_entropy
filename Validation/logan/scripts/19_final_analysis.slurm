#!/bin/bash
#SBATCH --job-name=logan_final
#SBATCH --account=pawsey1018
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=115G
#SBATCH --time=12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# Final all-shard analysis: histogram reduce, grouped-CV models (+ final
# scorer for the missed-gene scan), six-frame summary, figures.
#
#   sbatch --dependency=afterok:<maps> 19_final_analysis.slurm

set -euo pipefail
source "${LOGAN_SCRIPTS:-$HOME/GitHubs/genome_entropy/Validation/logan/scripts}/env.sh"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
cd "$LOGAN_ROOT"
python "$LOGAN_SCRIPTS/11_summarise_histograms.py" --out qc/analysis_final
python "$LOGAN_SCRIPTS/13_models.py" --out qc/analysis_models_final
python "$LOGAN_SCRIPTS/16_six_frame_summary.py" --out qc/six_frame_final
python "$LOGAN_SCRIPTS/15_analysis_figures.py" --models qc/analysis_models_final --out qc/figures_final
python "$LOGAN_SCRIPTS/17_six_frame_figures.py" --summary qc/six_frame_final --out qc/figures_final
echo "final analysis done"
