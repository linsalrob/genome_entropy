#!/bin/bash
#SBATCH --job-name=logan_qcontigs
#SBATCH --account=pawsey1018
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=14G
#SBATCH --time=06:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#
# Contigs carrying the Foldseek query ORFs, streamed from LOGAN, plus Prodigal
# (-p meta -s) on them. One array element per part:
#   sbatch --array=1-24 27_fetch_query_contigs.slurm 24

set -euo pipefail
LOGAN_ROOT="${LOGAN_ROOT:-/scratch/$PAWSEY_PROJECT/$USER/Logan}"
ENV="${LOGAN_ENV:-/scratch/$PAWSEY_PROJECT/$USER/software/envs/logan}"
export PATH="$ENV/bin:$PATH"
cd "$LOGAN_ROOT"
SCRIPTS="${LOGAN_SCRIPTS:-$HOME/GitHubs/genome_entropy/Validation/logan/scripts}"   # as every other LOGAN job
python "$SCRIPTS/27_fetch_query_contigs.py" \
    --selection qc/candidates/selection.parquet --out qc/candidates/contigs \
    --part "$SLURM_ARRAY_TASK_ID" --parts "$1" --workers 12
