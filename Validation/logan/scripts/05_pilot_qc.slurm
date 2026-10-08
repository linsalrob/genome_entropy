#!/bin/bash
#SBATCH --job-name=logan_qc
#SBATCH --account=pawsey1018
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# Pilot acceptance checks, Prodigal+/- separation and cost projection.
#
#   sbatch --wait 05_pilot_qc.slurm --batch pilot

set -euo pipefail
source "${LOGAN_SCRIPTS:-$HOME/GitHubs/genome_entropy/Validation/logan/scripts}/env.sh"
python "$LOGAN_SCRIPTS/05_pilot_qc.py" "$@"
