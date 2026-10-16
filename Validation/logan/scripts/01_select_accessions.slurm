#!/bin/bash
#SBATCH --job-name=logan_select
#SBATCH --account=pawsey1018
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=110G
#SBATCH --time=03:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# Metadata -> metagenomic candidates (DuckDB over the 11.8 GB CSV). The pilot
# draw itself is cheap and is run separately (01_select_accessions.py pilot).
#
#   sbatch --wait 01_select_accessions.slurm

set -euo pipefail
source "${LOGAN_SCRIPTS:-$HOME/GitHubs/genome_entropy/Validation/logan/scripts}/env.sh"
python "$LOGAN_SCRIPTS/01_select_accessions.py" candidates --threads "$SLURM_CPUS_PER_TASK" --memory 90GB
