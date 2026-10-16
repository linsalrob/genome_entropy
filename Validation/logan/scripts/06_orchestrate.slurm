#!/bin/bash
#SBATCH --job-name=logan_orch
#SBATCH --account=pawsey1018
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=4-00:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# Unattended hand-offs for one batch (06_orchestrate.py). Resubmit with the
# same arguments if it reaches the 4-day limit; it resumes from
# manifests/<batch>_orchestrator.json.
#
#   sbatch 06_orchestrate.slurm --batch phaseb1 --cpu-job <02_cpu_batch array id>

set -euo pipefail
source "${LOGAN_SCRIPTS:-$HOME/GitHubs/genome_entropy/Validation/logan/scripts}/env.sh"
python "$LOGAN_SCRIPTS/06_orchestrate.py" "$@"
