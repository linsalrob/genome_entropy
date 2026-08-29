#!/bin/bash
#SBATCH --job-name=rank_orfs
#SBATCH --account=pawsey1018
#SBATCH --partition=work
#SBATCH --time=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=58G
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

# XGBoost has no AMD GPU backend, so this ranking job is CPU-only and runs on
# the CPU partition rather than reserving GPUs it cannot use.

set -euo pipefail
trap 's=$?; echo "$0: Error on line "$LINENO": $BASH_COMMAND"; exit $s' ERR

if [[ $# -ne 2 ]]; then
	echo "$0 <directory of json files> <output directory>" >&2;
	exit 1;
fi

REPO_ROOT="${GENOME_ENTROPY_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$REPO_ROOT/scripts/rank_missing_orfs.py" ]]; then
	echo "Submit from a genome_entropy checkout, or set GENOME_ENTROPY_REPO" >&2;
	exit 1;
fi

VENV_DIR="/scratch/$PAWSEY_PROJECT/$USER/software/pip/genome_entropy"
if [[ ! -e "$VENV_DIR" ]]; then
	echo "Please run the installer to create the venv" >&2;
	exit 1;
fi

source "$VENV_DIR/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python "$REPO_ROOT/scripts/rank_missing_orfs.py" \
    --json-dir "$1" \
    --output-dir "$2" \
    --folds 5 \
    --minimum-probability 0.80 \
    --minimum-aa-length 30 \
    --device cpu
