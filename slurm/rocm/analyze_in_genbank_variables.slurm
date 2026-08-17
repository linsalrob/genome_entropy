#!/bin/bash
#SBATCH --job-name=ge_in_genbank_pairs
#SBATCH --account=pawsey1018-gpu
#SBATCH --partition=gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --time=0-1:00
#SBATCH --ntasks=1
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 INPUT.tsv OUTPUT_DIRECTORY" >&2
    exit 2
fi

VENV_DIR="/scratch/$PAWSEY_PROJECT/$USER/software/pip/genome_entropy"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Please run slurm/rocm/install.slurm to create ${VENV_DIR}" >&2
    exit 1
fi

source "${VENV_DIR}/bin/activate"
python scripts/analyze_in_genbank_variables.py "$1" "$2"
