#!/bin/bash
#SBATCH --job-name=ge_mi_tsv
#SBATCH --account=pawsey1018-gpu
#SBATCH --partition=gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --time=0-2:00
#SBATCH --ntasks=1
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 INPUT.json.gz OUTPUT.tsv" >&2
    exit 2
fi

INPUT=$1
OUTPUT=$2
VENV_DIR="/scratch/$PAWSEY_PROJECT/$USER/software/pip/genome_entropy"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Please run slurm/rocm/install.slurm to create ${VENV_DIR}" >&2
    exit 1
fi

source "${VENV_DIR}/bin/activate"
python scripts/extract_structural_entropy_tsv.py "${INPUT}" "${OUTPUT}"
