#!/bin/bash
#SBATCH --job-name=ge_mi_aggregate
#SBATCH --account=pawsey1018-gpu
#SBATCH --partition=gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --time=0-4:00
#SBATCH --ntasks=1
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail
shopt -s nullglob

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 RESULTS_DIRECTORY OUTPUT.tsv LEGACY_LABELS.json.gz" >&2
    exit 2
fi

RESULTS_DIRECTORY=$1
OUTPUT=$2
LABELS=$3
VENV_DIR="/scratch/$PAWSEY_PROJECT/$USER/software/pip/genome_entropy"
INPUTS=("${RESULTS_DIRECTORY}"/*.json.gz)

if [[ ${#INPUTS[@]} -eq 0 ]]; then
    echo "No JSON inputs found in ${RESULTS_DIRECTORY}" >&2
    exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Please run slurm/rocm/install.slurm to create ${VENV_DIR}" >&2
    exit 1
fi

source "${VENV_DIR}/bin/activate"
python scripts/extract_structural_entropy_tsv.py \
    "${INPUTS[@]}" "$OUTPUT" --labels "$LABELS"
