#!/bin/bash
#SBATCH --job-name=ge_mi_tsv
#SBATCH --account=pawsey1018
#SBATCH --partition=work
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-2:00
#SBATCH --ntasks=1
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail

if [[ $# -ne 2 && $# -ne 3 ]]; then
    echo "Usage: $0 INPUT.json.gz OUTPUT.tsv [LABELS.json.gz]" >&2
    exit 2
fi

REPO_ROOT="${GENOME_ENTROPY_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -d "${REPO_ROOT}/scripts" ]]; then
    echo "Submit from a genome_entropy checkout, or set GENOME_ENTROPY_REPO" >&2
    exit 1
fi

INPUT=$1
OUTPUT=$2
VENV_DIR="/scratch/$PAWSEY_PROJECT/$USER/software/pip/genome_entropy"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Please run slurm/rocm/install.slurm to create ${VENV_DIR}" >&2
    exit 1
fi

source "${VENV_DIR}/bin/activate"
if [[ $# -eq 3 ]]; then
    python "${REPO_ROOT}/scripts/extract_structural_entropy_tsv.py" "$INPUT" "$OUTPUT" --labels "$3"
else
    python "${REPO_ROOT}/scripts/extract_structural_entropy_tsv.py" "$INPUT" "$OUTPUT"
fi
