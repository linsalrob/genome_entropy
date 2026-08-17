#!/bin/bash
#SBATCH --job-name=ge_mi_run
#SBATCH --account=pawsey1018-gpu
#SBATCH --partition=gpu-dev
#SBATCH --gres=gpu:8
#SBATCH --time=0-4:00
#SBATCH --ntasks=1
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 INPUT.gbk.gz OUTPUT.json.gz" >&2
    exit 2
fi

INPUT=$1
OUTPUT=$2
VENV_DIR="/scratch/$PAWSEY_PROJECT/$USER/software/pip/genome_entropy"

if [[ ! -x "${VENV_DIR}/bin/genome_entropy" ]]; then
    echo "Please run slurm/rocm/install.slurm to create ${VENV_DIR}" >&2
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT}")"
source "${VENV_DIR}/bin/activate"

FASTA="${OUTPUT%.json.gz}.fasta.gz"
python scripts/genbank_to_fasta.py "${INPUT}" "${FASTA}"

genome_entropy run \
    --input "${FASTA}" \
    --output "${OUTPUT}" \
    --model gbouras13/modernprost-50M \
    --multi-gpu \
    --encoding-size 20000 \
    --log-file "${SLURM_JOB_ID}.log"
