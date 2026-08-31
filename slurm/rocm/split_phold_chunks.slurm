#!/bin/bash
#SBATCH --job-name=ge_split
#SBATCH --account=pawsey1018
#SBATCH --partition=work
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=0-2:00
#SBATCH --ntasks=1
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
set -euo pipefail

REPO_ROOT="${GENOME_ENTROPY_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -d "${REPO_ROOT}/scripts" ]]; then
    echo "Submit from a genome_entropy checkout, or set GENOME_ENTROPY_REPO" >&2
    exit 1
fi

source "/scratch/$PAWSEY_PROJECT/$USER/software/pip/genome_entropy/bin/activate"
python "${REPO_ROOT}/scripts/split_genbank_chunks.py" "$1" "$2" --records-per-chunk 250
