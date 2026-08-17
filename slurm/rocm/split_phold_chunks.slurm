#!/bin/bash
#SBATCH --job-name=ge_split
#SBATCH --account=pawsey1018-gpu
#SBATCH --partition=gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --time=0-2:00
#SBATCH --ntasks=1
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
set -euo pipefail
source "/scratch/$PAWSEY_PROJECT/$USER/software/pip/genome_entropy/bin/activate"
python scripts/split_genbank_chunks.py "$1" "$2" --records-per-chunk 250
