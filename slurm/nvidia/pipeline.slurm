#!/bin/bash
#SBATCH --job-name=genome_entropy_pipeline
#SBATCH --time=0-10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --mem=20G
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail
trap 's=$?; echo "$0: Error on line "$LINENO": $BASH_COMMAND"; exit $s' ERR

eval "$(conda shell.bash hook)"
conda activate genome_entropy

if [[ $# -ne 2 ]]; then
    echo "Usage: sbatch pipeline.slurm <input genbank file> <output results>"
    exit 1
fi

export TORCH_DISABLE_DYNAMO=1
export TORCH_COMPILE_DISABLE=0
# genome_entropy run -g $1 -o $2 --multi-gpu --model gbouras13/modernprost-base
genome_entropy run -g $1 -o $2 --model gbouras13/modernprost-base




