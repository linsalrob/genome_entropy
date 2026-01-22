#!/bin/bash
#SBATCH --job-name=encode3di
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
    echo "Usage: sbatch encoder.slurm <input protein json> <output 3di json>"
    exit 1
fi

genome_entropy encode3di -i $1 -o $2 --encoding-size 10000 --multi-gpu




