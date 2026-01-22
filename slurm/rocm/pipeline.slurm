#!/bin/bash
#SBATCH --job-name=ge_run
#SBATCH --account=pawsey1018-gpu
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu-dev
#SBATCH --time=0-4
#SBATCH --ntasks=1
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail
trap 's=$?; echo "$0: Error on line "$LINENO": $BASH_COMMAND"; exit $s' ERR



VENV_DIR="/scratch/pawsey1018/edwa0468/software/pip/genome_entropy"
if [[ ! -e "$VENV_DIR" ]]; then 
	echo "Please run the installer to create the venv" >&2;
	exit 1;
fi

source "$VENV_DIR/bin/activate"

ENCSZ=20000

if [[ $# -eq 2 ]]; then
	echo "Running: genome_entropy run -g $1 -o $2 --multi-gpu" >&2;
	genome_entropy run -g $1 -o $2 --multi-gpu --encoding-size $ENCSZ
	exit 0
fi

if [[ $# -eq 3 ]]; then
	echo "Running: genome_entropy run -i $1 -g $2 -o $3 --multi-gpu" >&2;
	genome_entropy run -i $1 -g $2 -o $3 --multi-gpu --encoding-size $ENCSZ
	exit 0
fi

echo "Can't run with $# commands. Need either 2 or 3"
