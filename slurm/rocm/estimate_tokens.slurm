#!/bin/bash
#SBATCH --job-name=ge_tokens
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

genome_entropy estimate-tokens --start 5000 --end 40000 --log-level DEBUG
