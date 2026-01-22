#!/bin/bash
#SBATCH --job-name=ge_dld
#SBATCH --account=pawsey1018-gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-dev
#SBATCH --time=0-2
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

genome_entropy download
