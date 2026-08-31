#!/bin/bash
#SBATCH --account=pawsey1018-gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-dev
#SBATCH --job-name=ge_plot_deps
#SBATCH --time=0-1:00
#SBATCH --ntasks=1
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail

VENV_DIR="/scratch/$PAWSEY_PROJECT/$USER/software/pip/genome_entropy"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Please run slurm/rocm/install.slurm to create ${VENV_DIR}" >&2
    exit 1
fi

source "${VENV_DIR}/bin/activate"
python -m pip install "matplotlib>=3.8" "seaborn>=0.13"
python -c 'import matplotlib, seaborn; print("matplotlib", matplotlib.__version__, "seaborn", seaborn.__version__)'
