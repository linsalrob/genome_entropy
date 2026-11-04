#!/bin/bash
#SBATCH --job-name=psm_install
#SBATCH --partition=work
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH -o %x-%j.out
# Usage: sbatch --account=$PAWSEY_PROJECT 00_install_env.slurm REPO_CHECKOUT
set -euo pipefail
REPO=${1:?usage: 00_install_env.slurm REPO_CHECKOUT}
export PHAGE_SMORF_INSTALLING=1
source "$REPO/Validation/phage_smorfs/scripts/env.sh"
python -m venv --system-site-packages "$PHAGE_SMORF_VENV"
source "$PHAGE_SMORF_VENV/bin/activate"
# Non-editable, no deps: the conda env already has torch/transformers.
pip install --no-deps --no-build-isolation "$REPO"
git -C "$REPO" rev-parse HEAD > "$PHAGE_SMORF_VENV/genome_entropy_commit.txt"
python -c "import genome_entropy, torch; print(genome_entropy.__version__, genome_entropy.__file__, torch.__version__)"
python -c "from genome_entropy.orf.finder import _minimum_aa_length_for_get_orfs as f; print('get_orfs -l for 30 nt:', f(30))"
cat "$PHAGE_SMORF_VENV/genome_entropy_commit.txt"
