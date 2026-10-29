#!/bin/bash
#SBATCH --job-name=phage_groups
#SBATCH --account=pawsey1018
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=58G
#SBATCH --time=04:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# skani ANI clusters + NCBI genera -> groups/groups_{ani95,genus}.tsv for 03_models.slurm.
# Needs meta/ncbi_genus.tsv from 09_ncbi_genus.py (run on the login node: it calls NCBI).
#   sbatch 10_genome_groups.slurm
set -euo pipefail
source "${PHAGE_REPO:?set PHAGE_REPO to the checkout}/Validation/Inphared/scripts/env.sh"
SKANI="${SKANI:-/scratch/$PAWSEY_PROJECT/$USER/software/envs/skani/bin/skani}"
echo "$("$SKANI" --version)"
python "$PHAGE_SCRIPTS/10_genome_groups.py" --gbk-dir "$PHAGE_WORK/genbank_split" \
    --genus "$PHAGE_WORK/meta/ncbi_genus.tsv" --out "$PHAGE_WORK/groups" --skani "$SKANI" \
    --threads "$SLURM_CPUS_PER_TASK"
