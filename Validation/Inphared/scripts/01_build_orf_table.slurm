#!/bin/bash
#SBATCH --job-name=phage_table
#SBATCH --account=pawsey1018
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=58G
#SBATCH --time=02:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Flatten the 312 genome_entropy JSONs + GenBank into orfs/seqs/cds parquet.
#   cd $PHAGE_WORK/logs && sbatch --export=NONE,PHAGE_REPO=<checkout> <checkout>/Validation/Inphared/scripts/01_build_orf_table.slurm
set -euo pipefail
source "${PHAGE_REPO:?set PHAGE_REPO to the checkout}/Validation/Inphared/scripts/env.sh"
python "$PHAGE_SCRIPTS/01_build_orf_table.py" --json-dir "$PHAGE_WORK/genbank_split_json" \
    --gbk-dir "$PHAGE_WORK/genbank_split" --out "$PHAGE_WORK/tables" --threads "$SLURM_CPUS_PER_TASK"
ls -la "$PHAGE_WORK/tables"
