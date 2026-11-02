#!/bin/bash
#SBATCH --job-name=phage_relabel
#SBATCH --account=pawsey1018
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=58G
#SBATCH --time=02:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Recompute in_genbank with the current (0.2.0+) CDS matcher; write relabelled JSONs.
set -euo pipefail
source "${PHAGE_REPO:?set PHAGE_REPO to the checkout}/Validation/Inphared/scripts/env.sh"
python "$PHAGE_SCRIPTS/02_relabel.py" --json-dir "$PHAGE_WORK/genbank_split_json" \
    --gbk-dir "$PHAGE_WORK/genbank_split" --out-json "$PHAGE_WORK/genbank_split_json_relabelled" \
    --labels "$PHAGE_WORK/tables/labels.parquet" --threads "$SLURM_CPUS_PER_TASK" 2> >(grep -v " INFO " >&2)
