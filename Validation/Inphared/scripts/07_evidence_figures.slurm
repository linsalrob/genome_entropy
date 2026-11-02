#!/bin/bash
#SBATCH --job-name=phage_report
#SBATCH --account=pawsey1018
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=58G
#SBATCH --time=02:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Foldseek evidence tables (07_evidence.py, skipped with "figures-only") and
# all figures (08_figures.py); committed outputs go to the checkout.
#   sbatch 07_evidence_figures.slurm [intrinsic] [figures-only]
set -euo pipefail
source "${PHAGE_REPO:?set PHAGE_REPO to the checkout}/Validation/Inphared/scripts/env.sh"
fs="${1:-intrinsic}"
T="$PHAGE_REPO/Validation/Inphared/tables"
orig=()
[[ -d "$PHAGE_WORK/rank_orfs_orig" ]] && orig=(--orig "$PHAGE_WORK/rank_orfs_orig")
if [[ "${2:-}" != figures-only ]]; then
    python "$PHAGE_SCRIPTS/07_evidence.py" --dir "$PHAGE_WORK/foldseek_$fs" --tables "$PHAGE_WORK/tables" \
        --rank "$PHAGE_WORK/rank_orfs_relabelled_$fs" "${orig[@]}" --out "$T"
fi
[[ "${2:-}" == evidence-only ]] && exit 0
python "$PHAGE_SCRIPTS/08_figures.py" --work "$PHAGE_WORK" --fs "$fs" \
    --figures "$PHAGE_REPO/Validation/Inphared/figures" --tables "$T"
