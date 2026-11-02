#!/bin/bash
#SBATCH --job-name=phage_models
#SBATCH --account=pawsey1018
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=58G
#SBATCH --time=12:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Grouped-CV feature-set comparison (03_models.py).
#   sbatch 03_models.slurm json       labels as stored in the JSONs (original run)
#   sbatch 03_models.slurm relabelled labels recomputed by 02_relabel.py
set -euo pipefail
source "${PHAGE_REPO:?set PHAGE_REPO to the checkout}/Validation/Inphared/scripts/env.sh"
#   sbatch 03_models.slurm relabelled <groups.tsv> <tag> [sets]
#                         hold out whole groups (genus, ANI cluster); output models_relabelled_<tag>
mode="${1:?json or relabelled}"
extra=()
[[ "$mode" == relabelled ]] && extra=(--labels "$PHAGE_WORK/tables/labels.parquet")
out="$PHAGE_WORK/models_$mode"
if [[ -n "${2:-}" ]]; then
    extra+=(--groups "$2" --sets "${4:-intrinsic,length_only}")
    out="${out}_${3:?tag for the grouped run}"
fi
python -W ignore::FutureWarning "$PHAGE_SCRIPTS/03_models.py" --orfs "$PHAGE_WORK/tables/orfs.parquet" \
    --original "$PHAGE_WORK/rank_orfs_orig/all_orf_oof_predictions.parquet" \
    --out "$out" --threads "$SLURM_CPUS_PER_TASK" "${extra[@]}"
