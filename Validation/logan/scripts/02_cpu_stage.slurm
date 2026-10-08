#!/bin/bash
#SBATCH --job-name=logan_cpu
#SBATCH --account=pawsey1018
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#
# One array element per accession: download, validate, Prodigal, ORFs,
# matching (02_cpu_stage.py). The accession list is a TSV with a header and an
# `accession` column; element i takes data row i.
#
#   sbatch --array=1-100 02_cpu_stage.slurm manifests/pilot_accessions.tsv [--repro-check]

set -euo pipefail
LIST="$1"; shift
source "${LOGAN_SCRIPTS:-$HOME/GitHubs/genome_entropy/Validation/logan/scripts}/env.sh"
export TMPDIR="${LOGAN_ROOT}/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

col=$(head -1 "$LIST" | tr '\t' '\n' | grep -nx accession | cut -d: -f1)
acc=$(awk -F'\t' -v r=$((SLURM_ARRAY_TASK_ID + 1)) -v c="$col" 'NR==r{print $c}' "$LIST")
[[ -n "$acc" ]] || { echo "no accession at row $SLURM_ARRAY_TASK_ID" >&2; exit 1; }
echo "$(date -Is) $acc on $(hostname)"
python "$LOGAN_SCRIPTS/02_cpu_stage.py" "$acc" "$@"
echo "$(date -Is) done"
