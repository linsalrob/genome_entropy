#!/bin/bash
#SBATCH --job-name=logan_up_analysis
#SBATCH --account=pawsey1018
#SBATCH --partition=copy
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#
# Upload the derived analysis tables (analysis/v1: per-shard orfs/genes
# Parquet, histograms, model sample, six-frame and candidate tables) to
# $LOGAN_REMOTE/analysis/v1, one array element per part. Part i of k takes
# every k-th file in sorted order, so the 90 large shard tables spread evenly.
#
# As for the cache (04_upload_verify.py), a copy alone is never taken as an
# upload: each part then runs `rclone check --one-way` (size + QuickXorHash)
# over its file list, and writes qc/upload_analysis/part_<i>.ok only if that
# passes. rclone copy skips files already present with matching size and
# hash, so rerunning a part resumes it.
#
#   sbatch --array=1-6 26_upload_analysis.slurm 6

set -euo pipefail
K="$1"
I="$SLURM_ARRAY_TASK_ID"
LOGAN_ROOT="${LOGAN_ROOT:-/scratch/$PAWSEY_PROJECT/$USER/Logan}"
REMOTE="${LOGAN_REMOTE:-genome_entropy:General/LOGAN}/analysis/v1"
RCLONE="${RCLONE:-$(command -v rclone || echo /software/projects/$PAWSEY_PROJECT/$USER/miniconda3/envs_dirs/rclone/bin/rclone)}"
SRC="$LOGAN_ROOT/analysis/v1"
OUT="$LOGAN_ROOT/qc/upload_analysis"
mkdir -p "$OUT"
LIST="$OUT/part_${I}_of_${K}.lst"
(cd "$SRC" && find . -type f -printf '%P\n' | LC_ALL=C sort | awk -v k="$K" -v i="$I" '(NR - 1) % k == i - 1') > "$LIST"
echo "$(date -Is) part $I/$K: $(wc -l < "$LIST") files, $(cd "$SRC" && xargs -d '\n' du -cb < "$LIST" | tail -1 | cut -f1) bytes -> $REMOTE"
"$RCLONE" copy "$SRC" "$REMOTE" --files-from-raw "$LIST" --transfers 8 --checkers 16 --retries 5 \
    --low-level-retries 20 --stats 10m --stats-one-line -v
echo "$(date -Is) copy done; checking"
"$RCLONE" check "$SRC" "$REMOTE" --one-way --files-from-raw "$LIST" --checkers 16
date -Is > "$OUT/part_${I}_of_${K}.ok"
echo "$(date -Is) part $I/$K verified"
