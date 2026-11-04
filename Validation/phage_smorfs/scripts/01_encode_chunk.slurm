#!/bin/bash
#SBATCH --job-name=psm_encode
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH -o %x-%A_%a.out
# Encode the 10-90 aa ORFs of one 250-genome Inphared chunk (phold_NNNN).
# Usage: sbatch --account=${PAWSEY_PROJECT}-gpu --array=I-J 01_encode_chunk.slurm REPO_CHECKOUT
set -euo pipefail
REPO=${1:?usage: 01_encode_chunk.slurm REPO_CHECKOUT}
source "$REPO/Validation/phage_smorfs/scripts/env.sh"
MIN_AA=${MIN_AA:-10}
MAX_AA=${MAX_AA:-90}
ENCODING_SIZE=${ENCODING_SIZE:-10000}   # as the Gadi >=90 aa run
idx=$(printf '%04d' "$SLURM_ARRAY_TASK_ID")
out_dir="$PHAGE_SMORF_WORK/json"
mkdir -p "$out_dir"
out="$out_dir/phold_${idx}.json.gz"
[[ -s "$out" ]] && { echo "exists: $out"; exit 0; }
tmp=$(mktemp -d "${MYSCRATCH:-/tmp}/psm_${idx}.XXXX")
trap 'rm -rf "$tmp"' EXIT
gzip -dc "$PHAGE_CHUNKS/phold_${idx}.gbk.gz" > "$tmp/in.gbk"
/usr/bin/time -v genome_entropy run --genbank "$tmp/in.gbk" \
    --model gbouras13/modernprost-50M --encoding-size "$ENCODING_SIZE" \
    --min-aa "$MIN_AA" --max-aa "$MAX_AA" \
    --output "$tmp/out.json" --log-level INFO --log-file "$tmp/run.log"
grep -E "Found [0-9]+ ORF|Skipped|ERROR" "$tmp/run.log" | tail -5 || true
gzip -c "$tmp/out.json" > "$out.part" && mv "$out.part" "$out"
mkdir -p "$PHAGE_SMORF_WORK/run_logs" && gzip -c "$tmp/run.log" > "$PHAGE_SMORF_WORK/run_logs/phold_${idx}.log.gz"
ls -la "$out"
