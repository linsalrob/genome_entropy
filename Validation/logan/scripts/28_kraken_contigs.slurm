#!/bin/bash
#SBATCH --job-name=logan_kraken
#SBATCH --account=pawsey1018
#SBATCH --partition=work
#SBATCH --exclusive
#SBATCH --mem=230G
#SBATCH --time=06:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# Kraken2 taxonomy of the query-carrying contigs (27_fetch_query_contigs).
# Database: /scratch/references/kraken2/pluspfp_20230605 (Standard plus
# protozoa, fungi and plants, so eukaryotic contigs can be recognised; the
# 148 GB hash is loaded into memory on a whole node). Default confidence (0)
# and minimum hit groups (2).
#
#   sbatch 28_kraken_contigs.slurm

set -euo pipefail
LOGAN_ROOT="${LOGAN_ROOT:-/scratch/$PAWSEY_PROJECT/$USER/Logan}"
KENV="${KENV:-/scratch/$PAWSEY_PROJECT/$USER/software/envs/kraken2}"
DB="${KRAKEN_DB:-/scratch/references/kraken2/pluspfp_20230605}"
D="$LOGAN_ROOT/qc/candidates/contigs"
ls "$D"/part_*.fa | wc -l
cat "$D"/part_*.fa > "$D/all_contigs.fa"
grep -c '>' "$D/all_contigs.fa"
"$KENV/bin/kraken2" --version | head -1
"$KENV/bin/kraken2" --db "$DB" --threads "$SLURM_CPUS_ON_NODE" --output "$D/kraken.out" \
    --report "$D/kraken.report" "$D/all_contigs.fa"
echo "$(date -Is) done: $(wc -l < "$D/kraken.out") contigs classified"
