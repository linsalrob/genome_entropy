#!/bin/bash
#SBATCH --job-name=logan_samp
#SBATCH --account=pawsey1018
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#
# Map step: weighted class-stratified ORF sample, one array element per built shard.
#
#   sbatch --array=1-N 12_sample_orfs.slurm shards.txt

set -euo pipefail
source "${LOGAN_SCRIPTS:-$HOME/GitHubs/genome_entropy/Validation/logan/scripts}/env.sh"
shard=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
python "$LOGAN_SCRIPTS/12_sample_orfs.py" --shard "$shard"
