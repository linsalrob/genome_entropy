#!/bin/bash
#SBATCH --job-name=logan_6f
#SBATCH --account=pawsey1018
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#
# Six-frame map step (1% of complete Prodigal+ genes), one array element per built shard.
#
#   sbatch --array=1-N 14_six_frame.slurm shards.txt

set -euo pipefail
source "${LOGAN_SCRIPTS:-$HOME/GitHubs/genome_entropy/Validation/logan/scripts}/env.sh"
shard=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
python "$LOGAN_SCRIPTS/14_six_frame.py" --shard "$shard"
