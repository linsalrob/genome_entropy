#!/bin/bash
#SBATCH --job-name=logan_calib
#SBATCH --account=pawsey1018-gpu
#SBATCH --partition=gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# Encoder processes per MI250X GCD: encode the first 100k ORFs of one planned
# chunk with 1, 2, 4 and 8 workers, without publishing anything. Each run
# appends aa/s to manifests/calibration.jsonl. GTDB (#92) found ~4 per V100.
#
#   sbatch --wait 03b_calibrate_workers.slurm [chunk_id]

set -euo pipefail
source "${LOGAN_SCRIPTS:-$HOME/GitHubs/genome_entropy/Validation/logan/scripts}/env.sh"
module load rocm/6.4.1
chunk="${1:-logan_metagenome_000001}"
for w in 1 2 4 8; do
    python "$LOGAN_SCRIPTS/03_encode_chunk.py" "$chunk" --workers "$w" --limit 100000
done
