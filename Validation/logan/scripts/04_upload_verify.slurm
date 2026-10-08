#!/bin/bash
#SBATCH --job-name=logan_upload
#SBATCH --account=pawsey1018
#SBATCH --partition=copy
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# Upload every cache_validated chunk and every finished accession's tables to
# genome_entropy:General/LOGAN/, verify each object, and (with --cleanup)
# delete intermediates of accessions that are complete. See 04_upload_verify.py.
#
#   sbatch 04_upload_verify.slurm [--cleanup]

set -euo pipefail
source "${LOGAN_SCRIPTS:-$HOME/GitHubs/genome_entropy/Validation/logan/scripts}/env.sh"
"$RCLONE" version | head -1
python "$LOGAN_SCRIPTS/04_upload_verify.py" "$@"
