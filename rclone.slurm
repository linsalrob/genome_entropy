#!/bin/bash
#SBATCH --account=pawsey1018
#SBATCH --job-name=rcloneSharks
#SBATCH --time=1-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail
trap 's=$?; echo "$0: Error on line "$LINENO": $BASH_COMMAND"; exit $s' ERR

eval "$(conda shell.bash hook)"
conda activate rclone

rclone copy Validation/ RobsFiles:genome_entropy/Validation/ --transfers 12 --checkers 24 --multi-thread-streams 4 --ignore-checksum --ignore-size

