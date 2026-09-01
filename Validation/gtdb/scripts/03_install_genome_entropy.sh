#!/usr/bin/env bash
# One-time setup: build the conda environment every other script here
# activates. Run on a LOGIN node -- Gadi compute nodes have no internet.
#
# This is a thin wrapper around ../../PBS/install.sh, which is the
# maintained installer. It exists only to pin the prefix to the one this
# run used, so `bash 03_install_genome_entropy.sh` with no arguments
# produces exactly the environment 00, 00b-00e, 02, 04, and 06 expect.
#
# Earlier this script built a venv at scripts/genome_entropy_venv while
# every job did `conda activate ${ENV_PREFIX}` on a conda prefix, so
# following the documented order left the jobs with no environment at all.
#
# get_orfs is a separate external executable; see the notes printed by the
# installer.

set -euo pipefail
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${WORKDIR}/../../.." && pwd)"

# Must match the ENV_PREFIX default in the PBS jobs and 00_smoke_test.sh.
ENV_PREFIX="${ENV_PREFIX:-/g/data/ob80/re3494/conda/genome_entropy}"

if [[ ! -f "${REPO_ROOT}/PBS/install.sh" ]]; then
    echo "ERROR: ${REPO_ROOT}/PBS/install.sh not found." >&2
    echo "Run this from a genome_entropy checkout." >&2
    exit 1
fi

echo "Creating the conda environment the jobs activate:"
echo "  ${ENV_PREFIX}"
bash "${REPO_ROOT}/PBS/install.sh" "${ENV_PREFIX}"

cat <<MSG

Next: bash 03b_download_model.sh

Every job here activates this prefix by default. To use another one, export
ENV_PREFIX before running this script and before submitting any job.
MSG
