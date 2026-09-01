#!/usr/bin/env bash
# Pre-download and cache the 3Di/12-state model on a LOGIN node.
#
# GPU compute nodes on Gadi have NO internet access, same as the CPU nodes.
# If a gpuvolta/gpua100 job tries to pull model weights from Hugging Face on
# first use, it will just hang or fail. So the model has to be cached here
# first, in a location the GPU jobs can read -- /g/data, not the default
# home-directory cache, since /home has a small quota and GPU jobs need to
# find the same cache path.
#
# A thin wrapper around ../../PBS/download_model.sh, pinned to the prefix
# and cache path this run used. It activates the same conda prefix as the
# jobs; it previously sourced a venv that 03 no longer creates.

set -euo pipefail
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${WORKDIR}/../../.." && pwd)"

# Must match ENV_PREFIX and HF_HOME in 04_run_entropy.pbs, which defaults
# HF_HOME to ${GDATA_ROOT}/hf_cache. Set for NCI project ob80.
ENV_PREFIX="${ENV_PREFIX:-/g/data/ob80/re3494/conda/genome_entropy}"
GDATA_ROOT="${GDATA_ROOT:-/g/data/ob80/re3494/gtdb_entropy}"
HF_HOME="${HF_HOME:-${GDATA_ROOT}/hf_cache}"
MODEL="${MODEL:-gbouras13/modernprost-50M}"   # default dual-head (3Di + 12-state)

if [[ ! -f "${REPO_ROOT}/PBS/download_model.sh" ]]; then
    echo "ERROR: ${REPO_ROOT}/PBS/download_model.sh not found." >&2
    echo "Run this from a genome_entropy checkout." >&2
    exit 1
fi

bash "${REPO_ROOT}/PBS/download_model.sh" "${ENV_PREFIX}" "${HF_HOME}" "${MODEL}"

cat <<MSG

Verify the model landed under ${HF_HOME} (not ~/.cache/huggingface),
since that is the path 04_run_entropy.pbs points GPU jobs at.

NOTE on trust_remote_code: the genome_entropy README states ModernProst
loads model-provided Python code with trust_remote_code=True. Review the
model repo (https://huggingface.co/gbouras13/modernprost-50M) and pin a
specific revision if your project needs reproducible/audited code.
MSG
