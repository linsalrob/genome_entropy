#!/usr/bin/env bash
# Cache an encoder model for offline GPU jobs on NCI Gadi. Run on a LOGIN node.
#
# No Gadi queue offers both GPUs and outbound internet: copyq has internet
# but no GPU and a hard 1-CPU/10-hour cap, while gpuvolta and gpua100 have
# GPUs and no route off the machine. A GPU job that tries to reach Hugging
# Face on first use burns its allocation waiting to fail, so the cache must
# be populated here first.
#
# The cache must sit on /g/data. The default cache lives under /home, which
# is capped at 10 GB on Gadi and is not the path a job should depend on.
#
# Usage:
#   bash PBS/download_model.sh <env prefix> <HF_HOME on /g/data> [model]

set -euo pipefail

ENV_PREFIX="${1:-}"
CACHE_DIR="${2:-}"
MODEL="${3:-gbouras13/modernprost-50M}"

if [[ -z "${ENV_PREFIX}" || -z "${CACHE_DIR}" ]]; then
    echo "Usage: bash PBS/download_model.sh <env prefix> <HF_HOME on /g/data> [model]" >&2
    exit 2
fi
case "${CACHE_DIR}" in
    /g/data/*) ;;
    *) echo "Refusing to cache outside /g/data: ${CACHE_DIR}" >&2; exit 2 ;;
esac

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

export HF_HOME="${CACHE_DIR}"
mkdir -p "${HF_HOME}"

genome_entropy download --model "${MODEL}"

echo ""
if find "${HF_HOME}" -name "config.json" -path "*$(basename "${MODEL}")*" | grep -q .; then
    echo "Cached under ${HF_HOME}:"
    du -sh "${HF_HOME}"
else
    echo "WARNING: no config.json found under ${HF_HOME}." >&2
    echo "The download may have gone to the default ~/.cache/huggingface instead." >&2
    echo "GPU jobs set HF_HUB_OFFLINE=1 and will fail if this cache is wrong." >&2
    exit 1
fi

cat <<EOF

Set the same path in every GPU job:
  export HF_HOME="${HF_HOME}"
  export HF_HUB_OFFLINE=1

ModernProst is loaded with trust_remote_code=True, so this downloads and
later executes Python from the model repository. Review it, and pin a
revision, if your project requires audited or reproducible code.
EOF
