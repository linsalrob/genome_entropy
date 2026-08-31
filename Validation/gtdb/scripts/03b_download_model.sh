#!/usr/bin/env bash
# Pre-download and cache the 3Di/12-state model on a LOGIN node.
#
# GPU compute nodes on Gadi have NO internet access, same as the CPU nodes.
# If a gpuvolta/gpua100 job tries to pull model weights from Hugging Face on
# first use, it will just hang or fail. So the model has to be cached here
# first, in a location the GPU jobs can read — /g/data, not the default
# home-directory cache, since /home has a small quota and GPU jobs need to
# find the same cache path.

set -euo pipefail
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${WORKDIR}"

source genome_entropy_venv/bin/activate

# Shared cache location — must match HF_HOME set in 04_run_entropy.pbs.
# Set for NCI project ob80.
export HF_HOME="/g/data/ob80/re3494/gtdb_entropy/hf_cache"
mkdir -p "${HF_HOME}"

MODEL="gbouras13/modernprost-50M"   # the default dual-head (3Di + 12-state) model

echo "Caching ${MODEL} into ${HF_HOME} ..."
genome_entropy download --model "${MODEL}"

echo ""
echo "Done. Verify the model landed under ${HF_HOME} (not ~/.cache/huggingface),"
echo "since that's the path 04_run_entropy.pbs points GPU jobs at."
echo ""
echo "NOTE on trust_remote_code: the genome_entropy README states ModernProst"
echo "loads model-provided Python code with trust_remote_code=True. Review the"
echo "model repo (https://huggingface.co/gbouras13/modernprost-50M) and pin a"
echo "specific revision if your project needs reproducible/audited code."
