#!/usr/bin/env bash
# Create a genome_entropy environment on NCI Gadi. Run on a LOGIN node.
#
# Gadi login nodes have direct outbound internet with no proxy; compute
# nodes, including gpuvolta and gpua100, have none. Everything that needs
# the network -- conda, pip, and the model download -- therefore happens
# here rather than in a job.
#
# Usage:
#   bash PBS/install.sh /g/data/<project>/<user>/conda/genome_entropy
#
# The environment prefix must live on /g/data. /home is quota-limited to
# 10 GB on Gadi and a CUDA PyTorch installation alone exceeds that;
# /scratch is periodically swept.

set -euo pipefail

ENV_PREFIX="${1:-}"
if [[ -z "${ENV_PREFIX}" ]]; then
    echo "Usage: bash PBS/install.sh <conda env prefix on /g/data>" >&2
    exit 2
fi
case "${ENV_PREFIX}" in
    /g/data/*) ;;
    *) echo "Refusing to build outside /g/data: ${ENV_PREFIX}" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Keep conda and pip caches off /home for the same quota reason.
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$(dirname "${ENV_PREFIX}")/pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$(dirname "${ENV_PREFIX}")/pip_cache}"
mkdir -p "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}"

# genome_entropy requires Python >= 3.10. Gadi's default python3 may be
# older, and NCBI's datasets CLI has no Gadi module, so both come from
# conda here. Drop ncbi-datasets-cli if you do not need to fetch genomes.
CONDA_FRONTEND="$(command -v mamba || command -v conda)"
"${CONDA_FRONTEND}" create -y -p "${ENV_PREFIX}" -c conda-forge -c bioconda \
    python=3.11 ncbi-datasets-cli

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

# Install PyTorch before genome_entropy so the CUDA build is not replaced
# by whatever the dependency resolver would otherwise choose.
#
# gpuvolta is Tesla V100 (compute capability sm_70). Confirm the wheel you
# install still ships sm_70 kernels -- recent CUDA toolkits have been
# dropping Volta, and a wheel without sm_70 fails only once it reaches a
# GPU node, after the job has already been queued and charged. The
# verification step below checks this. gpua100 is A100 (sm_80).
pip install torch --index-url https://download.pytorch.org/whl/cu126

pip install "${REPO_ROOT}[ml]"

echo ""
echo "=== Verification ==="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("compiled CUDA:", torch.version.cuda)
arches = torch.cuda.get_arch_list()
print("arch list:", arches)
for label, arch in (("gpuvolta (V100)", "sm_70"), ("gpua100 (A100)", "sm_80")):
    print(f"  {label}: {'OK' if arch in arches else 'NOT SUPPORTED by this wheel'}")
print("CUDA available here:", torch.cuda.is_available(), "(False on a login node is expected)")
PY

genome_entropy --help >/dev/null && echo "genome_entropy CLI: OK"
command -v datasets >/dev/null && echo "datasets CLI: $(datasets --version)"

cat <<EOF

Environment ready at ${ENV_PREFIX}

Still required:
  1. get_orfs -- an external executable, not a Python dependency. It is a
     C/CMake project; Gadi's default gcc and cmake are sufficient. Build it
     on a login node and put it on PATH or set GET_ORFS_PATH.
  2. The encoder model -- run PBS/download_model.sh next. GPU nodes cannot
     reach Hugging Face, so the cache must be populated from here.

Every PBS job must then activate this prefix explicitly:
  source \$(conda info --base)/etc/profile.d/conda.sh
  conda activate ${ENV_PREFIX}
EOF
