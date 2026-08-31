#!/usr/bin/env bash
# One-time setup: create a venv with genome_entropy installed, on a node
# with internet access (login or data-transfer node).
#
# get_orfs (a required external dependency, not installed by pip) needs to
# be built separately and put on PATH or pointed to via GET_ORFS_PATH.

set -euo pipefail
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${WORKDIR}"

# Adjust/module-load a Python 3.10+ before this if your cluster's default
# python3 is older, e.g.:
#   module load python/3.11

python3 -m venv genome_entropy_venv
source genome_entropy_venv/bin/activate

pip install --upgrade pip

# For the 3Di/12-state ML encoding, install a CUDA-matched PyTorch BEFORE
# genome_entropy so it doesn't pull in a CPU-only build. Check Gadi's
# available CUDA module version first (`module avail cuda`) and match it,
# e.g. for CUDA 12.x on gpuvolta/gpua100:
#   module load cuda/12.3.2   # match whatever's actually available
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "genome_entropy[ml]"

echo ""
echo "Installed genome_entropy with ML extras (needed for 3Di/12-state encoding)."
echo "Now build get_orfs:"
echo "  git clone https://github.com/linsalrob/get_orfs"
echo "  cd get_orfs && make   # (check that repo's README for exact build steps"
echo "                         # and dependencies on your cluster's toolchain)"
echo "  export GET_ORFS_PATH=\$(pwd)/get_orfs   # or put the binary on PATH"
echo ""
echo "Remember: every PBS job that calls genome_entropy needs to"
echo "  source ${WORKDIR}/genome_entropy_venv/bin/activate"
echo "and have GET_ORFS_PATH (or PATH) set correctly."
