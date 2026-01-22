#!/bin/bash
#SBATCH --account=pawsey1018-gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-dev
#SBATCH --job-name=ge_install
#SBATCH --time=0-1
#SBATCH --ntasks=1
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail
trap 's=$?; echo "$0: Error on line $LINENO: $BASH_COMMAND"; exit $s' ERR

# ------------------------
# Edit this configuration - particularly the rocm version on the next line
# The project pkgs are in the pyproject.toml
# ------------------------
ROCM_MODULE_DEFAULT="rocm/6.4.1"
VENV_DIR="/scratch/pawsey1018/edwa0468/software/pip/genome_entropy"
PROJECT_PKGS=(
  genome_entropy
  "transformers>=4.30.0"
  "pygenetic_code>=0.20.0"
  "typer>=0.9.0"
  "tqdm>=4.65.0"
  "protobuf>=6.33.1"
  "sentencepiece>=0.2.1"
  "biopython>=1.80"
  "huggingface-hub<1.0,>=0.34.0"
  packaging pyyaml regex requests safetensors tokenizers click rich shellingham
)

# ------------------------
# 1) Load ROCm module
# ------------------------
module load "${ROCM_MODULE_DEFAULT}"

# Discover the loaded rocm/<version> (prefer module list; fallback to default)
loaded_rocm_mod="$(
  module -t list 2>&1 | awk -F/ '$1=="rocm"{print $0; exit 0}' || true
)"
if [[ -z "${loaded_rocm_mod}" ]]; then
  loaded_rocm_mod="${ROCM_MODULE_DEFAULT}"
fi
rocm_ver="${loaded_rocm_mod#rocm/}"  # e.g. 6.4.1

# Convert 6.4.1 -> 6.4 (PyTorch wheel channel convention)
rocm_major_minor="$(echo "${rocm_ver}" | awk -F. '{print $1 "." $2}')"
rocm_index="https://download.pytorch.org/whl/rocm${rocm_major_minor}"

echo "Loaded ROCm module : ${loaded_rocm_mod}"
echo "ROCm wheel channel : rocm${rocm_major_minor}"
echo "PyTorch index-url  : ${rocm_index}"

# ------------------------
# 2) Create venv
# ------------------------
if [[ -e "${VENV_DIR}" ]]; then
  rm -rf "${VENV_DIR}"
fi
mkdir -p "${VENV_DIR}"
python -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# ------------------------
# 3) Resolve matching torch/vision/audio versions from PyTorch guidance
#    This scrapes https://pytorch.org/get-started/previous-versions/
#    for the line containing: --index-url https://download.pytorch.org/whl/rocmX.Y
# ------------------------
PYTORCH_GUIDE_URL="https://pytorch.org/get-started/previous-versions/"

install_line="$(
  curl -fsSL "${PYTORCH_GUIDE_URL}" \
    | tr '\n' ' ' \
    | sed 's/<[^>]*>/ /g' \
    | sed 's/&nbsp;/ /g' \
    | grep -oE "pip install torch==[^ ]+ torchvision==[^ ]+ torchaudio==[^ ]+ --index-url https://download\.pytorch\.org/whl/rocm${rocm_major_minor}" \
    | head -n 1 || true
)"

if [[ -z "${install_line}" ]]; then
  echo "WARNING: Could not find a pinned torch/vision/audio trio for rocm${rocm_major_minor} on ${PYTORCH_GUIDE_URL}"
  echo "Falling back to unpinned install from ${rocm_index} (less reproducible)."
  TORCH_SPEC="torch"
  VISION_SPEC="torchvision"
  AUDIO_SPEC="torchaudio"
else
  echo "Found PyTorch recommended install line:"
  echo "  ${install_line}"
  TORCH_SPEC="$(echo "${install_line}" | grep -oE 'torch==[^ ]+' | head -n 1)"
  VISION_SPEC="$(echo "${install_line}" | grep -oE 'torchvision==[^ ]+' | head -n 1)"
  AUDIO_SPEC="$(echo "${install_line}" | grep -oE 'torchaudio==[^ ]+' | head -n 1)"
fi

echo "Installing: ${TORCH_SPEC} ${VISION_SPEC} ${AUDIO_SPEC}"
python -m pip install --index-url "${rocm_index}" "${TORCH_SPEC}" "${VISION_SPEC}" "${AUDIO_SPEC}"

# ------------------------
# 4) Install your package WITHOUT torch dependency resolution
#    (prevents pip from swapping ROCm torch for CUDA wheels)
# ------------------------
python -m pip install --no-deps "${PROJECT_PKGS[0]}"

# Then install the rest of your runtime deps explicitly (excluding torch*)
python -m pip install "${PROJECT_PKGS[@]:1}"

echo "Running pip check"
python -m pip check || true

# ------------------------
# 5) Sanity checks
# ------------------------
python - <<'PY'
import sys
print("Python:", sys.version)

try:
    import torch
    print("PyTorch:", torch.__version__)
    print("torch.version:", getattr(torch, "version", None))
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device 0:", torch.cuda.get_device_name(0))
except Exception as e:
    print("Torch check failed:", e)

try:
    import genome_entropy
    print("genome_entropy:", getattr(genome_entropy, "__version__", "unknown"))
except Exception as e:
    print("genome_entropy import failed:", e)
PY

echo "Done. Virtual env at: ${VENV_DIR}"
