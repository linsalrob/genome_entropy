#!/bin/bash
#SBATCH --job-name=logan_install
#SBATCH --account=pawsey1018-gpu
#SBATCH --partition=gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=01:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# One environment for both stages of the LOGAN pilot (issue #102):
#
#   * Prodigal 2.6.3 and the CPU tools (duckdb, pyarrow, pandas) from conda;
#   * ROCm PyTorch from the PyTorch wheel index, installed with pip INTO the
#     conda prefix so pip never swaps it for a CUDA wheel;
#   * genome_entropy from the LOCAL CHECKOUT, not PyPI, so the encoded cache
#     can name the exact commit that produced it. The commit is written to
#     $ENV/GENOME_ENTROPY_COMMIT and every manifest copies it from there.
#
# WHY /scratch: /software is at 161k of 250k inodes and a ROCm torch install is
# tens of thousands of files. Scratch is purged after 21 days without access,
# so this script is the recovery path: rerun it and nothing else changes.
# Each job checks the environment by import before doing any work.
#
# Runs on gpu-dev so the closing check exercises a real MI250X GCD.
#
#   sbatch --wait Validation/logan/scripts/00_install_env.slurm

set -euo pipefail
trap 's=$?; echo "$0: error on line $LINENO: $BASH_COMMAND"; exit $s' ERR

REPO="${GENOME_ENTROPY_REPO:-$HOME/GitHubs/genome_entropy}"
ENV="${LOGAN_ENV:-/scratch/$PAWSEY_PROJECT/$USER/software/envs/logan}"
HF_HOME="${HF_HOME:-/scratch/$PAWSEY_PROJECT/$USER/software/hf_logan}"
ROCM_INDEX="${ROCM_INDEX:-https://download.pytorch.org/whl/rocm6.4}"
export HF_HOME

module load rocm/6.4.1

eval "$(conda shell.bash hook)"
if [[ ! -x "$ENV/bin/python" ]]; then
    rm -rf "$ENV"
    mamba create -y -p "$ENV" -c conda-forge -c bioconda \
        python=3.12 prodigal=2.6.3 zstd pigz \
        duckdb-cli python-duckdb pyarrow pandas numpy biopython \
        scikit-learn seaborn matplotlib tqdm typer
fi
conda activate "$ENV"

python -m pip install --upgrade pip
python -m pip install --index-url "$ROCM_INDEX" torch
python -m pip install "transformers>=5.14.1" "accelerate>=0.20.0" \
    "pygenetic_code>=0.20.0" "protobuf>=6.33.1" "sentencepiece>=0.2.1" \
    huggingface-hub safetensors tokenizers zstandard

# The checkout, not PyPI, and without dependency resolution so the ROCm
# torch above is left alone.
python -m pip install --no-deps --force-reinstall "$REPO"
git -C "$REPO" rev-parse HEAD > "$ENV/GENOME_ENTROPY_COMMIT"
git -C "$REPO" status --porcelain -- src > "$ENV/GENOME_ENTROPY_SRC_DIRTY"
echo "genome_entropy commit: $(cat "$ENV/GENOME_ENTROPY_COMMIT")"
if [[ -s "$ENV/GENOME_ENTROPY_SRC_DIRTY" ]]; then
    echo "WARNING: src/ has uncommitted changes; the commit does not describe the install"
fi

# Model for offline use by the encoding jobs.
genome_entropy download --model gbouras13/modernprost-50M

python - <<'PY'
import os, torch
from genome_entropy import __version__
from genome_entropy.io.genbank import normalise_orf_interval, evaluate_orf_genbank_cds_match  # noqa: F401
from genome_entropy.encode3di.modernprost import ModernProstThreeDiEncoder
print("torch", torch.__version__, "hip", torch.version.hip,
      "gpu", torch.cuda.is_available(), torch.cuda.device_count())
print("genome_entropy", __version__)
enc = ModernProstThreeDiEncoder("gbouras13/modernprost-50M", device="cuda")
out = enc.encode(["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWUAQ"])
print("3Di", out[0].three_di[:60], "...")
print("12st", (out[0].twelve_state or "")[:60], "...")
assert len(out[0].three_di) == len(out[0].twelve_state)
PY
prodigal -v 2>&1 | head -2
echo "environment ready: $ENV"
