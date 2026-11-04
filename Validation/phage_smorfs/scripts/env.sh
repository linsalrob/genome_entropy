# Sourced by every phage smORF job. The LOGAN conda env supplies ROCm torch and
# the offline ModernProst cache; a venv layered on it supplies genome_entropy
# from this checkout, which has the issue #103 fix (8f127b0) that 0.3.0 lacks.
export PHAGE_SMORF_WORK="${PHAGE_SMORF_WORK:-/scratch/$PAWSEY_PROJECT/$USER/genome_entropy/work/phage_smorfs}"
export PHAGE_CHUNKS="${PHAGE_CHUNKS:-/scratch/$PAWSEY_PROJECT/$USER/genome_entropy/work/phage_mi/chunks}"
export CONDA_BASE_ENV="${CONDA_BASE_ENV:-/scratch/$PAWSEY_PROJECT/$USER/software/envs/logan}"
export PHAGE_SMORF_VENV="${PHAGE_SMORF_VENV:-/scratch/$PAWSEY_PROJECT/$USER/software/envs/phage_smorfs}"
export HF_HOME="${HF_HOME:-/scratch/$PAWSEY_PROJECT/$USER/software/hf_logan}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PATH="$HOME/software/bin:$PATH"   # get_orfs
eval "$(conda shell.bash hook)"
conda activate "$CONDA_BASE_ENV"
if [[ -z "${PHAGE_SMORF_INSTALLING:-}" ]]; then
    source "$PHAGE_SMORF_VENV/bin/activate"
    python -c "from genome_entropy.orf.finder import _minimum_aa_length_for_get_orfs" \
        || { echo "venv $PHAGE_SMORF_VENV lacks the #103 fix; rerun 00_install_env.slurm" >&2; exit 1; }
fi
command -v get_orfs >/dev/null || { echo "get_orfs not on PATH" >&2; exit 1; }
