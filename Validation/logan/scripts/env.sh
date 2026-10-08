# Sourced by every LOGAN job: one place for paths and the environment check.
export LOGAN_ROOT="${LOGAN_ROOT:-/scratch/$PAWSEY_PROJECT/$USER/Logan}"
export LOGAN_ENV="${LOGAN_ENV:-/scratch/$PAWSEY_PROJECT/$USER/software/envs/logan}"
export HF_HOME="${HF_HOME:-/scratch/$PAWSEY_PROJECT/$USER/software/hf_logan}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LOGAN_SCRIPTS="${LOGAN_SCRIPTS:-$HOME/GitHubs/genome_entropy/Validation/logan/scripts}"
# Resolve rclone before activation replaces PATH; the user's rclone config is global.
export RCLONE="${RCLONE:-$(command -v rclone || echo /software/projects/$PAWSEY_PROJECT/$USER/miniconda3/envs_dirs/rclone/bin/rclone)}"
eval "$(conda shell.bash hook)"
conda activate "$LOGAN_ENV"
# Check by import, not version string (see Validation/gtdb CLAUDE_CODE_INSTRUCTIONS).
python -c "from genome_entropy.io.genbank import normalise_orf_interval, evaluate_orf_genbank_cds_match" \
    || { echo "LOGAN env at $LOGAN_ENV is missing or broken; rerun 00_install_env.slurm" >&2; exit 1; }
command -v prodigal >/dev/null && command -v get_orfs >/dev/null \
    || { echo "prodigal or get_orfs not on PATH" >&2; exit 1; }
