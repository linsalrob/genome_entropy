#!/usr/bin/env bash
# Run this FIRST, on a Gadi LOGIN node (not a compute node — compute nodes
# have no internet access; only copyq and login nodes do). It downloads 5
# representative genomes and runs genome_entropy on them so you can catch
# tool-installation / API problems on a small scale instead of after
# burning cluster hours in a PBS array job.
#
# This also doubles as a rough timer: note how long step 3 (download) and
# step 5 (entropy) take for 5 genomes — that's what you'll use to size
# CHUNK_SIZE for 02_download_genomes.pbs so each copyq array task finishes
# comfortably inside its 10-hour cap.

set -euo pipefail
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${WORKDIR}"

# Everything runs out of one conda env on /g/data. Not ~/.local: /home is
# capped at 10GB on Gadi and the CUDA torch stack alone exceeds that. Not
# the default python3 either, which is 3.9 here while genome_entropy needs
# 3.10+. Build it with 03_install_genome_entropy.sh.
ENV_PREFIX="${ENV_PREFIX:-/g/data/ob80/re3494/conda/genome_entropy}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

# get_orfs is an external C binary, not a pip dependency. Build it with
# CMake (see 03_install_genome_entropy.sh) -- it is not a Rust project,
# whatever the upstream docs used to say.
export GET_ORFS_PATH="${GET_ORFS_PATH:-/g/data/ob80/re3494/Projects/genome_entropy/get_orfs/bin/get_orfs}"

# Cache the model on /g/data from the outset, so the smoke test populates
# the very cache the GPU jobs will later read offline, rather than filling
# the 10GB /home quota with a copy nothing else uses.
export HF_HOME="${HF_HOME:-/g/data/ob80/re3494/gtdb_entropy/hf_cache}"
mkdir -p "${HF_HOME}"

# NCBI datasets names every GenBank file "genomic.gbff" and puts the
# accession in the parent directory:
#     .../ncbi_dataset/data/GCA_030638685.1/genomic.gbff
# Naming outputs from the basename would therefore give every genome the
# same output file. Derive the label from the accession directory instead.
genome_label () {
    local path="$1" base
    base="$(basename "${path}")"
    base="${base%.gz}"
    if [ "${base}" = "genomic.gbff" ]; then
        basename "$(dirname "${path}")"
    else
        echo "${base%.gbff}"
    fi
}

echo "=== Step 1: checking required tools ==="
for tool in curl datasets python3 genome_entropy; do
    command -v "${tool}" >/dev/null 2>&1 || { echo "MISSING: ${tool} — run 03_install_genome_entropy.sh first."; exit 1; }
done
[ -x "${GET_ORFS_PATH}" ] || { echo "MISSING: get_orfs at ${GET_ORFS_PATH} — build it (CMake) first."; exit 1; }
echo "OK: curl, datasets, python3, genome_entropy, get_orfs all found."

echo ""
echo "=== Step 2: fetching a handful of GTDB representative accessions ==="
mkdir -p smoke_test
curl -fsSL "https://data.gtdb.ecogenomic.org/releases/latest/ar53_metadata.tsv.gz" \
    -o smoke_test/ar53_metadata.tsv.gz
# Two steps rather than one pipeline into `head`: under `set -o pipefail`,
# head closing the pipe early makes zcat die of SIGPIPE and the whole
# pipeline return 141, which set -e then treats as a fatal error even
# though the accessions were extracted correctly.
zcat smoke_test/ar53_metadata.tsv.gz | awk -F'\t' '
    NR==1 { for (i=1;i<=NF;i++){ if($i=="accession") a=i; if($i=="gtdb_representative") r=i } next }
    $r=="t" { acc=$a; sub(/^RS_/,"",acc); sub(/^GB_/,"",acc); print acc }
' > smoke_test/all_ar53_reps.txt
head -n 5 smoke_test/all_ar53_reps.txt > smoke_test/five_accessions.txt

echo "Using these 5 accessions:"
cat smoke_test/five_accessions.txt

echo ""
echo "=== Step 3: downloading GenBank files for these 5 genomes ==="
cd smoke_test
datasets download genome accession \
    --inputfile five_accessions.txt \
    --include gbff \
    --filename five_genomes.zip
unzip -o -q five_genomes.zip -d five_genomes

find five_genomes -iname "*.gbff" -o -iname "*.gbff.gz" | sort > gbff_paths.txt
echo "Found $(wc -l < gbff_paths.txt) GenBank files:"
cat gbff_paths.txt

echo ""
echo "=== Step 4: checking the genome_entropy install ==="
# No install happens here any more. The env is built once by
# 03_install_genome_entropy.sh; installing from inside a smoke test made it
# hard to tell a tool failure apart from an environment failure.
genome_entropy --help >/dev/null && echo "genome_entropy CLI OK"
genome_entropy --version
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

echo ""
echo "=== Step 4b: checking the 'run' command's flags for 3Di/12-state ==="
echo "(Confirming the model flag name before scaling up — see comments at"
echo " the top of 04_run_entropy.pbs for why this matters.)"
genome_entropy run --help

echo ""
echo "=== Step 5: running genome_entropy (with model) on each smoke-test genome ==="
echo "NOTE: this login node likely has no GPU, so this will run on CPU and"
echo "be slow for 5 genomes — that's expected. It's here to validate the"
echo "command/flags work at all, not to time GPU performance. Time the real"
echo "per-genome GPU cost separately once you're confident this step works."
MODEL="gbouras13/modernprost-50M"
mkdir -p entropy_out
: > timings.tsv
while read -r gbff; do
    base="$(genome_label "${gbff}")"
    echo "  -> ${base}"
    start=$(date +%s)
    if genome_entropy run --genbank "${gbff}" --model "${MODEL}" \
            --output "entropy_out/${base}.json"; then
        status=ok
    else
        status=FAILED
        echo "     FAILED on ${base}"
    fi
    elapsed=$(( $(date +%s) - start ))
    printf '%s\t%s\t%s\n' "${base}" "${status}" "${elapsed}" >> timings.tsv
    echo "     ${status} in ${elapsed}s"
done < gbff_paths.txt

echo ""
echo "Per-genome CPU timings (seconds):"
column -t timings.tsv
echo "These are CPU-only login-node numbers. They bound nothing about GPU"
echo "cost -- time that separately before sizing any gpuvolta array."

echo ""
echo "=== Smoke test complete ==="
echo "Check smoke_test/entropy_out/*.json for 3di_entropy / twelve_state /"
echo "mutual_information-type fields (exact key names may vary by version —"
echo "inspect the raw JSON). If these look right, proceed to"
echo "01_get_gtdb_reps.sh, 03b_download_model.sh, and the PBS scripts."
