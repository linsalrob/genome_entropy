#!/bin/bash
#SBATCH --job-name=logan_fsdb
#SBATCH --account=pawsey1018
#SBATCH --partition=copy
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# Foldseek and its target databases for the missed-gene structure search,
# following Validation/gtdb/scripts/12_foldseek_databases.pbs (#92):
# PDB100, AFDB/Swiss-Prot, CATH50, BFVD and AFDB50.
#
# `foldseek databases` downloads without resume, so each tarball is staged
# first with `curl -C -` under the exact name upstream looks for (pdb.tar.gz,
# not pdb100.tar.gz; bfvd.tar.gz); foldseek then skips the download and
# only unpacks. TMP lives next to the databases so the final mvdb is a rename,
# not a copy. Rerunning resumes and skips finished databases.
#
#   sbatch 20_foldseek_setup.slurm

set -euo pipefail
FSENV="${FSENV:-/scratch/$PAWSEY_PROJECT/$USER/software/envs/foldseek}"
FSROOT="${FSROOT:-/scratch/$PAWSEY_PROJECT/$USER/Logan/foldseek_db}"
mkdir -p "$FSROOT/tmp"
eval "$(conda shell.bash hook)"
if [[ ! -x "$FSENV/bin/foldseek" ]]; then
    mamba create -y -p "$FSENV" -c conda-forge -c bioconda foldseek
fi
conda activate "$FSENV"
foldseek version

stage() {   # url  staged-name
    local url="$1" name="$2" want
    want=$(curl -sIL "$url" | awk 'tolower($1)=="content-length:"{v=$2} END{print v+0}')
    if [[ -s "$FSROOT/tmp/$name" && $(stat -c %s "$FSROOT/tmp/$name") -eq $want ]]; then
        echo "$name staged ($want bytes)"; return
    fi
    curl -sSfL -C - --retry 10 --retry-delay 30 --retry-all-errors -o "$FSROOT/tmp/$name" "$url"
    [[ $(stat -c %s "$FSROOT/tmp/$name") -eq $want ]] || { echo "ERROR: $name size mismatch" >&2; exit 1; }
}

fetch() {   # selection  tag  url  staged-name
    local sel="$1" tag="$2" url="$3" name="$4"
    if [[ -e "$FSROOT/$tag.dbtype" ]]; then echo "$tag present"; return; fi
    stage "$url" "$name"
    foldseek databases "$sel" "$FSROOT/$tag" "$FSROOT/tmp" --threads "$SLURM_CPUS_PER_TASK"
    [[ -e "$FSROOT/$tag.dbtype" ]] || { echo "ERROR: $tag not created" >&2; exit 1; }
    rm -f "$FSROOT/tmp/$name"
    echo "$(date -Is) $tag ready"
}

W=https://foldseek.steineggerlab.workers.dev
fetch PDB                         pdb100          $W/pdb100.tar.gz          pdb.tar.gz
fetch Alphafold/Swiss-Prot        afdb_swissprot  $W/afdb_swissprot.tar.gz  afdb_swissprot.tar.gz
fetch CATH50                      cath50          $W/cath50.tar.gz          cath50.tar.gz
fetch BFVD                        bfvd            https://bfvd.steineggerlab.workers.dev/bfvd_foldseekdb.tar.gz bfvd.tar.gz
fetch Alphafold/UniProt50-minimal afdb50          $W/afdb50.tar.gz          afdb50.tar.gz
ls -la "$FSROOT" | head -40
du -sh "$FSROOT"
