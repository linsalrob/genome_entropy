#!/bin/bash
#SBATCH --job-name=logan_roads
#SBATCH --account=pawsey1018
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# File lists, sizes and sha256 for the LOGAN ROADS candidates (R5a-c), in the
# same format as Gadi's R1-R4 (agent-handoffs#24, checkpoint E). Run from
# $LOGAN_ROOT on Setonix:
#
#   sbatch manuscript/prep/logan_roads_manifest.sh <out_dir>
#
# R5a's chunk manifests are packed into one deterministic tar.zst (sorted
# names, fixed mtime and owner), rebuilt on every run so it always reflects
# the current manifests; identical inputs give an identical archive. Every
# requested path or glob must match something (directories must be
# non-empty), otherwise the job fails rather than checksumming an incomplete
# package. Nothing is deleted.

set -euo pipefail
OUT="$(realpath "$1")"
cd "${LOGAN_ROOT:-/scratch/$PAWSEY_PROJECT/$USER/Logan}"
mkdir -p "$OUT" roads
export LC_ALL=C

# R5a: chunk manifests as one archive, rebuilt every run
find manifests/chunks -name '*.json' -printf '%P\n' | sort > roads/chunk_manifests.list
[[ -s roads/chunk_manifests.list ]] || { echo "ERROR: no chunk manifests" >&2; exit 1; }
# --format=gnu: GNU tar 1.34 defaults to POSIX/PAX, whose headers carry each
# file's atime and ctime, so reading the files changed the archive between runs.
tar --format=gnu --sort=name --mtime='2026-10-16 00:00:00Z' --owner=0 --group=0 --numeric-owner \
    --mode='u+rw,go+r,a-x' -C manifests/chunks -cf - -T roads/chunk_manifests.list | zstd -19 -T"${SLURM_CPUS_PER_TASK:-1}" -q \
    > roads/R5a_chunk_manifests.tar.zst.partial
if cmp -s roads/R5a_chunk_manifests.tar.zst.partial roads/R5a_chunk_manifests.tar.zst; then
    rm roads/R5a_chunk_manifests.tar.zst.partial
    echo "R5a archive unchanged ($(wc -l < roads/chunk_manifests.list) manifests)"
else
    mv roads/R5a_chunk_manifests.tar.zst.partial roads/R5a_chunk_manifests.tar.zst
    echo "R5a archive rebuilt ($(wc -l < roads/chunk_manifests.list) manifests)"
fi

list() {   # name, then paths/globs relative to LOGAN_ROOT; each must match
    local name="$1"; shift
    local tmp="$OUT/$name.files.partial" missing=0 p x n
    : > "$tmp"
    for p in "$@"; do
        shopt -s nullglob
        local m=( $p )
        shopt -u nullglob
        if (( ${#m[@]} == 0 )); then
            echo "ERROR: $name: nothing matches '$p'" >&2; missing=1; continue
        fi
        for x in "${m[@]}"; do
            if [[ -d "$x" ]]; then
                n=$(find "$x" -type f | tee -a "$tmp" | wc -l)
                (( n > 0 )) || { echo "ERROR: $name: '$x' is empty" >&2; missing=1; }
            elif [[ -f "$x" ]]; then
                printf '%s\n' "$x" >> "$tmp"
            else
                echo "ERROR: $name: '$x' is not a regular file" >&2; missing=1
            fi
        done
    done
    (( missing == 0 )) || { rm -f "$tmp"; exit 1; }
    sort -u "$tmp" > "$OUT/$name.files"; rm -f "$tmp"
    xargs -d '\n' sha256sum < "$OUT/$name.files" > "$OUT/$name.SHA256SUMS"
    xargs -d '\n' stat -c '%s	%n' < "$OUT/$name.files" > "$OUT/$name.sizes"
    awk -F'\t' '{s+=$1; n++} END{printf "%d files, %.3f GB\n", n, s/1e9}' "$OUT/$name.sizes" > "$OUT/$name.total"
    echo "$name: $(cat "$OUT/$name.total")"
}

list R5a_logan_selection_provenance \
    meta/logan_metagenomic_candidates.parquet meta/candidate_ledger.tsv meta/census_librarysource_assay.tsv \
    meta/total_records.txt meta/SHA256SUMS \
    manifests/pilot_accessions.tsv manifests/phaseb1_accessions.tsv manifests/phaseb2_accessions.tsv \
    manifests/accession_status.tsv roads/R5a_chunk_manifests.tar.zst

list R5b_logan_analysis_summaries \
    analysis/v1/hist analysis/v1/sample analysis/v1/sixframe analysis/v1/candidates \
    'analysis/v1/[0-9][0-9][0-9].json' \
    qc/analysis_final qc/analysis_models_final qc/six_frame_final

list R5c_logan_missed_gene_evidence \
    qc/candidates/selection.parquet qc/candidates/selection.json qc/candidates/seqs \
    qc/candidates/search/best qc/candidates/search/fullcov \
    'qc/candidates/*.tsv' qc/candidates/evidence_per_query.parquet qc/candidates/fullcov_struct_best.parquet \
    qc/candidates/taxonomy qc/candidates/skip \
    qc/candidates/contigs/all_contigs.fa 'qc/candidates/contigs/part_*.gff' 'qc/candidates/contigs/part_*.json' \
    qc/candidates/contigs/kraken.out qc/candidates/contigs/kraken.report \
    qc/candidates_extra/selection.parquet qc/candidates_extra/selection.json qc/candidates_extra/seqs \
    qc/candidates_extra/search/best qc/candidates_extra/search/fullcov \
    'qc/candidates_extra/*.tsv' qc/candidates_extra/evidence_per_query.parquet qc/candidates_extra/lowscore
