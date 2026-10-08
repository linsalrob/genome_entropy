#!/bin/bash
#SBATCH --job-name=logan_meta
#SBATCH --account=pawsey1018
#SBATCH --partition=copy
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# Fetch the two LOGAN v1.2 metadata objects that accession selection reads.
#
#   stats/logan_accessions_v1.2_SRA2025.csv.zst   SRA run metadata (~11.8 GB)
#   stats/logan-seqstats-contigs-v1.2.parquet     per-accession contig stats (~1.8 GB)
#
# Each is staged under a .partial name, its byte count checked against the
# S3 Content-Length, zstd-tested where applicable, and only then renamed.
# An existing file of the right size is kept, so rerunning is cheap.

set -euo pipefail
trap 's=$?; echo "$0: error on line $LINENO: $BASH_COMMAND"; exit $s' ERR

LOGAN_ROOT="${LOGAN_ROOT:-/scratch/$PAWSEY_PROJECT/$USER/Logan}"
BASE="https://s3.amazonaws.com/logan-pub/stats"
OUT="$LOGAN_ROOT/meta"
mkdir -p "$OUT"

fetch() {
    local name="$1" url="$BASE/$1" dest="$OUT/$1"
    local want
    want=$(curl -sSfI "$url" | awk 'tolower($1)=="content-length:"{print $2+0}')
    if [[ -s "$dest" && "$(stat -c %s "$dest")" == "$want" ]]; then
        echo "$name: present ($want bytes)"
        return
    fi
    curl -sSf --retry 5 --retry-delay 10 -o "$dest.partial" "$url"
    local got
    got=$(stat -c %s "$dest.partial")
    if [[ "$got" != "$want" ]]; then
        echo "ERROR: $name is $got bytes, S3 says $want" >&2
        exit 1
    fi
    if [[ "$name" == *.zst ]]; then
        zstd -tq "$dest.partial"
    fi
    mv "$dest.partial" "$dest"
    echo "$name: fetched $got bytes, etag $(curl -sSfI "$url" | awk 'tolower($1)=="etag:"{print $2}')"
}

date -Is
fetch logan-seqstats-contigs-v1.2.parquet
fetch logan-seqstats-contigs-v1.1.parquet      # authoritative for pre-2024 accessions
fetch logan_accessions_v1.2_SRA2025.csv.zst
sha256sum "$OUT"/logan-seqstats-contigs-v1.*.parquet "$OUT"/logan_accessions_v1.2_SRA2025.csv.zst \
    > "$OUT/SHA256SUMS"
cat "$OUT/SHA256SUMS"
date -Is
