#!/usr/bin/env bash
# Split one domain's accession list into per-chunk files for the PBS arrays.
#
# Bacteria and archaea are kept in entirely separate chunk namespaces so
# that every downstream artefact -- GenBank archive, JSON archive, entropy
# TSV, summary -- belongs to exactly one domain and the two can be run,
# re-run, and analysed independently.
#
#   bash 01b_make_chunks.sh bac 400
#   bash 01b_make_chunks.sh arc 400
#
# Produces accessions/bac_000.txt ... and prints the PBS -J range to use.

set -euo pipefail
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${WORKDIR}"

DOMAIN="${1:-}"
CHUNK_SIZE="${2:-400}"

case "${DOMAIN}" in
    bac) SRC="accessions/bacteria.txt" ;;
    arc) SRC="accessions/archaea.txt" ;;
    *) echo "Usage: bash 01b_make_chunks.sh {bac|arc} [chunk_size]" >&2; exit 2 ;;
esac

if [ ! -s "${SRC}" ]; then
    echo "ERROR: ${SRC} missing or empty — run 01_get_gtdb_reps.sh first." >&2
    exit 1
fi

# Remove any previous chunks for THIS domain only, so re-chunking one
# domain never disturbs the other's in-flight work.
find accessions -maxdepth 1 -name "${DOMAIN}_*.txt" -delete

# -a 3 gives three-digit suffixes, matching the %03d in the PBS scripts.
split -l "${CHUNK_SIZE}" -d -a 3 --additional-suffix=.txt \
      "${SRC}" "accessions/${DOMAIN}_"

n_chunks=$(find accessions -maxdepth 1 -name "${DOMAIN}_*.txt" | wc -l)
n_acc=$(wc -l < "${SRC}")
last=$(( n_chunks - 1 ))

# Cross-check: chunks must account for every accession, no more and no
# fewer. A silent off-by-one here would quietly drop genomes.
n_in_chunks=$(cat accessions/${DOMAIN}_*.txt | wc -l)
if [ "${n_in_chunks}" -ne "${n_acc}" ]; then
    echo "ERROR: ${n_in_chunks} accessions across chunks but ${n_acc} in ${SRC}" >&2
    exit 1
fi

echo "domain      : ${DOMAIN}"
echo "source      : ${SRC} (${n_acc} accessions)"
echo "chunk size  : ${CHUNK_SIZE}"
echo "chunks      : ${n_chunks}  (verified: ${n_in_chunks} accessions total)"
echo ""
echo "PBS array range:  -J 0-${last}"
if [ "${n_chunks}" -gt 1000 ]; then
    echo "WARNING: ${n_chunks} exceeds the 1000 max_queued limit per queue." >&2
    echo "         Submit in batches, e.g. -J 0-999 then -J 1000-${last}." >&2
fi
echo ""
echo "Then:"
echo "  qsub -v DOMAIN=${DOMAIN} -J 0-${last} 02_download_genomes.pbs"
echo "  qsub -v DOMAIN=${DOMAIN} -J 0-${last} 04_run_entropy.pbs"
