#!/usr/bin/env bash
# Download GTDB bacterial + archaeal metadata and extract the accessions of
# species representative genomes.
#
# Output:
#   accessions/bacteria.txt   - one NCBI accession per line (e.g. GCF_000005825.2)
#   accessions/archaea.txt    - same, for archaea
#   accessions/all.txt        - concatenation of both
#
# GTDB accession column looks like "RS_GCF_000005825.2" or "GB_GCA_...";
# the RS_/GB_ prefix (RefSeq vs GenBank source) is stripped here because the
# NCBI datasets tool wants the bare GCA_/GCF_ accession.

set -euo pipefail

BASE_URL="https://data.gtdb.ecogenomic.org/releases/latest"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${WORKDIR}/gtdb_metadata" "${WORKDIR}/accessions"
cd "${WORKDIR}/gtdb_metadata"

echo "Fetching GTDB release version..."
# The file is VERSION.txt; a bare "VERSION" path 404s.
curl -fsSL "${BASE_URL}/VERSION.txt" -o VERSION.txt || echo "(couldn't fetch VERSION.txt, continuing anyway)"
cat VERSION.txt 2>/dev/null || true

for domain in bac120 ar53; do
    fname="${domain}_metadata.tsv.gz"
    if [ ! -f "${fname}" ]; then
        echo "Downloading ${domain} metadata..."
        # GTDB serves both a generic "latest" name and a release-numbered name;
        # try the generic one first, fall back to guessing isn't reliable so
        # we just tell the user if this 404s.
        curl -fsSL "${BASE_URL}/${domain}_metadata.tsv.gz" -o "${fname}" \
            || { echo "ERROR: could not download ${BASE_URL}/${domain}_metadata.tsv.gz"; \
                 echo "Check https://gtdb.ecogenomic.org/downloads for the current filename"; \
                 exit 1; }
    fi
done

echo "Extracting representative genome accessions..."

# Column names in the GTDB metadata TSV: "accession" and "gtdb_representative".
# Use the header to find the right column indices robustly instead of
# hardcoding column numbers (GTDB has changed column order between releases).
extract_reps () {
    local infile="$1"
    local outfile="$2"
    zcat "${infile}" | awk -F'\t' '
        NR==1 {
            for (i=1; i<=NF; i++) {
                if ($i == "accession") acc_col = i
                if ($i == "gtdb_representative") rep_col = i
            }
            if (!acc_col || !rep_col) {
                print "ERROR: could not find accession/gtdb_representative columns" > "/dev/stderr"
                exit 1
            }
            next
        }
        $rep_col == "t" || $rep_col == "TRUE" || $rep_col == "True" {
            acc = $acc_col
            sub(/^RS_/, "", acc)
            sub(/^GB_/, "", acc)
            print acc
        }
    ' > "${outfile}"
}

extract_reps "bac120_metadata.tsv.gz" "${WORKDIR}/accessions/bacteria.txt"
extract_reps "ar53_metadata.tsv.gz" "${WORKDIR}/accessions/archaea.txt"

cat "${WORKDIR}/accessions/bacteria.txt" "${WORKDIR}/accessions/archaea.txt" \
    > "${WORKDIR}/accessions/all.txt"

echo ""
echo "Done."
echo "  Bacteria reps: $(wc -l < "${WORKDIR}/accessions/bacteria.txt")"
echo "  Archaea reps:  $(wc -l < "${WORKDIR}/accessions/archaea.txt")"
echo "  Total:         $(wc -l < "${WORKDIR}/accessions/all.txt")"
echo ""
echo "Next: split accessions/all.txt into chunks for the PBS download array,"
echo "e.g.:  split -l 2000 -d --additional-suffix=.txt accessions/all.txt accessions/chunk_"
