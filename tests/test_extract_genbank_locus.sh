#!/bin/sh
set -eu

program=${1:-bin/extract_genbank_locus}
case "$program" in
    /*) ;;
    *) program="$(pwd)/$program" ;;
esac

test_dir=$(mktemp -d "${TMPDIR:-/tmp}/extract-genbank-locus.XXXXXX")
trap 'rm -rf "$test_dir"' EXIT HUP INT TERM
cd "$test_dir"

printf '%s\n' \
    'LOCUS       PREFIX             10 bp    DNA' \
    'DEFINITION  prefix record.' \
    'ORIGIN' \
    '        1 aaaaaaaaaa' \
    '//' \
    'LOCUS       PREFIX_LONG        12 bp    DNA' \
    'DEFINITION  exact record.' \
    'ORIGIN' \
    '        1 cccccccccccc' \
    '//' \
    'LOCUS       AFTER              8 bp    DNA' \
    'DEFINITION  must not be copied.' \
    '//' > records.gb

"$program" records.gb PREFIX_LONG extracted.gb
printf '%s\n' \
    'LOCUS       PREFIX_LONG        12 bp    DNA' \
    'DEFINITION  exact record.' \
    'ORIGIN' \
    '        1 cccccccccccc' \
    '//' > expected.gb
cmp expected.gb extracted.gb

if "$program" records.gb MISSING missing.gb 2>/dev/null; then
    echo "missing LOCUS unexpectedly succeeded" >&2
    exit 1
fi
test ! -e missing.gb

printf 'LOCUS       CRLF_ID       4 bp    DNA\r\nORIGIN\r\n        1 acgt\r\n//\r\ntrailing data\r\n' > crlf.gb
"$program" crlf.gb CRLF_ID crlf-output.gb
printf 'LOCUS       CRLF_ID       4 bp    DNA\r\nORIGIN\r\n        1 acgt\r\n//\r\n' > crlf-expected.gb
cmp crlf-expected.gb crlf-output.gb

cp records.gb same.gb
if "$program" same.gb PREFIX same.gb 2>/dev/null; then
    echo "same input and output unexpectedly succeeded" >&2
    exit 1
fi
cmp records.gb same.gb

printf '%s\n' 'extract_genbank_locus tests passed'
