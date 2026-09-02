#!/usr/bin/env python3
"""Extract per-ORF entropy rows from genome_entropy JSON into one TSV.

Run inside 04_run_entropy.pbs, on node-local jobfs, before the JSON is
packed into a chunk archive. The point is that downstream analysis reads
these TSVs and never has to unpack a multi-gigabyte JSON archive: the JSON
keeps the sequences and encodings, the TSV keeps the numbers.

Schema 2.2.0: the top level is a list of records, one per sequence record
in the input, each with input_id and a features dict keyed by ORF id. ORF
ids are only unique within a record, so genome and input_id are both
carried on every row.

Two input modes:

  --input-dir  a directory of unpacked per-genome JSON. What
               04_run_entropy.pbs uses, on jobfs, before packing.
  --archive    a packed <tag>.tar.zst, streamed in place. For regenerating a
               chunk TSV after the fact -- the 760 bacterial TSVs written
               before contig_length was added cannot be used by
               10_missed_genes.py, and unpacking a chunk to get it costs
               ~6.9 GB of jobfs per worker.

--output ending in .gz is gzipped directly; --gzip forces it for a staging
name like <final>.tsv.gz.partial, which does not end in .gz.
"""
import argparse
import csv
import gzip
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

ENTROPY_FIELDS = (
    "dna_entropy",
    "protein_entropy",
    "three_di_entropy",
    "twelve_state_entropy",
    "three_di_twelve_state_mutual_information",
)

COLUMNS = (
    "domain",
    "chunk",
    "genome",
    "input_id",
    "orf_id",
    "start",
    "end",
    "strand",
    "aa_length",
    "in_genbank",
    *ENTROPY_FIELDS,
    # Appended, not inserted: 07_count_orfs.pbs and
    # 11_genome_annotation_status.pbs address this file by column number.
    #
    # Length of the contig this ORF was called on. Needed downstream because
    # negative-strand start/end index the reverse complement, so placing an
    # ORF on the forward genomic axis is impossible without it -- see
    # 10_missed_genes.py.
    "contig_length",
)


def load(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        data = json.load(fh)
    return [data] if isinstance(data, dict) else data


def iter_archive(archive, zstd="zstd"):
    """Yield (genome, records) for each JSON in a chunk archive, in order.

    For regenerating a chunk TSV from an archive that has already been
    packed, rather than from the unpacked JSON on jobfs. zstd is a stream
    format with no random access, so this walks the stream once; nothing is
    written to disk, which is the point -- unpacking a chunk costs ~6.9 GB
    of jobfs per worker and 48 workers would want ~330 GB.
    """
    proc = subprocess.Popen([zstd, "-dq", "-c", str(archive)],
                            stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".json"):
                    continue
                genome = os.path.basename(member.name).split(".json")[0]
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                data = json.load(handle)
                yield genome, ([data] if isinstance(data, dict) else data)
    finally:
        # zstd is left writing into a pipe nobody reads if the caller stops
        # early; close our end and reap it rather than leaking a process.
        if proc.stdout:
            proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
        proc.wait()
    if proc.returncode not in (0, -15):
        # A truncated archive must not look like a short chunk. The caller
        # publishes this TSV as the complete row set for the chunk.
        raise OSError(f"{zstd} exited {proc.returncode} on {archive}")


def rows_from_records(genome, records, chunk, domain):
    for record in records:
        input_id = record.get("input_id", "")
        contig_length = record.get("input_dna_length")
        for orf_id, feat in record.get("features", {}).items():
            if not isinstance(feat, dict):
                continue
            loc = feat.get("location") or {}
            protein = feat.get("protein") or {}
            meta = feat.get("metadata") or {}
            entropy = feat.get("entropy") or {}
            row = {
                "domain": domain,
                "chunk": chunk,
                "genome": genome,
                "input_id": input_id,
                "orf_id": orf_id,
                "start": loc.get("start"),
                "end": loc.get("end"),
                "strand": loc.get("strand"),
                "aa_length": protein.get("length"),
                "in_genbank": meta.get("in_genbank"),
                "contig_length": "" if contig_length is None else contig_length,
            }
            for field in ENTROPY_FIELDS:
                value = entropy.get(field)
                # Legacy encoders and schema 2.1 write these as null or omit
                # them. Leave the cell empty rather than writing 0, which
                # would silently bias any downstream mean.
                row[field] = "" if value is None else value
            yield row


def rows_for(path, chunk, domain):
    genome = os.path.basename(path).split(".json")[0]
    return rows_from_records(genome, load(path), chunk, domain)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-dir",
                     help="directory of unpacked per-genome JSON; what "
                          "04_run_entropy.pbs uses, on jobfs")
    src.add_argument("--archive",
                     help="a packed <tag>.tar.zst, streamed in place. For "
                          "regenerating a chunk TSV after the fact without "
                          "unpacking ~6.9 GB per worker")
    ap.add_argument("--chunk", required=True)
    ap.add_argument("--domain", required=True,
                    help="bac or arc; carried on every row so the two "
                         "domains stay distinguishable after any merge")
    ap.add_argument("--output", required=True)
    ap.add_argument("--gzip", action="store_true",
                    help="gzip the output whatever it is called. Required "
                         "when writing to a staging name such as "
                         "<final>.tsv.gz.partial, which does NOT end in .gz "
                         "and would otherwise be written as plain text and "
                         "then renamed to a .gz that is not gzipped")
    ap.add_argument("--failures",
                    help="write the label of every unparseable genome here, "
                         "one per line, so the caller can fold them into the "
                         "chunk's failure state")
    args = ap.parse_args()

    if args.input_dir:
        files = sorted(Path(args.input_dir).glob("*.json"))
        if not files:
            print(f"ERROR: no JSON under {args.input_dir}", file=sys.stderr)
            return 1
        sources = ((os.path.basename(str(p)).split(".json")[0], p)
                   for p in files)
        n_sources = len(files)
    else:
        if not os.path.exists(args.archive):
            print(f"ERROR: no such archive {args.archive}", file=sys.stderr)
            return 1
        sources = iter_archive(args.archive)
        n_sources = None       # not known until the stream is exhausted

    n_rows = 0
    n_genomes = 0
    bad = []
    # Gzip when asked for it, so a regenerated chunk TSV can be written
    # straight to its final .tsv.gz rather than staged and compressed.
    # --gzip is explicit rather than inferred because callers stage through
    # a .partial name, and inferring from a name the caller has appended to
    # silently produces an uncompressed file under a .gz name.
    opener = (gzip.open
              if args.gzip or str(args.output).endswith(".gz")
              else open)
    with opener(args.output, "wt", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        for genome, source in sources:
            try:
                # Buffer the genome so a source that fails partway through
                # cannot leave half its ORFs in an otherwise complete TSV.
                if args.input_dir:
                    genome_rows = list(rows_for(source, args.chunk,
                                                args.domain))
                else:
                    genome_rows = list(rows_from_records(
                        genome, source, args.chunk, args.domain))
            except (json.JSONDecodeError, OSError, ValueError) as e:
                # Count and name it rather than dropping it silently; a
                # chunk that quietly lost genomes is worse than one that
                # reports the loss.
                print(f"WARNING: could not parse {genome}: {e}",
                      file=sys.stderr)
                bad.append(genome)
                continue
            n_genomes += 1
            for row in genome_rows:
                writer.writerow(row)
                n_rows += 1
    if n_sources is None:
        n_sources = n_genomes + len(bad)

    if args.failures:
        with open(args.failures, "w") as fh:
            for label in bad:
                fh.write(f"{label}\n")

    print(f"chunk {args.domain}_{args.chunk}: {n_genomes} genomes, {n_rows} ORF rows"
          f"{f', {len(bad)} unparseable' if bad else ''}")

    # Any unreadable genome means this TSV is missing science the chunk was
    # supposed to produce, and downstream aggregation reads only the TSV.
    # Exit non-zero for even one, so the caller cannot finalise the chunk as
    # clean -- which in 04_run_entropy.pbs would also let it delete the
    # GenBank archive the genome would have to be recovered from.
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
