#!/usr/bin/env python3
"""Pull amino-acid and 3Di sequences for named ORFs out of one chunk archive.

The entropy TSVs keep the numbers; the sequences only exist inside the
per-chunk archives of per-genome JSON (see 04_run_entropy.pbs). Foldseek
needs both strings, so anything downstream of the candidate table has to
come back through here.

The archive is a zstd-compressed tar and zstd is a stream format, so there
is no random access: getting one genome means walking the stream. This
therefore takes a whole wanted-list at once, extracts every ORF it can find
in a single pass, and stops early once all of them are accounted for.

Emits paired FASTAs whose ids are identical and in the same order, ready for
15_build_query_db.py:

    >genome|input_id|orf_id|group

Every wanted ORF that is not found is written to <tag>.missing.tsv rather
than passed over. A silently short query set would understate the hit rate
of whichever arm lost members.

  14b_extract_orf_seqs.py --archive bac_000.tar.zst --wanted wanted.tsv \
      --out-dir seqs
"""
import argparse
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pandas as pd

KEY_COLS = ["genome", "input_id", "orf_id"]


def load_wanted(path, chunk):
    df = pd.read_csv(path, sep="\t", dtype={"chunk": "str"})
    for col in KEY_COLS + ["group"]:
        if col not in df.columns:
            raise SystemExit(f"ERROR: {path} has no {col!r} column")
    # The candidate table stores domain and chunk separately ("bac", "000"),
    # and the archive is named bac_000.tar.zst. Rebuild the tag rather than
    # matching on the chunk number alone, which would pull arc rows into a
    # bac extraction.
    if {"domain", "chunk"}.issubset(df.columns):
        tags = df.domain.astype(str) + "_" + df.chunk.astype(str)
        df = df[tags == chunk]
    elif "chunk" in df.columns:
        df = df[df.chunk.astype(str) == chunk]
    wanted = {}
    for genome, input_id, orf_id, group in df[KEY_COLS + ["group"]].itertuples(index=False):
        wanted.setdefault(genome, {})[(input_id, orf_id)] = group
    return wanted, len(df)


def stream_members(archive, zstd="zstd"):
    """Yield (genome, file object) for each JSON in the archive, in order."""
    proc = subprocess.Popen([zstd, "-dq", "-c", str(archive)],
                            stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".json"):
                    continue
                genome = os.path.basename(member.name)[: -len(".json")]
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                yield genome, handle
    finally:
        # The early exit below leaves zstd writing into a pipe nobody reads;
        # closing our end and killing it avoids a hung process per chunk.
        if proc.stdout:
            proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
        proc.wait()


def extract(archive, wanted_path, out_dir, zstd, quiet):
    tag = os.path.basename(archive).split(".tar.zst")[0]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted, n_rows = load_wanted(wanted_path, tag)
    n_wanted = sum(len(v) for v in wanted.values())
    if n_wanted == 0:
        if not quiet:
            print(f"{tag}: nothing wanted from this chunk")
        return 0

    found = 0
    n_len_mismatch = 0
    seen = set()
    aa_path = out_dir / f"{tag}.aa.fasta"
    ss_path = out_dir / f"{tag}.3di.fasta"

    with open(aa_path, "w") as aa_out, open(ss_path, "w") as ss_out:
        for genome, handle in stream_members(archive, zstd):
            if genome not in wanted:
                continue
            want = wanted[genome]
            data = json.load(handle)
            for record in (data if isinstance(data, list) else [data]):
                input_id = record.get("input_id", "")
                for orf_id, feat in record.get("features", {}).items():
                    group = want.get((input_id, orf_id))
                    if group is None:
                        continue
                    aa = (feat.get("protein") or {}).get("aa_sequence") or ""
                    ss = (feat.get("three_di") or {}).get("encoding") or ""
                    if len(aa) != len(ss) or not aa:
                        # One structural state per residue, by definition. A
                        # mismatch means these two strings did not come from
                        # the same encoding of the same ORF.
                        n_len_mismatch += 1
                        continue
                    name = f"{genome}|{input_id}|{orf_id}|{group}"
                    aa_out.write(f">{name}\n{aa}\n")
                    ss_out.write(f">{name}\n{ss}\n")
                    seen.add((genome, input_id, orf_id))
                    found += 1
            if found >= n_wanted:
                break                  # every wanted ORF accounted for

    missing = [(g, i, o, grp) for g, sub in wanted.items()
               for (i, o), grp in sub.items() if (g, i, o) not in seen]
    if missing:
        pd.DataFrame(missing, columns=KEY_COLS + ["group"]).to_csv(
            out_dir / f"{tag}.missing.tsv", sep="\t", index=False)

    if not quiet:
        print(f"{tag}: {found:,} of {n_wanted:,} ORFs extracted"
              + (f", {len(missing):,} missing" if missing else "")
              + (f", {n_len_mismatch:,} length-mismatched and dropped"
                 if n_len_mismatch else ""))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--archive", required=True)
    ap.add_argument("--wanted", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--zstd", default="zstd")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    return extract(args.archive, args.wanted, args.out_dir, args.zstd,
                   args.quiet)


if __name__ == "__main__":
    sys.exit(main())
