#!/usr/bin/env python3
"""Marginal residue composition of the ORF population, per alphabet.

WHY THIS EXISTS

`31_entropy_null.py` needs a background distribution to draw random sequences
from. The entropy rows carry per-ORF entropy but not composition, and the
sequences live only in the 1.5 TB of packed per-genome JSON, so the
composition has to be recovered from there.

It does not need much of it. A 20-symbol frequency vector is determined to
~1e-4 by a few million residues, and the residual uncertainty is biological
variation between genomes rather than sampling error. So this reads a few
genomes from every chunk -- broad and shallow -- rather than every genome from
a few chunks, and writes per-genome counts as well as the pooled vector so
that between-genome variation can be measured instead of assumed.

CONVENTION THAT MATTERS

`genome_entropy.entropy.shannon.shannon_entropy` takes no alphabet: it counts
whatever characters are present. So the null has to be drawn over exactly the
symbol set that actually occurs, including any `X` in a translation. This
script therefore counts every character it sees and reports the full symbol
set; it does not filter to the canonical 20.
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ALPHABETS = ("protein", "three_di", "twelve_state")


def _seq_for(feature, alphabet):
    if alphabet == "protein":
        return (feature.get("protein") or {}).get("aa_sequence")
    if alphabet == "three_di":
        return (feature.get("three_di") or {}).get("encoding")
    return (feature.get("twelve_state") or {}).get("encoding")


def genome_counts(records):
    """Per-alphabet Counter over every ORF in one genome's JSON."""
    counts = {a: collections.Counter() for a in ALPHABETS}
    n_orfs = 0
    lengths = []
    for record in records:
        for feature in (record.get("features") or {}).values():
            n_orfs += 1
            aa = _seq_for(feature, "protein")
            if aa:
                lengths.append(len(aa))
            for alphabet in ALPHABETS:
                seq = _seq_for(feature, alphabet)
                if seq:
                    counts[alphabet].update(seq)
    return counts, n_orfs, lengths


class ChunkReadError(RuntimeError):
    """An archive could not be read. Fatal: see read_chunk."""


def read_chunk(archive, max_genomes):
    """Stream the first `max_genomes` genome JSONs out of a chunk archive.

    Members are read in archive order and the stream is abandoned as soon as
    the quota is met, so only a prefix of the archive is decompressed.

    A read failure RAISES rather than returning what it managed to collect.
    This composition drives every simulated reference in stage 31, so quietly
    returning a short sample would bias the background and nothing downstream
    could tell. That is the defect family this run kept producing: a stage
    uses whatever inputs are present and publishes a result that reads as
    covering the whole set.
    """
    proc = subprocess.Popen(
        ["zstd", "-dc", str(archive)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    out = []
    abandoned_early = False
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".json"):
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                genome = Path(member.name).stem
                # tarfile's "r|" stream is not seekable, so TextIOWrapper
                # cannot wrap it -- read the member and decode it instead.
                out.append((genome, json.loads(handle.read().decode("utf-8"))))
                if len(out) >= max_genomes:
                    abandoned_early = True
                    break
    except (tarfile.TarError, OSError, json.JSONDecodeError) as exc:
        raise ChunkReadError(f"{archive.name}: {type(exc).__name__}: {exc}") from exc
    finally:
        try:
            proc.stdout.close()
        except OSError:
            pass
        proc.wait()

    # Abandoning the stream early makes zstd die on SIGPIPE, which is intended
    # and not an error. Reading it to the end and STILL getting a non-zero exit
    # means the archive is truncated, and that is.
    if not abandoned_early and proc.returncode not in (0, None):
        raise ChunkReadError(
            f"{archive.name}: zstd exited {proc.returncode} after the archive was "
            f"read to the end -- treating as truncated")
    if not out:
        raise ChunkReadError(f"{archive.name}: no genome JSON members found")
    return out


def process_chunk(args):
    domain, chunk, archive, max_genomes = args
    rows = []
    for genome, records in read_chunk(Path(archive), max_genomes):
        counts, n_orfs, lengths = genome_counts(records)
        rows.append((domain, chunk, genome, n_orfs, lengths, counts))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True,
                    help="entropy_3di_results/, holding <domain>/<chunk>.tar.zst")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--genomes-per-chunk", type=int, default=3)
    ap.add_argument("--bac-stride", type=int, default=10,
                    help="take every Nth bacterial chunk (760 of them)")
    ap.add_argument("--accessions", required=True,
                    help="accessions/ directory, the authoritative chunk manifest "
                         "written by 01b_make_chunks.sh")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("NPROC", 8)))
    args = ap.parse_args()

    results = Path(args.results_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # The expected chunk set comes from the accession manifests, which
    # 01b_make_chunks.sh writes and verifies -- not from whatever happens to be
    # on disk. A chunk that is missing from entropy_3di_results/ must be an
    # error, not a silently smaller sample.
    tasks = []
    for domain, stride in (("arc", 1), ("bac", args.bac_stride)):
        expected = sorted(p.stem for p in Path(args.accessions).glob(f"{domain}_*.txt"))
        if not expected:
            sys.exit(f"no {domain}_*.txt accession manifests in {args.accessions}")
        present = {p.name.split(".")[0] for p in (results / domain).glob(f"{domain}_*.tar.zst")}
        missing = [tag for tag in expected if tag not in present]
        if missing:
            sys.exit(f"{domain}: {len(missing)} of {len(expected)} chunk archives "
                     f"missing from {results / domain}, first few "
                     f"{missing[:5]} -- refusing to build a composition that "
                     f"would read as covering the domain")
        for tag in expected[::stride]:
            chunk = tag.split("_")[1]
            tasks.append((domain, chunk, str(results / domain / f"{tag}.tar.zst"),
                          args.genomes_per_chunk))
        print(f"{domain}: {len(expected)} chunks present, sampling every "
              f"{stride}", flush=True)
    print(f"{len(tasks)} chunks, {args.genomes_per_chunk} genomes each, "
          f"{args.workers} workers", flush=True)

    pooled = {d: {a: collections.Counter() for a in ALPHABETS} for d in ("arc", "bac")}
    n_genomes = collections.Counter()
    n_orfs_total = collections.Counter()
    length_hist = {d: collections.Counter() for d in ("arc", "bac")}

    failures = []
    staged = outdir / "composition_per_genome.tsv.partial"
    per_genome = staged.open("w")
    per_genome.write("domain\tchunk\tgenome\talphabet\tn_orfs\tn_residues\tsymbol\tcount\n")

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_chunk, t): t for t in tasks}
        for future in as_completed(futures):
            domain, chunk = futures[future][0], futures[future][1]
            try:
                rows = future.result()
            except Exception as exc:                       # noqa: BLE001
                failures.append(f"{domain}_{chunk}: {exc}")
                continue
            for dom, chk, genome, n_orfs, lengths, counts in rows:
                n_genomes[dom] += 1
                n_orfs_total[dom] += n_orfs
                length_hist[dom].update(lengths)
                for alphabet in ALPHABETS:
                    total = sum(counts[alphabet].values())
                    pooled[dom][alphabet].update(counts[alphabet])
                    for symbol, count in sorted(counts[alphabet].items()):
                        per_genome.write(
                            f"{dom}\t{chk}\t{genome}\t{alphabet}\t{n_orfs}\t{total}\t"
                            f"{symbol}\t{count}\n")
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(tasks)} chunks", flush=True)
    per_genome.close()

    if failures:
        staged.unlink(missing_ok=True)
        for line in failures:
            print(f"  ! {line}", file=sys.stderr)
        sys.exit(f"{len(failures)} of {len(tasks)} chunks failed to read -- "
                 f"refusing to publish a composition built from the rest")
    staged.rename(outdir / "composition_per_genome.tsv")

    with (outdir / "composition_pooled.tsv").open("w") as fh:
        fh.write("domain\talphabet\tn_genomes\tn_orfs\tn_residues\tsymbol\tcount\tfrequency\n")
        for domain in ("arc", "bac"):
            for alphabet in ALPHABETS:
                counter = pooled[domain][alphabet]
                total = sum(counter.values())
                if not total:
                    continue
                for symbol, count in sorted(counter.items()):
                    fh.write(f"{domain}\t{alphabet}\t{n_genomes[domain]}\t"
                             f"{n_orfs_total[domain]}\t{total}\t{symbol}\t{count}\t"
                             f"{count / total:.10f}\n")

    with (outdir / "sampled_length_hist.tsv").open("w") as fh:
        fh.write("domain\taa_length\tn_orfs\n")
        for domain in ("arc", "bac"):
            for length, count in sorted(length_hist[domain].items()):
                fh.write(f"{domain}\t{length}\t{count}\n")

    for domain in ("arc", "bac"):
        print(f"\n{domain}: {n_genomes[domain]} genomes, {n_orfs_total[domain]:,} ORFs")
        for alphabet in ALPHABETS:
            counter = pooled[domain][alphabet]
            total = sum(counter.values())
            if not total:
                print(f"  {alphabet:<13} EMPTY")
                continue
            top = counter.most_common(3)
            print(f"  {alphabet:<13} {len(counter):>2} symbols, {total:>13,} residues, "
                  f"top {', '.join(f'{s} {c/total:.3f}' for s, c in top)}")


if __name__ == "__main__":
    main()
