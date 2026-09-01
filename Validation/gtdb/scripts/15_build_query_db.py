#!/usr/bin/env python3
"""Build a Foldseek query database from precomputed amino-acid and 3Di FASTAs.

Foldseek has no supported route for pre-computed 3Di input
(steineggerlab/foldseek#511). The recipe here follows Phold's
`generate_foldseek_db_from_aa_3di`
(gbouras13/phold, src/phold/features/create_foldseek_db.py), itself adapted
from the ProstT5 author's generate_foldseek_db.py
(mheinzinger/ProstT5, scripts/, see mheinzinger/ProstT5#41). Recommended by
@gbouras13 on issue #92, and preferred here over the more obvious
`foldseek base:createdb <fasta> <db>_ss --shuffle 0`.

WHY tsv2db AND NOT base:createdb:
  Both work -- verified to produce identical search output on the same
  input. But base:createdb only lines the two databases up because it
  happens to preserve input order, which has to be forced with
  `--shuffle 0` (the default is 1, and a shuffled _ss carries the right 3Di
  strings under the wrong internal ids: every alignment silently wrong, no
  error). tsv2db instead takes the numeric key explicitly, so the
  correspondence between <db> and <db>_ss is a property of the input rather
  than of a flag nobody will remember to pass. It also avoids the stray
  _ss_h, _ss.lookup and _ss.source files createdb leaves behind.

The database that comes out has <db> (amino acids), <db>_ss (3Di) and
<db>_h (identifiers) and no <db>_ca, exactly like one built by
`createdb --prostt5-model`. It therefore supports --alignment-type 0 and
the default 2 (3Di+AA), but not 1 (TMalign), TM-score or LDDT.

  15_build_query_db.py --aa aa.fasta --3di 3di.fasta --db-dir out --prefix qdb
"""
import argparse
import random
import subprocess
import sys
from pathlib import Path

# foldseek's dbtype codes, per Phold's usage: 0 amino acid, 12 generic text
# (used for the header database).
DBTYPE_AA = 0
DBTYPE_HEADER = 12


def read_fasta(path):
    """Minimal FASTA reader: returns dict id -> sequence, in file order.

    Ids are split on whitespace, matching how Foldseek and BioPython take
    record.id, so a description after the id cannot silently create two
    different keys for the same sequence.
    """
    seqs = {}
    name = None
    parts = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(parts)
                name = line[1:].split()[0]
                if name in seqs:
                    raise SystemExit(f"ERROR: duplicate id {name!r} in {path}")
                parts = []
            elif line:
                parts.append(line)
    if name is not None:
        seqs[name] = "".join(parts)
    return seqs


def pair_up(aa, three_di, quiet=False):
    """Keep only ids present in both, with equal lengths. Report every drop.

    A 3Di string must be the same length as its amino-acid sequence: it is
    one structural state per residue. A mismatch means the two files came
    from different runs, and silently keeping it would misalign every
    residue downstream, so it is dropped and counted.
    """
    kept, dropped = [], {"aa_only": 0, "3di_only": 0, "length_mismatch": 0}
    for name, seq in aa.items():
        ss = three_di.get(name)
        if ss is None:
            dropped["aa_only"] += 1
            continue
        if len(ss) != len(seq):
            dropped["length_mismatch"] += 1
            if not quiet:
                print(f"WARNING: {name}: aa {len(seq)} != 3Di {len(ss)}, dropped",
                      file=sys.stderr)
            continue
        kept.append((name, seq, ss))
    dropped["3di_only"] = sum(1 for name in three_di if name not in aa)
    return kept, dropped


def tsv2db(foldseek, in_tsv, out_db, dbtype):
    subprocess.run([foldseek, "tsv2db", str(in_tsv), str(out_db),
                    "--output-dbtype", str(dbtype), "-v", "1"], check=True)


def shuffle_3di(kept, seed):
    """Replace each 3Di string with a permutation of its own characters.

    The technical null for the pilot: identical amino acids, identical
    length, identical 3Di composition, but the order -- which is what
    carries structural signal -- destroyed. A hit that survives this is a
    hit the amino-acid side earned, or noise.
    """
    rng = random.Random(seed)
    out = []
    for name, aa, ss in kept:
        chars = list(ss)
        rng.shuffle(chars)
        out.append((name, aa, "".join(chars)))
    return out


def build(aa_path, ss_path, db_dir, prefix, foldseek, quiet, shuffle_seed=None):
    db_dir = Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)

    aa = read_fasta(aa_path)
    ss = read_fasta(ss_path)
    kept, dropped = pair_up(aa, ss, quiet)
    if shuffle_seed is not None:
        kept = shuffle_3di(kept, shuffle_seed)
        if not quiet:
            print(f"  3Di shuffled with seed {shuffle_seed} — this database is "
                  "the technical null, not a query set")
    if not kept:
        print("ERROR: no id present in both FASTAs with matching length",
              file=sys.stderr)
        return 1

    # Numeric keys, 1..N, identical across all three tables -- this is the
    # whole point of the tsv2db route.
    tsvs = {"aa": db_dir / "aa.tsv", "3di": db_dir / "3di.tsv",
            "header": db_dir / "header.tsv"}
    with open(tsvs["aa"], "w") as fa, open(tsvs["3di"], "w") as fs, \
         open(tsvs["header"], "w") as fh:
        for i, (name, seq, three) in enumerate(kept, start=1):
            fa.write(f"{i}\t{seq}\n")
            fs.write(f"{i}\t{three}\n")
            fh.write(f"{i}\t{name}\n")

    db = db_dir / prefix
    tsv2db(foldseek, tsvs["aa"], db, DBTYPE_AA)
    tsv2db(foldseek, tsvs["3di"], f"{db}_ss", DBTYPE_AA)
    tsv2db(foldseek, tsvs["header"], f"{db}_h", DBTYPE_HEADER)
    for path in tsvs.values():
        path.unlink()

    # The invariant worth asserting before any search: the amino-acid and
    # 3Di databases must hold the same keys at the same offsets. They are
    # byte-identical here because 3Di and AA have equal length per entry.
    idx_aa = Path(f"{db}.index").read_text()
    idx_ss = Path(f"{db}_ss.index").read_text()
    if idx_aa != idx_ss:
        print(f"ERROR: {db}.index and {db}_ss.index disagree -- the 3Di and "
              "amino-acid databases are not aligned; refusing to leave a "
              "database that would align the wrong pairs", file=sys.stderr)
        return 1

    if not quiet:
        n_drop = sum(dropped.values())
        print(f"{db}: {len(kept):,} entries"
              + (f", {n_drop:,} dropped ("
                 + ", ".join(f"{k}={v:,}" for k, v in dropped.items() if v)
                 + ")" if n_drop else ""))
        print(f"  index check: {db}.index == {db}_ss.index")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--aa", required=True, help="amino-acid FASTA")
    ap.add_argument("--3di", dest="three_di", required=True, help="3Di FASTA")
    ap.add_argument("--db-dir", required=True)
    ap.add_argument("--prefix", default="qdb")
    ap.add_argument("--foldseek", default="foldseek")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--shuffle-3di", type=int, default=None, metavar="SEED",
                    help="permute each 3Di string, keeping composition and "
                         "length: builds the technical null database")
    args = ap.parse_args()
    return build(args.aa, args.three_di, args.db_dir, args.prefix,
                 args.foldseek, args.quiet, args.shuffle_3di)


if __name__ == "__main__":
    sys.exit(main())
