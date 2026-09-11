#!/usr/bin/env python3
"""Score observed entropies against the length-conditioned random null.

Given an entropy and the length it was computed from, report where it falls in
the distribution of entropies of RANDOM sequences of that same length, drawn
from the population residue composition (see 31_entropy_null.py).

Outputs per observation:

  null_mean, null_sd   the null's centre and spread AT THAT LENGTH
  z                    (observed - null_mean) / null_sd
  percentile           P(H_null <= H_observed), by interpolation of the
                       tabulated quantile grid
  p_low                one-sided p for "less diverse than chance" = percentile
  p_high               one-sided p for "more diverse than chance"
  tail                 `exact` inside the tabulated range, `extrapolated`
                       outside it, where a normal approximation is used and
                       should not be read as a precise p-value

INPUT MODES

  --entropy-rows  the pipeline's own entropy_rows TSVs: scores protein,
                  three_di and twelve_state together, one output row per ORF.
  --table         any TSV with a length column and an entropy column.
  --fasta         sequences, entropy computed directly. Use --alphabet to say
                  which null to score against: `protein` for amino acids, or
                  `three_di` / `twelve_state` for an encoding FASTA. There is
                  no way to get a 3Di entropy from an amino acid sequence
                  without running the encoder, so this mode will not invent one.

A percentile near 0 means the sequence is LESS diverse than a random sequence
of its length -- compositionally biased, low-complexity, or repetitive. Near 1
means more even than chance. Most real proteins sit low: real sequence is not
an i.i.d. draw, and this null does not pretend otherwise.

WHICH BACKGROUND -- THIS MATTERS MORE FOR 3Di THAN FOR PROTEIN

The precomputed tables use one pooled background per domain. Measured over 351
sampled genomes, that is a fair summary for amino acids and a poor one for the
structural alphabets:

  alphabet       between-genome sd of H(p)      range
  protein              0.06 bits            3.77 - 4.27
  three_di             0.35 bits            1.89 - 3.63
  twelve_state         0.28 bits            1.87 - 3.43

A 1.7-bit spread in the 3Di background is far wider than the length effect the
null exists to remove, so a pooled 3Di percentile partly measures which genome
a sequence came from. Two consequences:

  - the **z-score** is the robust output. It removes the mechanical length
    trend, which is what makes entropies of different lengths comparable at
    all, and that is what it is for.
  - a **percentile or p-value against the pooled 3Di background should not be
    read as significance.** Use --background to supply the composition the
    sequence should actually be judged against (its own genome, its own
    dataset) and the null is simulated for that background instead.
"""
import argparse
import gzip
import math
import sys
from collections import Counter

import numpy as np

ROW_ALPHABETS = {
    "protein": "protein_entropy",
    "three_di": "three_di_entropy",
    "twelve_state": "twelve_state_entropy",
}


def plug_in_entropy(sequence):
    """Shannon entropy in bits over whatever symbols are present."""
    if not sequence:
        return 0.0
    counts = Counter(sequence)
    total = len(sequence)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


class QuantileGrid:
    """Shared lookup over a (length x quantile) grid.

    Both references have the same shape -- a quantile grid indexed by length --
    so the interpolation, the tail handling and the z-score live here once.
    They differ only in what the grid was built from, which is the whole point
    of reporting them side by side.
    """

    KEYS = ("lengths", "quantiles", "mean", "sd")
    OPTIONAL_KEYS = ("n",)

    def __init__(self, path, key_filter=None):
        data = np.load(path)
        self.levels = data["quantile_levels"]
        self.grids = {}
        for key in sorted({k.rsplit("|", 1)[0] for k in data.files
                           if k != "quantile_levels"}):
            if key_filter and not key_filter(key):
                continue
            self.grids[key] = {name: data[f"{key}|{name}"] for name in self.KEYS}
            for name in self.OPTIONAL_KEYS:
                if f"{key}|{name}" in data.files:
                    self.grids[key][name] = data[f"{key}|{name}"]

    def available(self):
        return sorted(self.grids)

    def _at_length(self, key, length):
        grid = self.grids[key]
        lengths = grid["lengths"]
        clipped = min(max(float(length), lengths[0]), lengths[-1])
        position = np.interp(clipped, lengths, np.arange(lengths.size))
        low, high = int(np.floor(position)), int(np.ceil(position))
        weight = position - low
        blend = lambda name: ((1 - weight) * grid[name][low]
                              + weight * grid[name][high])
        return (blend("quantiles"), float(blend("mean")), float(blend("sd")),
                clipped != float(length))

    def score(self, key, length, entropy):
        quantiles, mean, sd, clamped = self._at_length(key, length)
        z = (entropy - mean) / sd if sd > 0 else float("nan")
        if entropy <= quantiles[0] or entropy >= quantiles[-1]:
            percentile = (0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
                          if sd > 0 else float("nan"))
            tail = "extrapolated"
        else:
            percentile = float(np.interp(entropy, quantiles, self.levels))
            tail = "exact"
        if clamped:
            tail += ",length_clamped"
        return mean, sd, z, percentile, tail


class SimulatedNull(QuantileGrid):
    """i.i.d. null simulated on demand from a caller-supplied background.

    Stage 31 precomputes this for the pooled backgrounds. When the background
    is the caller's own -- one genome's composition, one dataset's -- there is
    nothing precomputed, so it is drawn here and cached per length. Cheap for
    the handful of sequences this mode is meant for; do not point it at a
    million rows.
    """

    QUANTILES = (0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
                 0.75, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9995)

    def __init__(self, probs, reps=200_000, seed=20260910):
        probs = np.asarray(probs, dtype=float)
        self.probs = probs / probs.sum()
        self.reps = reps
        self.seed = seed
        self.levels = np.array(self.QUANTILES)
        self.grids = {"custom": None}
        self._cache = {}

    def _at_length(self, key, length):
        length = max(int(length), 1)
        if length not in self._cache:
            rng = np.random.default_rng(self.seed + length)
            counts = rng.multinomial(length, self.probs, size=self.reps)
            with np.errstate(divide="ignore", invalid="ignore"):
                terms = np.where(counts > 0, counts * np.log(counts), 0.0)
            draws = np.log2(length) - terms.sum(axis=1) / (length * math.log(2.0))
            draws.sort()
            self._cache[length] = (np.quantile(draws, self.QUANTILES),
                                   float(draws.mean()), float(draws.std(ddof=1)))
        quantiles, mean, sd = self._cache[length]
        return quantiles, mean, sd, False


class NullTable(QuantileGrid):
    """i.i.d. random-sequence null from stage 31. Keys are `<domain>|<alphabet>`."""


class EmpiricalTable(QuantileGrid):
    """Real-ORF distribution from stage 33.

    Keys are `<domain>|<stratum>|<alphabet>`, so a stratum has to be chosen.
    Stratum 3 -- a genome carrying deposited CDS, ORF matched to one -- is the
    default because it is the set of ORFs that are known real genes, which is
    the reference a reader means by "compared to real proteins of that length".
    """

    def __init__(self, path, domain, stratum=3):
        self.stratum = stratum
        prefix = f"{domain}|{stratum}|"
        super().__init__(path, key_filter=lambda k: k.startswith(prefix))
        if not self.grids:
            sys.exit(f"no empirical cells for {prefix}* in {path}")


def load_background(path, domain=None, alphabet=None, genome=None):
    """Symbol frequencies from a composition TSV, as a probability vector.

    Accepts either of the two files stage 30 writes. `composition_pooled.tsv`
    has a `frequency` column; `composition_per_genome.tsv` has raw `count`,
    which is normalised here. Rows are filtered by whichever of domain,
    alphabet and genome are supplied -- all three are needed to isolate a
    single background in the per-genome file, and leaving one out silently
    pools over it, which is a different question.
    """
    weights = {}
    with open_maybe_gzip(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        index = {name: i for i, name in enumerate(header)}
        value_column = "frequency" if "frequency" in index else "count"
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if domain and fields[index["domain"]] != domain:
                continue
            if alphabet and fields[index["alphabet"]] != alphabet:
                continue
            if genome and index.get("genome") is not None \
                    and fields[index["genome"]] != genome:
                continue
            weights[fields[index["symbol"]]] = weights.get(
                fields[index["symbol"]], 0.0) + float(fields[index[value_column]])
    if not weights:
        sys.exit(f"no rows in {path} matched the requested background")
    if alphabet is None and "alphabet" in index:
        sys.exit(f"{path} carries an `alphabet` column but no alphabet was "
                 f"requested -- that would pool every alphabet into one vector")
    total = sum(weights.values())
    probs = np.array([weights[s] / total for s in sorted(weights)])
    entropy = -(probs[probs > 0] * np.log2(probs[probs > 0])).sum()
    print(f"# background: {len(probs)} symbols ({''.join(sorted(weights))}), "
          f"H(p) = {entropy:.4f} bits", file=sys.stderr)
    return probs


def open_maybe_gzip(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def iter_fasta(path):
    name, chunks = None, []
    with open_maybe_gzip(path) as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name, chunks = line[1:].split()[0], []
            elif line:
                chunks.append(line)
    if name is not None:
        yield name, "".join(chunks)


def write_header(out, extra, empirical):
    columns = extra + ["alphabet", "aa_length", "entropy", "null_mean",
                       "null_sd", "z", "percentile", "p_low", "p_high", "tail"]
    if empirical:
        columns += ["emp_mean", "emp_sd", "emp_z", "emp_percentile", "emp_n", "emp_tail"]
    out.write("\t".join(columns) + "\n")


def emit(out, prefix, alphabet, length, entropy, scored, empirical_scored=None,
         empirical_n=None):
    mean, sd, z, percentile, tail = scored
    fields = prefix + [
        alphabet, str(length), f"{entropy:.6f}", f"{mean:.6f}", f"{sd:.6f}",
        f"{z:.4f}", f"{percentile:.6g}", f"{percentile:.6g}",
        f"{1.0 - percentile:.6g}", tail]
    if empirical_scored is not None:
        e_mean, e_sd, e_z, e_pct, e_tail = empirical_scored
        fields += [f"{e_mean:.6f}", f"{e_sd:.6f}", f"{e_z:.4f}", f"{e_pct:.6g}",
                   str(empirical_n if empirical_n is not None else ""), e_tail]
    out.write("\t".join(fields) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--null", help="entropy_null.npz from stage 31 (pooled background)")
    ap.add_argument("--background",
                    help="composition TSV (symbol/frequency columns, as written by "
                         "30_residue_composition.py) to simulate the null from "
                         "instead of the pooled tables. Filtered by --background-domain "
                         "/ --background-alphabet / --background-genome when given.")
    ap.add_argument("--background-domain")
    ap.add_argument("--background-alphabet")
    ap.add_argument("--background-genome")
    ap.add_argument("--background-reps", type=int, default=200_000)
    ap.add_argument("--empirical",
                    help="length_entropy_<domain>.npz from stage 33. Adds the "
                         "REAL-ORF reference alongside whichever i.i.d. null is "
                         "in use, as emp_* columns. This is the calibrated one.")
    ap.add_argument("--empirical-stratum", type=int, default=3,
                    help="3 = ORFs matched to a deposited CDS in an annotated "
                         "genome (default: known real genes). 2 = unmatched in "
                         "an annotated genome. 0 = ORFs of unannotated genomes.")
    ap.add_argument("--domain", default="bac", choices=("bac", "arc"),
                    help="which background composition to score against")
    ap.add_argument("--output", default="-")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--entropy-rows")
    source.add_argument("--table")
    source.add_argument("--fasta")
    ap.add_argument("--alphabet", choices=sorted(ROW_ALPHABETS),
                    help="required for --fasta")
    ap.add_argument("--length-column", default="aa_length")
    ap.add_argument("--entropy-column")
    ap.add_argument("--id-column", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if bool(args.null) == bool(args.background):
        sys.exit("give exactly one of --null (pooled tables) or --background "
                 "(simulate from a supplied composition)")

    # A background belongs to exactly one alphabet. The stage 30 composition
    # file holds all three, so leaving the alphabet unset pools protein, 3Di
    # and twelve-state symbols into a single vector and every score computed
    # from it is meaningless while looking perfectly plausible. Resolve it
    # explicitly for every input mode, and refuse a mismatch.
    background_alphabet = args.background_alphabet
    if args.background:
        if args.entropy_rows:
            if not background_alphabet:
                sys.exit("--entropy-rows with --background needs "
                         "--background-alphabet: one background scores one "
                         "alphabet, not all three")
        else:
            if background_alphabet and args.alphabet \
                    and background_alphabet != args.alphabet:
                sys.exit(f"--background-alphabet {background_alphabet} does not "
                         f"match --alphabet {args.alphabet}: a background can only "
                         f"score the alphabet it was measured on")
            background_alphabet = background_alphabet or args.alphabet
            if not background_alphabet:
                sys.exit("--background needs --alphabet (or --background-alphabet) "
                         "so the composition can be filtered to one alphabet")
        args.background_alphabet = background_alphabet

    if args.background:
        probs = load_background(args.background, args.background_domain,
                                background_alphabet, args.background_genome)
        table = SimulatedNull(probs, reps=args.background_reps)
        custom = True
    else:
        table = NullTable(args.null)
        custom = False
    empirical = (EmpiricalTable(args.empirical, args.domain, args.empirical_stratum)
                 if args.empirical else None)
    out = sys.stdout if args.output == "-" else open(args.output, "w")

    def empirical_for(alphabet, length, entropy):
        """(scored, n) against the real-ORF reference, or (None, None)."""
        if empirical is None:
            return None, None
        key = f"{args.domain}|{args.empirical_stratum}|{alphabet}"
        if key not in empirical.grids:
            return None, None
        grid = empirical.grids[key]
        lengths = grid["lengths"]
        nearest = int(np.abs(lengths - float(length)).argmin())
        return empirical.score(key, length, entropy), int(grid["n"][nearest])

    def key_for(alphabet):
        if custom:
            return "custom"
        key = f"{args.domain}|{alphabet}"
        if key not in table.grids:
            sys.exit(f"no null for {key}; have {table.available()}")
        return key

    if args.fasta:
        if not args.alphabet:
            sys.exit("--fasta requires --alphabet")
        key = key_for(args.alphabet)
        write_header(out, ["id"], empirical)
        for n, (name, sequence) in enumerate(iter_fasta(args.fasta)):
            if args.limit and n >= args.limit:
                break
            entropy = plug_in_entropy(sequence)
            e_scored, e_n = empirical_for(args.alphabet, len(sequence), entropy)
            emit(out, [name], args.alphabet, len(sequence), entropy,
                 table.score(key, len(sequence), entropy), e_scored, e_n)

    elif args.entropy_rows:
        write_header(out, ["domain", "chunk", "genome", "orf_id", "in_genbank"],
                         empirical)
        with open_maybe_gzip(args.entropy_rows) as handle:
            header = handle.readline().rstrip("\n").split("\t")
            index = {name: i for i, name in enumerate(header)}
            for n, line in enumerate(handle):
                if args.limit and n >= args.limit:
                    break
                fields = line.rstrip("\n").split("\t")
                length = int(fields[index["aa_length"]])
                prefix = [fields[index[c]] for c in
                          ("domain", "chunk", "genome", "orf_id", "in_genbank")]
                for alphabet, column in sorted(ROW_ALPHABETS.items()):
                    # A custom background belongs to ONE alphabet. Scoring a
                    # protein entropy against a 3Di background would produce a
                    # number with no meaning, so refuse rather than guess.
                    if custom and alphabet != args.background_alphabet:
                        continue
                    raw = fields[index[column]]
                    if not raw or raw == "NA":
                        continue
                    entropy = float(raw)
                    e_scored, e_n = empirical_for(alphabet, length, entropy)
                    emit(out, prefix, alphabet, length, entropy,
                         table.score(key_for(alphabet), length, entropy),
                         e_scored, e_n)

    else:
        if not args.alphabet or not args.entropy_column:
            sys.exit("--table requires --alphabet and --entropy-column")
        key = key_for(args.alphabet)
        with open_maybe_gzip(args.table) as handle:
            header = handle.readline().rstrip("\n").split("\t")
            index = {name: i for i, name in enumerate(header)}
            id_column = args.id_column or header[0]
            write_header(out, [id_column], empirical)
            for n, line in enumerate(handle):
                if args.limit and n >= args.limit:
                    break
                fields = line.rstrip("\n").split("\t")
                length = int(fields[index[args.length_column]])
                entropy = float(fields[index[args.entropy_column]])
                e_scored, e_n = empirical_for(args.alphabet, length, entropy)
                emit(out, [fields[index[id_column]]], args.alphabet, length,
                     entropy, table.score(key, length, entropy), e_scored, e_n)

    if out is not sys.stdout:
        out.close()


if __name__ == "__main__":
    main()
