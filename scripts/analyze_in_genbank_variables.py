#!/usr/bin/env python3
"""Rank entropy-variable pairs for their association with ``in_genbank``.

The input is streamed, while a deterministic class-balanced reservoir sample is
kept in memory.  Each pair is scored by empirical mutual information (bits)
between the binary label and its 20-by-20 quantile-binned joint values.
"""

import argparse
import bisect
import csv
import html
import math
import random
from collections import Counter
from itertools import combinations
from pathlib import Path


VARIABLES = (
    "dna_entropy",
    "protein_entropy",
    "three_di_entropy",
    "twelve_state_entropy",
    "three_di_twelve_state_mutual_information",
)
LABELS = (False, True)


def update_reservoir(
    reservoir: list[tuple[float, ...]],
    value: tuple[float, ...],
    seen: int,
    limit: int,
    random_source: random.Random,
) -> None:
    """Add ``value`` using standard reservoir sampling."""
    if len(reservoir) < limit:
        reservoir.append(value)
        return
    replacement = random_source.randrange(seen)
    if replacement < limit:
        reservoir[replacement] = value


def read_balanced_reservoir(
    input_path: Path, limit: int, seed: int
) -> tuple[dict[bool, list[tuple[float, ...]]], Counter[bool], int]:
    """Read valid rows into equal-size independent reservoirs per class."""
    random_source = random.Random(seed)
    reservoirs: dict[bool, list[tuple[float, ...]]] = {False: [], True: []}
    counts: Counter[bool] = Counter()
    skipped = 0

    with input_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            label_text = row["in_genbank"].strip().lower()
            if label_text not in {"true", "false"}:
                skipped += 1
                continue
            try:
                values = tuple(float(row[variable]) for variable in VARIABLES)
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue
            if not all(math.isfinite(value) for value in values):
                skipped += 1
                continue

            label = label_text == "true"
            counts[label] += 1
            update_reservoir(
                reservoirs[label], values, counts[label], limit, random_source
            )
    return reservoirs, counts, skipped


def quantile_edges(values: list[float], bins: int) -> list[float]:
    """Return interior empirical quantile boundaries."""
    ordered = sorted(values)
    return [ordered[(len(ordered) * index) // bins] for index in range(1, bins)]


def pair_mutual_information(
    observations: list[tuple[tuple[float, ...], bool]], index_a: int, index_b: int,
    bins: int,
) -> float:
    """Estimate I((A, B); in_genbank) after quantile binning A and B."""
    edges_a = quantile_edges([values[index_a] for values, _ in observations], bins)
    edges_b = quantile_edges([values[index_b] for values, _ in observations], bins)
    joint_counts: Counter[tuple[int, int, bool]] = Counter()
    pair_counts: Counter[tuple[int, int]] = Counter()
    label_counts: Counter[bool] = Counter()

    for values, label in observations:
        pair = (
            bisect.bisect_right(edges_a, values[index_a]),
            bisect.bisect_right(edges_b, values[index_b]),
        )
        joint_counts[(*pair, label)] += 1
        pair_counts[pair] += 1
        label_counts[label] += 1

    total = len(observations)
    return sum(
        (count / total)
        * math.log2(
            (count * total)
            / (pair_counts[(bin_a, bin_b)] * label_counts[label])
        )
        for (bin_a, bin_b, label), count in joint_counts.items()
    )


def write_plot(
    observations: list[tuple[tuple[float, ...], bool]], index_a: int, index_b: int,
    output_path: Path, random_source: random.Random,
) -> None:
    """Write a class-balanced SVG scatter plot for the selected pair."""
    width, height, margin = 1100, 850, 100
    by_label = {label: [values for values, observed_label in observations if observed_label == label]
                for label in LABELS}
    sampled = []
    for label, values in by_label.items():
        sampled.extend((value, label) for value in random_source.sample(
            values, min(20_000, len(values))
        ))

    values_a = [values[index_a] for values, _ in sampled]
    values_b = [values[index_b] for values, _ in sampled]
    min_a, max_a = min(values_a), max(values_a)
    min_b, max_b = min(values_b), max(values_b)
    range_a = max(max_a - min_a, 1e-12)
    range_b = max(max_b - min_b, 1e-12)

    def coordinate_a(value: float) -> float:
        return margin + (value - min_a) / range_a * (width - 2 * margin)

    def coordinate_b(value: float) -> float:
        return height - margin - (value - min_b) / range_b * (height - 2 * margin)

    colours = {False: "#e66101", True: "#1b9e77"}
    points = "\n".join(
        f'<circle cx="{coordinate_a(values[index_a]):.2f}" '
        f'cy="{coordinate_b(values[index_b]):.2f}" r="1.4" '
        f'fill="{colours[label]}" fill-opacity="0.18"/>'
        for values, label in sampled
    )
    label_a = html.escape(VARIABLES[index_a])
    label_b = html.escape(VARIABLES[index_b])
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="42" text-anchor="middle" font-family="sans-serif" font-size="24">Most informative entropy-variable pair for in_genbank</text>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="black"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="black"/>
<text x="{width / 2}" y="{height - 35}" text-anchor="middle" font-family="sans-serif" font-size="18">{label_a}</text>
<text x="30" y="{height / 2}" transform="rotate(-90 30 {height / 2})" text-anchor="middle" font-family="sans-serif" font-size="18">{label_b}</text>
<text x="{margin}" y="{height - margin + 25}" font-family="sans-serif" font-size="14">{min_a:.3f}</text>
<text x="{width - margin}" y="{height - margin + 25}" text-anchor="end" font-family="sans-serif" font-size="14">{max_a:.3f}</text>
<text x="{margin - 10}" y="{height - margin}" text-anchor="end" font-family="sans-serif" font-size="14">{min_b:.3f}</text>
<text x="{margin - 10}" y="{margin + 5}" text-anchor="end" font-family="sans-serif" font-size="14">{max_b:.3f}</text>
<circle cx="{width - 350}" cy="75" r="6" fill="{colours[True]}"/><text x="{width - 335}" y="80" font-family="sans-serif" font-size="16">in GenBank</text>
<circle cx="{width - 190}" cy="75" r="6" fill="{colours[False]}"/><text x="{width - 175}" y="80" font-family="sans-serif" font-size="16">not in GenBank</text>
{points}
</svg>'''
    output_path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="TSV created by the MI workflow")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--reservoir-size", type=int, default=200_000)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()

    args.output_directory.mkdir(parents=True, exist_ok=True)
    reservoirs, counts, skipped = read_balanced_reservoir(
        args.input, args.reservoir_size, args.seed
    )
    observations = [
        (values, label) for label in LABELS for values in reservoirs[label]
    ]
    if not all(reservoirs.values()):
        raise ValueError("Both in_genbank classes require at least one valid row")

    rankings = [
        (pair_mutual_information(observations, index_a, index_b, args.bins), index_a, index_b)
        for index_a, index_b in combinations(range(len(VARIABLES)), 2)
    ]
    rankings.sort(reverse=True)
    ranking_path = args.output_directory / "pairwise_in_genbank_mutual_information.tsv"
    with ranking_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("variable_a", "variable_b", "mutual_information_bits"))
        writer.writerows(
            (VARIABLES[index_a], VARIABLES[index_b], f"{information:.8f}")
            for information, index_a, index_b in rankings
        )

    _, index_a, index_b = rankings[0]
    write_plot(
        observations, index_a, index_b,
        args.output_directory / "most_informative_pair_scatter.svg",
        random.Random(args.seed + 1),
    )
    print(f"Valid rows: in_genbank={counts[True]}, not_in_genbank={counts[False]}")
    print(f"Skipped rows: {skipped}")
    print(
        "Best pair: "
        f"{VARIABLES[index_a]} and {VARIABLES[index_b]} "
        f"({rankings[0][0]:.8f} bits)"
    )


if __name__ == "__main__":
    main()
