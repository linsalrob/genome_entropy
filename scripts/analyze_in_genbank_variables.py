#!/usr/bin/env python3
"""Rank entropy-variable pairs for their association with ``in_genbank``.

The input is streamed, while a deterministic class-balanced reservoir sample is
kept in memory.  Each pair is scored by empirical mutual information (bits)
between the binary label and its 20-by-20 quantile-binned joint values.
"""

import argparse
import bisect
import csv
import math
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure


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


def write_results_record(
    output_path: Path, counts: Counter[bool], skipped: int,
    rankings: list[tuple[float, int, int]], reservoir_size: int, bins: int,
) -> None:
    """Write a concise, linkable record of the analysis results."""
    information, index_a, index_b = rankings[0]
    output_path.write_text(
        "# In-GenBank entropy and structural-state analysis\n\n"
        "## Result\n\n"
        f"The most informative variable pair was `{VARIABLES[index_a]}` and "
        f"`{VARIABLES[index_b]}` (**{information:.8f} bits**).\n\n"
        "## Interpretation\n\n"
        "This pair ranked first because its *joint* distribution across the "
        "quantile bins differs more strongly between in-GenBank and "
        "non-GenBank ORFs than the joint distributions of the other tested "
        "pairs. In practical terms, knowing both "
        f"`{VARIABLES[index_a]}` and `{VARIABLES[index_b]}` reduces "
        "uncertainty about the `in_genbank` label more than knowing any other "
        "tested pair.\n\n"
        "This is an association, not a causal explanation: the result does "
        "not show that either entropy causes inclusion in GenBank, nor does it "
        "identify a single separating threshold. The scatter plots should be "
        "used to inspect the regions where the two classes differ.\n\n"
        "## Outputs\n\n"
        "- Most informative pair scatter: [SVG](most_informative_pair_scatter.svg) | [PNG](most_informative_pair_scatter.png)\n"
        "- Four-panel entropy and mutual-information scatter plot: [SVG](in_genbank_entropy_mi_panels.svg) | [PNG](in_genbank_entropy_mi_panels.png)\n"
        "- [All pairwise rankings](pairwise_in_genbank_mutual_information.tsv)\n\n"
        "## Method\n\n"
        f"- Valid ORFs: {counts[True]:,} in GenBank; {counts[False]:,} not in GenBank.\n"
        f"- Skipped rows: {skipped:,}.\n"
        f"- Pair ranking: empirical mutual information with `in_genbank`, using {bins} by {bins} quantile bins and a deterministic class-balanced reservoir of up to {reservoir_size:,} ORFs per class.\n"
        "- Figures were made with Seaborn/Matplotlib. Each scatter panel displays a deterministic balanced random sample of up to 10,000 ORFs per class.\n",
        encoding="utf-8",
    )


def save_figure(figure: Figure, output_stem: Path) -> None:
    """Save a Seaborn/Matplotlib figure as both PNG and SVG."""
    figure.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def sampled_by_label(
    observations: list[tuple[tuple[float, ...], bool]], limit: int,
    random_source: random.Random,
) -> dict[bool, list[tuple[float, ...]]]:
    """Return an equally sized random sample of each label's observations."""
    values_by_label = {
        label: [values for values, observed_label in observations if observed_label == label]
        for label in LABELS
    }
    return {
        label: random_source.sample(values, min(limit, len(values)))
        for label, values in values_by_label.items()
    }


def seaborn_scatter(
    axis: Axes, samples: dict[bool, list[tuple[float, ...]]], index_a: int,
    index_b: int,
) -> None:
    """Draw a class-coloured scatter plot with Seaborn."""
    palette = {False: "#e66101", True: "#1b9e77"}
    labels = {False: "not in GenBank", True: "in GenBank"}
    for label in LABELS:
        values = samples[label]
        sns.scatterplot(
            x=[value[index_a] for value in values],
            y=[value[index_b] for value in values],
            ax=axis,
            color=palette[label],
            label=labels[label],
            alpha=0.18,
            s=9,
            linewidth=0,
            rasterized=False,
        )
    axis.set_xlabel(VARIABLES[index_a])
    axis.set_ylabel(VARIABLES[index_b])


def write_seaborn_figures(
    observations: list[tuple[tuple[float, ...], bool]], output_directory: Path,
    best_index_a: int, best_index_b: int, random_source: random.Random,
) -> None:
    """Write the single best-pair and requested four-panel Seaborn figures."""
    sns.set_theme(style="whitegrid", context="notebook")

    best_figure, best_axis = plt.subplots(figsize=(9, 7))
    seaborn_scatter(
        best_axis,
        sampled_by_label(observations, 20_000, random_source),
        best_index_a,
        best_index_b,
    )
    best_axis.set_title("Most informative entropy-variable pair for in_genbank")
    best_axis.legend(title="GenBank status", markerscale=2)
    save_figure(best_figure, output_directory / "most_informative_pair_scatter")

    panels = (
        (1, 4, "Protein entropy vs structural mutual information"),
        (2, 3, "3Di entropy vs 12-state entropy"),
        (2, 4, "3Di entropy vs structural mutual information"),
        (3, 4, "12-state entropy vs structural mutual information"),
    )
    panel_figure, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    for axis, (index_a, index_b, title) in zip(axes.flat, panels):
        seaborn_scatter(
            axis,
            sampled_by_label(observations, 10_000, random_source),
            index_a,
            index_b,
        )
        axis.set_title(title)
        axis.legend(title="GenBank status", markerscale=2)
    panel_figure.suptitle(
        "Entropy and structural mutual information by GenBank status", fontsize=16
    )
    save_figure(panel_figure, output_directory / "in_genbank_entropy_mi_panels")


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
    if not all(reservoirs.values()):
        raise ValueError("Both in_genbank classes require at least one valid row")

    observations = [
        (values, label) for label in LABELS for values in reservoirs[label]
    ]

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
    write_seaborn_figures(
        observations,
        args.output_directory,
        index_a,
        index_b,
        random.Random(args.seed + 2),
    )
    write_results_record(
        args.output_directory / "in_genbank_analysis_results.md",
        counts,
        skipped,
        rankings,
        args.reservoir_size,
        args.bins,
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
