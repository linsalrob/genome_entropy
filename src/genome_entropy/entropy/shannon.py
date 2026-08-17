"""Shannon entropy calculation for sequences."""

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional, Set

from ..logging_config import get_logger

logger = get_logger(__name__)

DNA_ALPHABET_SIZE = 4
PROTEIN_ALPHABET_SIZE = 20
THREE_DI_ALPHABET_SIZE = 20
TWELVE_STATE_ALPHABET_SIZE = 12


def normalise_entropy(entropy: float | None, alphabet_size: int) -> float | None:
    """Normalise a raw Shannon entropy using its theoretical alphabet size.

    This helper is intended for downstream analysis. Normalised values are
    derived from raw entropy and are therefore not stored in standard output.

    Args:
        entropy: Raw Shannon entropy in bits, or ``None`` for missing data.
        alphabet_size: The theoretical number of symbols in the representation.

    Returns:
        Entropy divided by ``log2(alphabet_size)``, or ``None`` when entropy is
        ``None``.

    Raises:
        ValueError: If ``alphabet_size`` is not greater than one.
    """
    if entropy is None:
        return None
    if alphabet_size <= 1:
        raise ValueError("alphabet_size must be greater than 1")
    return entropy / math.log2(alphabet_size)


def normalise_dna_entropy(entropy: float | None) -> float | None:
    """Normalise raw DNA entropy using the theoretical four-symbol alphabet."""
    return normalise_entropy(entropy, DNA_ALPHABET_SIZE)


def normalise_protein_entropy(entropy: float | None) -> float | None:
    """Normalise raw protein entropy using the theoretical 20-symbol alphabet."""
    return normalise_entropy(entropy, PROTEIN_ALPHABET_SIZE)


def normalise_three_di_entropy(entropy: float | None) -> float | None:
    """Normalise raw 3Di entropy using the theoretical 20-symbol alphabet."""
    return normalise_entropy(entropy, THREE_DI_ALPHABET_SIZE)


def normalise_twelve_state_entropy(entropy: float | None) -> float | None:
    """Normalise raw 12-state entropy using its theoretical alphabet."""
    return normalise_entropy(entropy, TWELVE_STATE_ALPHABET_SIZE)


@dataclass
class EntropyReport:
    """Report containing entropy values at different representation levels.

    Attributes:
        dna_entropy_global: Entropy of the entire input DNA sequence
        orf_nt_entropy: Dictionary mapping ORF IDs to their nucleotide entropy
        protein_aa_entropy: Dictionary mapping ORF IDs to their amino acid entropy
        three_di_entropy: Dictionary mapping ORF IDs to their 3Di token entropy
        alphabet_sizes: Dictionary with alphabet sizes for each representation
        twelve_state_entropy: Optional mapping of ORF IDs to 12-state entropy
        three_di_twelve_state_mutual_information: Optional mapping of ORF IDs
            to raw mutual information in bits between aligned 3Di and 12-state
            encodings
    """

    dna_entropy_global: float
    orf_nt_entropy: Dict[str, float]
    protein_aa_entropy: Dict[str, float]
    three_di_entropy: Dict[str, float]
    alphabet_sizes: Dict[str, int]
    twelve_state_entropy: Dict[str, float] | None = None
    three_di_twelve_state_mutual_information: Dict[str, float] | None = None


def shannon_entropy(
    sequence: str, alphabet: Optional[Set[str]] = None, normalize: bool = False
) -> float:
    """Calculate Shannon entropy of a sequence.

    Shannon entropy: :math:`H = -\\sum_i p_i \\log_2(p_i)`, where :math:`p_i` is
    the frequency of symbol :math:`i`.

    Args:
        sequence: String to calculate entropy for
        alphabet: Optional set of symbols in the alphabet for normalization
        normalize: Legacy explicit in-memory normalisation switch. Standard
            pipeline output never enables it; prefer ``normalise_entropy`` for
            downstream analysis.

    Returns:
        Shannon entropy value (bits)
        - Returns 0.0 for empty sequences
        - Returns normalized entropy in [0, 1] if normalize=True

    Examples:
        >>> shannon_entropy("AAAA")
        0.0
        >>> shannon_entropy("ACGT")
        2.0
    """
    if not sequence:
        return 0.0

    # Count symbol frequencies
    counts = Counter(sequence)
    total = len(sequence)

    # Calculate entropy: -sum_i p_i * log2(p_i)
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p_i = count / total
            entropy -= p_i * math.log2(p_i)

    # Normalize if requested
    if normalize and alphabet:
        alphabet_size = len(alphabet)
        if alphabet_size > 1:
            max_entropy = math.log2(alphabet_size)
            return entropy / max_entropy if max_entropy > 0 else 0.0

    return entropy


def mutual_information(sequence_a: str, sequence_b: str) -> float:
    """Calculate empirical mutual information between aligned sequences.

    The value is reported in bits from the observed joint distribution. Both
    sequences must describe the same residue positions.

    Args:
        sequence_a: First categorical sequence.
        sequence_b: Second categorical sequence aligned to ``sequence_a``.

    Returns:
        Raw mutual information in bits. Empty aligned sequences return ``0.0``.

    Raises:
        ValueError: If the sequences have unequal lengths.
    """
    if len(sequence_a) != len(sequence_b):
        raise ValueError(
            "Mutual information requires aligned sequences of equal length"
        )
    if not sequence_a:
        return 0.0

    total = len(sequence_a)
    counts_a = Counter(sequence_a)
    counts_b = Counter(sequence_b)
    joint_counts = Counter(zip(sequence_a, sequence_b))

    information = 0.0
    for (state_a, state_b), joint_count in joint_counts.items():
        probability_joint = joint_count / total
        probability_a = counts_a[state_a] / total
        probability_b = counts_b[state_b] / total
        information += probability_joint * math.log2(
            probability_joint / (probability_a * probability_b)
        )

    # Mutual information is non-negative; preserve meaningful errors while
    # removing only possible floating-point noise.
    if -1e-12 < information < 0:
        return 0.0
    return information


def calculate_sequence_entropy(
    sequence: str, alphabet: Optional[Set[str]] = None, normalize: bool = False
) -> float:
    """Calculate entropy for a biological sequence.

    Convenience wrapper around shannon_entropy that handles common
    preprocessing (e.g., converting to uppercase).

    Args:
        sequence: Biological sequence (DNA, protein, 3Di tokens)
        alphabet: Optional alphabet for the legacy normalisation switch
        normalize: Legacy explicit normalisation switch; standard output is raw

    Returns:
        Shannon entropy in bits, or a legacy explicitly normalised value
    """
    # Convert to uppercase for consistency
    sequence = sequence.upper()
    return shannon_entropy(sequence, alphabet=alphabet, normalize=normalize)


def calculate_entropies_for_sequences(
    sequences: Dict[str, str],
    alphabet: Optional[Set[str]] = None,
    normalize: bool = False,
) -> Dict[str, float]:
    """Calculate entropy for multiple sequences.

    Args:
        sequences: Dictionary mapping IDs to sequences
        alphabet: Optional alphabet for normalization
        normalize: Whether to normalize by alphabet size

    Returns:
        Dictionary mapping IDs to entropy values
    """
    logger.debug("Calculating entropy for %d sequence(s)", len(sequences))
    entropies = {
        seq_id: calculate_sequence_entropy(seq, alphabet=alphabet, normalize=normalize)
        for seq_id, seq in sequences.items()
    }
    logger.debug("Calculated entropy for %d sequence(s)", len(entropies))
    return entropies
