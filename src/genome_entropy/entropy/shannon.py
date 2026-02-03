"""Shannon entropy calculation for sequences."""

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional, Set

from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class EntropyReport:
    """Report containing entropy values at different representation levels.

    Attributes:
        dna_entropy_global: Entropy of the entire input DNA sequence
        orf_nt_entropy: Dictionary mapping ORF IDs to their nucleotide entropy
        protein_aa_entropy: Dictionary mapping ORF IDs to their amino acid entropy
        three_di_entropy: Dictionary mapping ORF IDs to their 3Di token entropy
        alphabet_sizes: Dictionary with alphabet sizes for each representation
    """

    dna_entropy_global: float
    orf_nt_entropy: Dict[str, float]
    protein_aa_entropy: Dict[str, float]
    three_di_entropy: Dict[str, float]
    alphabet_sizes: Dict[str, int]


def shannon_entropy(
    sequence: str, alphabet: Optional[Set[str]] = None, normalize: bool = False
) -> float:
    """Calculate Shannon entropy of a sequence.

    Shannon entropy: H = -Σ(p_i × log₂(p_i))
    where p_i is the frequency of symbol i.

    Args:
        sequence: String to calculate entropy for
        alphabet: Optional set of symbols in the alphabet for normalization
        normalize: If True, normalize entropy by max possible entropy (log₂|alphabet|)

    Returns:
        Shannon entropy value (bits)
        - Returns 0.0 for empty sequences
        - Returns normalized entropy in [0, 1] if normalize=True

    Examples:
        >>> shannon_entropy("AAAA")
        0.0
        >>> shannon_entropy("ACGT")
        2.0
        >>> shannon_entropy("ACGT", normalize=True, alphabet=set("ACGT"))
        1.0
    """
    if not sequence:
        return 0.0

    # Count symbol frequencies
    counts = Counter(sequence)
    total = len(sequence)

    # Calculate entropy: -Σ(p_i × log₂(p_i))
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


def calculate_sequence_entropy(
    sequence: str, alphabet: Optional[Set[str]] = None, normalize: bool = False
) -> float:
    """Calculate entropy for a biological sequence.

    Convenience wrapper around shannon_entropy that handles common
    preprocessing (e.g., converting to uppercase).

    Args:
        sequence: Biological sequence (DNA, protein, 3Di tokens)
        alphabet: Optional alphabet for normalization
        normalize: Whether to normalize by alphabet size

    Returns:
        Shannon entropy in bits (or normalized to [0, 1])
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
