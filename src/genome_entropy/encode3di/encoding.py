"""Core encoding functions for amino acid to 3Di conversion."""

import math
import re
import time
from typing import Any, Callable, Iterator, List, Tuple

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from ..errors import EncodingError
from ..logging_config import get_logger

logger = get_logger(__name__)


def preprocess_sequences(aa_sequences: List[str]) -> List[str]:
    """Preprocess amino acid sequences for ProstT5 encoding.

    Args:
        aa_sequences: List of raw amino acid sequences

    Returns:
        List of preprocessed sequences ready for ProstT5 model
    """
    # Replace all rare/ambiguous amino acids by X (3Di sequences does not
    # have those) and introduce white-space between all sequences (AAs and 3Di)
    processed = [
        " ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in aa_sequences
    ]

    # Add pre-fixes accordingly.
    # For the translation from AAs to 3Di, you need to prepend "<AA2fold>"
    # and we convert to uppercase to fix that they are proteins not 3Dis
    processed = ["<AA2fold>" + " " + s.upper() for s in processed]

    return processed


def format_seconds(seconds: float) -> str:
    """Format seconds as H:MM:SS (or M:SS for < 1 hour)."""
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def get_memory_info() -> Tuple[float, float]:
    """Get current CUDA memory allocation and reservation in GB.

    Returns:
        Tuple of (allocated_gb, reserved_gb). Returns (0, 0) if CUDA not available.
    """
    if torch is None or not torch.cuda.is_available():
        return 0.0, 0.0

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    return allocated, reserved


def process_batches(
    batches_iter: Iterator[Any],
    encode_batch_fn: Callable[[List[str]], List[str]],
    total_sequences: int,
    total_batches: int,
) -> List[str]:
    """Process batches of sequences and return results in original order.

    Args:
        batches_iter: Iterator yielding batches of IndexedSeq objects
        encode_batch_fn: Function to encode a batch of sequences
        total_sequences: Total number of sequences being processed
        total_batches: Total number of batches to process

    Returns:
        List of encoded 3Di sequences in original input order

    Raises:
        EncodingError: If encoding fails
        RuntimeError: If some sequences were not encoded
    """
    three_di_sequences: List[str] = [None] * total_sequences  # type: ignore[list-item]

    processed_sequences = 0
    t0 = time.perf_counter()
    avg_batch_sec: float | None = None

    try:
        for idx, batch in enumerate(batches_iter, start=1):
            logger.info(
                "Preparing batch %d with %d sequences, total len: %d",
                idx,
                len(batch),
                sum(len(x.seq) for x in batch),
            )

            batch_seqs = [x.seq for x in batch]
            batch_idxs = [x.idx for x in batch]
            processed_sequences += len(batch_seqs)

            # Calculate ETA
            remaining = total_batches - (idx - 1)
            eta_str = (
                "--"
                if avg_batch_sec is None
                else format_seconds(avg_batch_sec * remaining)
            )

            # Get memory info
            allocated, reserved = get_memory_info()

            logger.info(
                "3Di encoding batch %d of %d batches (sequences %d of %d). "
                "Estimated %s remaining. Cuda memory allocated: %.1f GB reserved: %.1f GB",
                idx,
                total_batches,
                processed_sequences,
                total_sequences,
                eta_str,
                allocated,
                reserved,
            )

            batch_start = time.perf_counter()
            batch_results = encode_batch_fn(batch_seqs)

            if len(batch_results) != len(batch_seqs):
                raise ValueError(
                    f"encoder returned {len(batch_results)} results "
                    f"for a batch of {len(batch_seqs)} sequences"
                )

            # Reorder results to match original input order
            for bi, br in zip(batch_idxs, batch_results):
                three_di_sequences[bi] = br

            # Update timing
            batch_elapsed = time.perf_counter() - batch_start
            if idx == 1:
                avg_batch_sec = batch_elapsed
            else:
                elapsed_total = time.perf_counter() - t0
                avg_batch_sec = elapsed_total / idx

    except Exception as e:
        raise EncodingError(f"Failed to encode sequences: {e}") from e

    # Check all sequences encoded
    missing = [i for i, v in enumerate(three_di_sequences) if v is None]
    if missing:
        raise RuntimeError(
            f"Missing encodings for {len(missing)} sequences "
            f"(e.g., indices {missing[:10]})"
        )

    return three_di_sequences


def encode(
    aa_sequences: List[str],
    encode_batch_fn: Callable[[List[str]], List[str]],
    token_budget_batches_fn: Callable[[List[str], int], Iterator[Any]],
    encoding_size: int,
) -> List[str]:
    """Encode amino acid sequences to 3Di tokens.

    This is a standalone encoding function that orchestrates the encoding pipeline.

    Args:
        aa_sequences: List of amino acid sequences (uppercase, standard 20 AAs)
        encode_batch_fn: Function that encodes a batch of preprocessed sequences
        token_budget_batches_fn: Function that batches sequences under token budget
        encoding_size: Maximum size (approx. amino acids) to encode per batch

    Returns:
        List of 3Di token sequences (one per input sequence)

    Raises:
        EncodingError: If encoding fails
    """
    # Preprocess sequences
    processed_seqs = preprocess_sequences(aa_sequences)

    # Calculate batch info
    total_sequences = len(processed_seqs)
    total_batches = math.ceil(sum(map(len, processed_seqs)) / encoding_size)

    # Create batches iterator
    batches = token_budget_batches_fn(processed_seqs, encoding_size)

    # Process all batches
    return process_batches(
        batches,
        encode_batch_fn,
        total_sequences,
        total_batches,
    )
