"""Token size estimation for optimal GPU memory usage in 3Di encoding."""

import logging
import random
from typing import Any, Dict, List, Optional

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from ..config import AA_ALPHABET


def generate_random_protein(length: int, seed: Optional[int] = None) -> str:
    """Generate a random protein sequence of specified length.

    Args:
        length: Length of the protein sequence
        seed: Random seed for reproducibility (optional)

    Returns:
        Random protein sequence using the 20 standard amino acids
    """
    if seed is not None:
        random.seed(seed)

    aa_list = list(AA_ALPHABET)
    return "".join(random.choices(aa_list, k=length))


def generate_combined_proteins(
    target_length: int, base_length: int = 100, seed: Optional[int] = None
) -> List[str]:
    """Generate multiple shorter proteins that combine to target length.

    Args:
        target_length: Total target length across all proteins
        base_length: Approximate length of each individual protein
        seed: Random seed for reproducibility (optional)

    Returns:
        List of protein sequences that total approximately target_length
    """
    if seed is not None:
        random.seed(seed)

    proteins = []
    remaining = target_length

    while remaining > 0:
        # Vary protein length slightly for realism
        variation = int(base_length * 0.2)  # 20% variation
        length = random.randint(base_length - variation, base_length + variation)
        length = min(length, remaining)  # Don't exceed target

        proteins.append(generate_random_protein(length))
        remaining -= length

    return proteins


def estimate_token_size(
    encoder: Any,
    start_length: int = 3000,
    end_length: int = 10000,
    step: int = 1000,
    num_trials: int = 3,
    base_protein_length: int = 100,
) -> Dict[str, Any]:
    """Estimate optimal token size for GPU encoding by testing increasing lengths.

    This function generates random protein sequences of increasing total length
    and attempts to encode them. It catches OutOfMemoryError to find the maximum
    length that can be encoded on the available GPU.

    Args:
        encoder: ProstT5ThreeDiEncoder instance to use for encoding
        start_length: Starting total length to test (default: 3000)
        end_length: Maximum total length to test (default: 10000)
        step: Increment between test lengths (default: 1000)
        num_trials: Number of trials per length for robustness (default: 3)
        base_protein_length: Approximate length of individual proteins (default: 100)

    Returns:
        Dictionary with estimation results:
            - 'max_length': Maximum length successfully encoded
            - 'recommended_token_size': Recommended token budget (90% of max)
            - 'trials_per_length': Dictionary of successful trials per length
            - 'device': Device used for testing

    Raises:
        ValueError: If encoder doesn't have required attributes or torch not available
    """
    if torch is None:
        raise ValueError("PyTorch is required for token size estimation")

    if not hasattr(encoder, "encode") or not hasattr(encoder, "device"):
        raise ValueError("encoder must be a ProstT5ThreeDiEncoder instance")

    logging.info("Starting token size estimation on device: %s", encoder.device)
    logging.info("Testing range: %d to %d (step: %d)", start_length, end_length, step)

    max_successful_length = 0
    trials_per_length: Dict[int, int] = {}

    for total_length in range(start_length, end_length + 1, step):
        logging.info("Testing total length: %d amino acids", total_length)

        successful_trials = 0

        for trial in range(num_trials):
            try:
                # Generate proteins that combine to target length
                proteins = generate_combined_proteins(
                    total_length,
                    base_length=base_protein_length,
                    seed=trial,  # Different seed per trial
                )

                logging.info(
                    "  Trial %d/%d: encoding %d proteins (total %d AA)",
                    trial + 1,
                    num_trials,
                    len(proteins),
                    sum(len(p) for p in proteins),
                )

                # Attempt encoding with token budget
                # Use the encoder's token_budget_batches for realistic batching
                batches = list(encoder.token_budget_batches(proteins, total_length))

                # Try to encode all batches
                for batch_idx, batch in enumerate(batches):
                    batch_seqs = [item.seq for item in batch]
                    _ = encoder._encode_batch(batch_seqs)
                    logging.info(
                        "    Batch %d/%d encoded successfully",
                        batch_idx + 1,
                        len(batches),
                    )

                successful_trials += 1
                logging.info("  Trial %d/%d: SUCCESS", trial + 1, num_trials)

            except torch.cuda.OutOfMemoryError as e:
                logging.warning(
                    "  Trial %d/%d: Out of memory at length %d: %s",
                    trial + 1,
                    num_trials,
                    total_length,
                    str(e),
                )
                # Clear cache and break on OOM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break
            except Exception as e:
                logging.error(
                    "  Trial %d/%d: Unexpected error at length %d: %s",
                    trial + 1,
                    num_trials,
                    total_length,
                    str(e),
                )
                # Don't break on other errors, might be transient
                continue

        trials_per_length[total_length] = successful_trials

        # If no trials succeeded, we've hit the limit
        if successful_trials == 0:
            logging.info(
                "No successful trials at length %d, stopping estimation", total_length
            )
            break

        # Update max successful length if at least one trial succeeded
        if successful_trials > 0:
            max_successful_length = total_length

    # Calculate recommended token size (90% of max for safety margin)
    recommended = (
        int(max_successful_length * 0.9) if max_successful_length > 0 else start_length
    )

    results: Dict[str, Any] = {
        "max_length": max_successful_length,
        "recommended_token_size": recommended,
        "trials_per_length": trials_per_length,
        "device": encoder.device,
    }

    logging.info("=" * 60)
    logging.info("Token Size Estimation Complete")
    logging.info("=" * 60)
    logging.info("Device: %s", results["device"])
    logging.info("Max successful length: %d amino acids", results["max_length"])
    logging.info(
        "Recommended token size: %d amino acids", results["recommended_token_size"]
    )
    logging.info("Trials per length: %s", results["trials_per_length"])
    logging.info("=" * 60)

    return results
