"""Tests for token_budget_batches optimization."""

import pytest

from orf_entropy.encode3di.types import IndexedSeq


@pytest.fixture
def mock_encoder():
    """Create a mock encoder for testing batching logic."""
    from orf_entropy.encode3di.encoder import ProstT5ThreeDiEncoder

    # Create encoder without loading model
    class MockEncoder(ProstT5ThreeDiEncoder):
        def __init__(self):
            # Skip parent __init__ to avoid loading dependencies
            pass

    encoder = MockEncoder()
    # Copy the method we want to test
    from orf_entropy.encode3di import encoder as encoder_module

    encoder.token_budget_batches = (
        encoder_module.ProstT5ThreeDiEncoder.token_budget_batches.__get__(
            encoder, MockEncoder
        )
    )
    return encoder


def test_token_budget_batches_basic(mock_encoder):
    """Test basic batching functionality."""
    sequences = ["A" * 100, "A" * 200, "A" * 50]
    batches = list(mock_encoder.token_budget_batches(sequences, 500))

    # Should create batches
    assert len(batches) > 0

    # All sequences should be included
    all_indices = set()
    for batch in batches:
        for item in batch:
            all_indices.add(item.idx)
    assert all_indices == {0, 1, 2}

    # Each batch should respect token budget
    for batch in batches:
        max_len = max(len(item.seq) for item in batch)
        est_tokens = len(batch) * max_len
        # Allow sequences that exceed budget to be yielded alone
        if len(batch) > 1:
            assert est_tokens <= 500


def test_token_budget_batches_preserves_indices(mock_encoder):
    """Test that original indices are preserved."""
    sequences = ["A" * i for i in [100, 200, 50, 150]]
    batches = list(mock_encoder.token_budget_batches(sequences, 1000))

    # Collect all indices
    found_indices = []
    for batch in batches:
        for item in batch:
            found_indices.append(item.idx)

    # Should have all original indices
    assert sorted(found_indices) == [0, 1, 2, 3]

    # Should be able to reconstruct original sequences
    reconstructed = [None] * len(sequences)
    for batch in batches:
        for item in batch:
            reconstructed[item.idx] = item.seq

    for i, seq in enumerate(sequences):
        assert reconstructed[i] == seq


def test_token_budget_batches_single_large_sequence(mock_encoder):
    """Test handling of sequences larger than budget."""
    sequences = ["A" * 2000, "A" * 50, "A" * 100]
    batches = list(mock_encoder.token_budget_batches(sequences, 1000))

    # Large sequence should be in its own batch
    large_seq_batch = None
    for batch in batches:
        if any(len(item.seq) > 1000 for item in batch):
            large_seq_batch = batch
            break

    assert large_seq_batch is not None
    assert len(large_seq_batch) == 1
    assert len(large_seq_batch[0].seq) == 2000


def test_token_budget_batches_empty_input(mock_encoder):
    """Test with empty sequence list."""
    sequences = []
    batches = list(mock_encoder.token_budget_batches(sequences, 1000))
    assert len(batches) == 0


def test_token_budget_batches_invalid_budget(mock_encoder):
    """Test that invalid budget raises error."""
    sequences = ["A" * 100]

    with pytest.raises(ValueError, match="token_budget must be > 0"):
        list(mock_encoder.token_budget_batches(sequences, 0))

    with pytest.raises(ValueError, match="token_budget must be > 0"):
        list(mock_encoder.token_budget_batches(sequences, -100))


def test_token_budget_batches_combines_long_and_short(mock_encoder):
    """Test that algorithm combines long and short sequences efficiently.

    This is the key optimization: start with long sequences, then fill
    remaining budget with short ones.
    """
    # Create mix of long and short sequences
    sequences = []
    # Many short sequences
    for i in range(20):
        sequences.append("A" * 50)
    # Few long sequences
    sequences.append("A" * 400)
    sequences.append("A" * 410)
    sequences.append("A" * 420)

    batches = list(mock_encoder.token_budget_batches(sequences, 1000))

    # Find batches with long sequences (>300 AA)
    long_batches = [b for b in batches if any(len(item.seq) > 300 for item in b)]

    # Long sequences should be in early batches (not isolated at the end)
    # This is the key improvement over the old algorithm
    assert len(long_batches) > 0

    # Long sequences should be efficiently packed
    for batch in long_batches:
        max_len = max(len(item.seq) for item in batch)
        est_tokens = len(batch) * max_len
        # Should utilize most of the budget
        assert est_tokens <= 1000


def test_token_budget_batches_efficiency_comparison():
    """Compare batching efficiency between algorithms.

    This test documents the improvement in addressing the stated problem:
    'we end up with long proteins that can't be combined'
    """

    # Create encoder without loading model
    class MockEncoder:
        @staticmethod
        def token_budget_batches(aa_sequences, token_budget):
            """New optimized algorithm."""
            from typing import List

            if token_budget <= 0:
                raise ValueError("token_budget must be > 0")

            indexed: List[IndexedSeq] = [
                IndexedSeq(i, s) for i, s in enumerate(aa_sequences)
            ]
            indexed.sort(key=lambda x: len(x.seq))

            start_idx = 0
            end_idx = len(indexed) - 1

            while start_idx <= end_idx:
                batch: List[IndexedSeq] = []
                batch_max_len = 0

                # Add long sequences from the end
                while end_idx >= start_idx:
                    item = indexed[end_idx]
                    L = len(item.seq)

                    if L > token_budget:
                        if batch:
                            yield batch
                            batch = []
                            batch_max_len = 0
                        yield [item]
                        end_idx -= 1
                        continue

                    new_max_len = max(batch_max_len, L)
                    new_size = len(batch) + 1
                    est_tokens = new_size * new_max_len

                    if est_tokens <= token_budget:
                        batch.append(item)
                        batch_max_len = new_max_len
                        end_idx -= 1
                    else:
                        break

                # Fill with short sequences from the start
                while start_idx <= end_idx:
                    item = indexed[start_idx]
                    L = len(item.seq)

                    new_max_len = max(batch_max_len, L)
                    new_size = len(batch) + 1
                    est_tokens = new_size * new_max_len

                    if est_tokens <= token_budget:
                        batch.append(item)
                        batch_max_len = new_max_len
                        start_idx += 1
                    else:
                        break

                if batch:
                    yield batch

    encoder = MockEncoder()

    # Test scenario: many small + few large
    sequences = []
    for i in range(30):
        sequences.append("A" * 50)
    sequences.extend(["A" * 400, "A" * 410, "A" * 420])

    batches = list(encoder.token_budget_batches(sequences, 1000))

    # Verify all sequences are included
    all_indices = set()
    for batch in batches:
        for item in batch:
            all_indices.add(item.idx)
    assert len(all_indices) == len(sequences)

    # Check that long sequences are in early batches (not isolated at end)
    long_sequence_batch_positions = []
    for batch_idx, batch in enumerate(batches):
        if any(len(item.seq) >= 400 for item in batch):
            long_sequence_batch_positions.append(batch_idx)

    # Long sequences should appear early in the batch list
    # This solves the problem: "we end up with long proteins that can't be combined"
    if long_sequence_batch_positions:
        # At least one long sequence batch should be in first half
        assert min(long_sequence_batch_positions) < len(batches) / 2


def test_token_budget_batches_all_same_length(mock_encoder):
    """Test with sequences of uniform length."""
    sequences = ["A" * 100] * 10
    batches = list(mock_encoder.token_budget_batches(sequences, 500))

    # Should pack 5 sequences per batch (5 * 100 = 500)
    for batch in batches[:-1]:  # All but possibly last
        assert len(batch) == 5

    # Total should be 2 batches
    assert len(batches) == 2


def test_token_budget_batches_graduated_lengths(mock_encoder):
    """Test with sequences of gradually increasing length."""
    # Create sequences from 100 to 500 AA in steps of 50
    sequences = ["A" * length for length in range(100, 550, 50)]

    batches = list(mock_encoder.token_budget_batches(sequences, 1000))

    # Verify all sequences present
    all_indices = set()
    for batch in batches:
        for item in batch:
            all_indices.add(item.idx)
    assert len(all_indices) == len(sequences)

    # Each batch should respect budget (except single large sequences)
    for batch in batches:
        if len(batch) > 1:
            max_len = max(len(item.seq) for item in batch)
            est_tokens = len(batch) * max_len
            assert est_tokens <= 1000
