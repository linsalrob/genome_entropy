"""Tests for multi-GPU asynchronous encoding."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest


def test_multi_gpu_encoder_initialization():
    """Test MultiGPUEncoder initialization with specified GPUs."""
    from genome_entropy.encode3di.multi_gpu import MultiGPUEncoder
    
    # Mock encoder class
    mock_encoder_class = MagicMock()
    mock_encoder_instance = MagicMock()
    mock_encoder_class.return_value = mock_encoder_instance
    
    # Test with specified GPU IDs
    with patch("genome_entropy.encode3di.multi_gpu.validate_gpu_availability") as mock_validate:
        mock_validate.return_value = [0, 1, 2]
        
        encoder = MultiGPUEncoder(
            model_name="test-model",
            encoder_class=mock_encoder_class,
            gpu_ids=[0, 1, 2],
        )
        
        assert encoder.num_gpus == 3
        assert encoder.is_multi_gpu()
        assert len(encoder.encoders) == 3


def test_multi_gpu_encoder_auto_discovery():
    """Test MultiGPUEncoder with automatic GPU discovery."""
    from genome_entropy.encode3di.multi_gpu import MultiGPUEncoder
    
    # Mock encoder class
    mock_encoder_class = MagicMock()
    
    # Mock GPU discovery
    with patch("genome_entropy.encode3di.multi_gpu.discover_available_gpus") as mock_discover:
        with patch("genome_entropy.encode3di.multi_gpu.validate_gpu_availability") as mock_validate:
            mock_discover.return_value = [0, 1]
            mock_validate.return_value = [0, 1]
            
            encoder = MultiGPUEncoder(
                model_name="test-model",
                encoder_class=mock_encoder_class,
                gpu_ids=None,  # Auto-discover
            )
            
            assert encoder.num_gpus == 2
            assert encoder.is_multi_gpu()


def test_multi_gpu_encoder_fallback_no_gpus():
    """Test MultiGPUEncoder fallback when no GPUs are available."""
    from genome_entropy.encode3di.multi_gpu import MultiGPUEncoder
    
    # Mock encoder class
    mock_encoder_class = MagicMock()
    
    # Mock no GPUs available
    with patch("genome_entropy.encode3di.multi_gpu.discover_available_gpus") as mock_discover:
        mock_discover.return_value = []
        
        encoder = MultiGPUEncoder(
            model_name="test-model",
            encoder_class=mock_encoder_class,
            gpu_ids=None,
        )
        
        # Should fall back to single encoder
        assert encoder.num_gpus == 1
        assert not encoder.is_multi_gpu()


@pytest.mark.asyncio
async def test_encode_batch_async():
    """Test asynchronous batch encoding."""
    from genome_entropy.encode3di.multi_gpu import MultiGPUEncoder
    from genome_entropy.encode3di.types import IndexedSeq
    
    # Mock encoder class and instance
    mock_encoder_class = MagicMock()
    mock_encoder_instance = MagicMock()
    mock_encoder_instance._encode_batch = MagicMock(return_value=["aaa", "bbb"])
    mock_encoder_class.return_value = mock_encoder_instance
    
    with patch("genome_entropy.encode3di.multi_gpu.validate_gpu_availability") as mock_validate:
        mock_validate.return_value = [0]
        
        encoder = MultiGPUEncoder(
            model_name="test-model",
            encoder_class=mock_encoder_class,
            gpu_ids=[0],
        )
        
        # Create test batch
        batch = [
            IndexedSeq(idx=0, seq="AAA"),
            IndexedSeq(idx=1, seq="BBB"),
        ]
        
        # Encode batch
        indices, results = await encoder.encode_batch_async(0, batch)
        
        assert indices == [0, 1]
        assert results == ["aaa", "bbb"]


@pytest.mark.asyncio
async def test_encode_all_batches_async():
    """Test encoding multiple batches across GPUs."""
    from genome_entropy.encode3di.multi_gpu import MultiGPUEncoder
    from genome_entropy.encode3di.types import IndexedSeq
    
    # Mock encoder class and instances
    mock_encoder_class = MagicMock()
    mock_encoder1 = MagicMock()
    mock_encoder2 = MagicMock()
    
    # Mock encoding results
    mock_encoder1._encode_batch = MagicMock(return_value=["aaa", "bbb"])
    mock_encoder2._encode_batch = MagicMock(return_value=["ccc"])
    
    mock_encoder_class.side_effect = [mock_encoder1, mock_encoder2]
    
    with patch("genome_entropy.encode3di.multi_gpu.validate_gpu_availability") as mock_validate:
        mock_validate.return_value = [0, 1]
        
        encoder = MultiGPUEncoder(
            model_name="test-model",
            encoder_class=mock_encoder_class,
            gpu_ids=[0, 1],
        )
        
        # Create test batches
        batches = [
            [IndexedSeq(idx=0, seq="AAA"), IndexedSeq(idx=1, seq="BBB")],  # Batch 0 -> GPU 0
            [IndexedSeq(idx=2, seq="CCC")],  # Batch 1 -> GPU 1
        ]
        
        # Encode all batches
        results = await encoder.encode_all_batches_async(batches, total_sequences=3)
        
        # Check results are in correct order
        assert len(results) == 3
        assert results[0] == "aaa"
        assert results[1] == "bbb"
        assert results[2] == "ccc"


def test_encode_multi_gpu_with_multiple_gpus():
    """Test multi-GPU encoding with multiple GPUs."""
    from genome_entropy.encode3di.multi_gpu import MultiGPUEncoder
    from genome_entropy.encode3di.types import IndexedSeq
    
    # Mock encoder class and instances
    mock_encoder_class = MagicMock()
    mock_encoder1 = MagicMock()
    mock_encoder2 = MagicMock()
    
    mock_encoder1._encode_batch = MagicMock(return_value=["aaa"])
    mock_encoder1._load_model = MagicMock()
    mock_encoder2._encode_batch = MagicMock(return_value=["bbb"])
    mock_encoder2._load_model = MagicMock()
    
    mock_encoder_class.side_effect = [mock_encoder1, mock_encoder2]
    
    # Mock token budget batches function
    def mock_batches(seqs, budget):
        return [
            [IndexedSeq(idx=0, seq=seqs[0])],
            [IndexedSeq(idx=1, seq=seqs[1])],
        ]
    
    with patch("genome_entropy.encode3di.multi_gpu.validate_gpu_availability") as mock_validate:
        mock_validate.return_value = [0, 1]
        
        encoder = MultiGPUEncoder(
            model_name="test-model",
            encoder_class=mock_encoder_class,
            gpu_ids=[0, 1],
        )
        
        # Encode sequences
        results = encoder.encode_multi_gpu(
            aa_sequences=["AAA", "BBB"],
            token_budget_batches_fn=mock_batches,
            encoding_size=10,
        )
        
        assert len(results) == 2
        assert results == ["aaa", "bbb"]


def test_encode_multi_gpu_single_gpu_fallback():
    """Test that single GPU falls back to sequential processing."""
    from genome_entropy.encode3di.multi_gpu import MultiGPUEncoder
    from genome_entropy.encode3di.types import IndexedSeq
    
    # Mock encoder class
    mock_encoder_class = MagicMock()
    mock_encoder = MagicMock()
    mock_encoder._encode_batch = MagicMock(side_effect=[["aaa"], ["bbb"]])
    mock_encoder._load_model = MagicMock()
    mock_encoder_class.return_value = mock_encoder
    
    # Mock token budget batches function
    def mock_batches(seqs, budget):
        return [
            [IndexedSeq(idx=0, seq=seqs[0])],
            [IndexedSeq(idx=1, seq=seqs[1])],
        ]
    
    with patch("genome_entropy.encode3di.multi_gpu.discover_available_gpus") as mock_discover:
        mock_discover.return_value = []
        
        encoder = MultiGPUEncoder(
            model_name="test-model",
            encoder_class=mock_encoder_class,
            gpu_ids=None,
        )
        
        # Should use single GPU
        assert not encoder.is_multi_gpu()
        
        # Encode sequences
        results = encoder.encode_multi_gpu(
            aa_sequences=["AAA", "BBB"],
            token_budget_batches_fn=mock_batches,
            encoding_size=10,
        )
        
        assert len(results) == 2
        assert results == ["aaa", "bbb"]


@pytest.mark.asyncio
async def test_encode_all_batches_error_handling():
    """Test error handling in multi-GPU encoding."""
    from genome_entropy.encode3di.multi_gpu import MultiGPUEncoder
    from genome_entropy.encode3di.types import IndexedSeq
    from genome_entropy.errors import EncodingError
    
    # Mock encoder class
    mock_encoder_class = MagicMock()
    mock_encoder = MagicMock()
    mock_encoder._encode_batch = MagicMock(side_effect=RuntimeError("GPU error"))
    mock_encoder_class.return_value = mock_encoder
    
    with patch("genome_entropy.encode3di.multi_gpu.validate_gpu_availability") as mock_validate:
        mock_validate.return_value = [0]
        
        encoder = MultiGPUEncoder(
            model_name="test-model",
            encoder_class=mock_encoder_class,
            gpu_ids=[0],
        )
        
        batches = [[IndexedSeq(idx=0, seq="AAA")]]
        
        # Should raise EncodingError
        with pytest.raises(EncodingError):
            await encoder.encode_all_batches_async(batches, total_sequences=1)


def test_single_gpu_sequential_encoding():
    """Test sequential encoding for single GPU."""
    from genome_entropy.encode3di.multi_gpu import MultiGPUEncoder
    from genome_entropy.encode3di.types import IndexedSeq
    
    # Mock encoder class
    mock_encoder_class = MagicMock()
    mock_encoder = MagicMock()
    mock_encoder._encode_batch = MagicMock(side_effect=[["aaa"], ["bbb"], ["ccc"]])
    mock_encoder_class.return_value = mock_encoder
    
    with patch("genome_entropy.encode3di.multi_gpu.validate_gpu_availability") as mock_validate:
        mock_validate.return_value = [0]
        
        encoder = MultiGPUEncoder(
            model_name="test-model",
            encoder_class=mock_encoder_class,
            gpu_ids=[0],
        )
        
        batches = [
            [IndexedSeq(idx=0, seq="AAA")],
            [IndexedSeq(idx=1, seq="BBB")],
            [IndexedSeq(idx=2, seq="CCC")],
        ]
        
        results = encoder._encode_single_gpu_sequential(batches, total_sequences=3)
        
        assert len(results) == 3
        assert results == ["aaa", "bbb", "ccc"]
