"""Tests for GPU discovery and management utilities."""

import os
from unittest.mock import patch

import pytest


def test_discover_gpus_from_slurm_job_gpus(monkeypatch):
    """Test GPU discovery from SLURM_JOB_GPUS environment variable."""
    from genome_entropy.encode3di.gpu_utils import discover_available_gpus
    
    # Set SLURM_JOB_GPUS
    monkeypatch.setenv("SLURM_JOB_GPUS", "0,1,2")
    monkeypatch.delenv("SLURM_GPUS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    
    gpu_ids = discover_available_gpus()
    assert gpu_ids == [0, 1, 2]


def test_discover_gpus_from_slurm_gpus(monkeypatch):
    """Test GPU discovery from SLURM_GPUS environment variable."""
    from genome_entropy.encode3di.gpu_utils import discover_available_gpus
    
    # Set SLURM_GPUS (when SLURM_JOB_GPUS is not set)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.setenv("SLURM_GPUS", "2,3,4")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    
    gpu_ids = discover_available_gpus()
    assert gpu_ids == [2, 3, 4]


def test_discover_gpus_from_cuda_visible_devices(monkeypatch):
    """Test GPU discovery from CUDA_VISIBLE_DEVICES."""
    from genome_entropy.encode3di.gpu_utils import discover_available_gpus
    
    # Set CUDA_VISIBLE_DEVICES (when SLURM vars are not set)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_GPUS", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,6")
    
    gpu_ids = discover_available_gpus()
    # CUDA_VISIBLE_DEVICES remaps to local indices
    assert gpu_ids == [0, 1]


def test_discover_gpus_priority_slurm_job_over_slurm(monkeypatch):
    """Test that SLURM_JOB_GPUS has priority over SLURM_GPUS."""
    from genome_entropy.encode3di.gpu_utils import discover_available_gpus
    
    # Set both, SLURM_JOB_GPUS should win
    monkeypatch.setenv("SLURM_JOB_GPUS", "0,1")
    monkeypatch.setenv("SLURM_GPUS", "2,3,4")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    
    gpu_ids = discover_available_gpus()
    assert gpu_ids == [0, 1]  # From SLURM_JOB_GPUS


def test_discover_gpus_priority_slurm_over_cuda(monkeypatch):
    """Test that SLURM vars have priority over CUDA_VISIBLE_DEVICES."""
    from genome_entropy.encode3di.gpu_utils import discover_available_gpus
    
    # Set SLURM_GPUS and CUDA_VISIBLE_DEVICES, SLURM should win
    monkeypatch.setenv("SLURM_GPUS", "1,2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,6,7")
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    
    gpu_ids = discover_available_gpus()
    assert gpu_ids == [1, 2]  # From SLURM_GPUS


def test_discover_gpus_fallback_to_torch_cuda(monkeypatch):
    """Test fallback to torch.cuda when no env vars are set."""
    from genome_entropy.encode3di.gpu_utils import discover_available_gpus
    
    # Clear all env vars
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_GPUS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    
    # Mock torch.cuda
    with patch("genome_entropy.encode3di.gpu_utils.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 4
        
        gpu_ids = discover_available_gpus()
        assert gpu_ids == [0, 1, 2, 3]


def test_discover_gpus_no_gpus_available(monkeypatch):
    """Test when no GPUs are available."""
    from genome_entropy.encode3di.gpu_utils import discover_available_gpus
    
    # Clear all env vars
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_GPUS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    
    # Mock torch.cuda as unavailable
    with patch("genome_entropy.encode3di.gpu_utils.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        
        gpu_ids = discover_available_gpus()
        assert gpu_ids == []


def test_discover_gpus_invalid_slurm_format(monkeypatch, caplog):
    """Test handling of invalid SLURM_JOB_GPUS format."""
    from genome_entropy.encode3di.gpu_utils import discover_available_gpus
    
    # Set invalid format
    monkeypatch.setenv("SLURM_JOB_GPUS", "invalid,format,x")
    monkeypatch.delenv("SLURM_GPUS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    
    # Mock torch.cuda as unavailable (so it doesn't fallback)
    with patch("genome_entropy.encode3di.gpu_utils.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        
        gpu_ids = discover_available_gpus()
        assert gpu_ids == []  # Falls back to empty
        assert "Failed to parse SLURM_JOB_GPUS" in caplog.text


def test_discover_gpus_empty_string(monkeypatch):
    """Test handling of empty GPU string."""
    from genome_entropy.encode3di.gpu_utils import discover_available_gpus
    
    # Set empty string
    monkeypatch.setenv("SLURM_JOB_GPUS", "")
    monkeypatch.delenv("SLURM_GPUS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    
    # Mock torch.cuda as unavailable
    with patch("genome_entropy.encode3di.gpu_utils.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        
        gpu_ids = discover_available_gpus()
        assert gpu_ids == []


def test_parse_gpu_list():
    """Test GPU list parsing."""
    from genome_entropy.encode3di.gpu_utils import _parse_gpu_list
    
    assert _parse_gpu_list("0,1,2") == [0, 1, 2]
    assert _parse_gpu_list("5") == [5]
    assert _parse_gpu_list("0, 1, 2") == [0, 1, 2]  # With spaces
    assert _parse_gpu_list("") == []
    assert _parse_gpu_list("  ") == []


def test_parse_gpu_list_invalid():
    """Test GPU list parsing with invalid input."""
    from genome_entropy.encode3di.gpu_utils import _parse_gpu_list
    
    with pytest.raises(ValueError):
        _parse_gpu_list("a,b,c")
    
    with pytest.raises(ValueError):
        _parse_gpu_list("0,1,invalid")


def test_select_device_for_gpu():
    """Test device string generation."""
    from genome_entropy.encode3di.gpu_utils import select_device_for_gpu
    
    assert select_device_for_gpu(0) == "cuda:0"
    assert select_device_for_gpu(1) == "cuda:1"
    assert select_device_for_gpu(5) == "cuda:5"


def test_validate_gpu_availability():
    """Test GPU availability validation."""
    from genome_entropy.encode3di.gpu_utils import validate_gpu_availability
    
    # Mock torch.cuda with 4 devices
    with patch("genome_entropy.encode3di.gpu_utils.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 4
        
        # Valid IDs
        assert validate_gpu_availability([0, 1, 2]) == [0, 1, 2]
        assert validate_gpu_availability([0]) == [0]
        assert validate_gpu_availability([3]) == [3]
        
        # Invalid IDs (out of range)
        assert validate_gpu_availability([4, 5]) == []
        assert validate_gpu_availability([0, 1, 10]) == [0, 1]
        
        # Empty input
        assert validate_gpu_availability([]) == []


def test_validate_gpu_availability_no_cuda():
    """Test GPU validation when CUDA is not available."""
    from genome_entropy.encode3di.gpu_utils import validate_gpu_availability
    
    # Mock torch.cuda as unavailable
    with patch("genome_entropy.encode3di.gpu_utils.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        
        assert validate_gpu_availability([0, 1, 2]) == []


def test_validate_gpu_availability_no_torch(monkeypatch):
    """Test GPU validation when torch is not available."""
    from genome_entropy.encode3di import gpu_utils
    from genome_entropy.encode3di.gpu_utils import validate_gpu_availability
    
    # Mock torch as None
    monkeypatch.setattr(gpu_utils, "torch", None)
    
    assert validate_gpu_availability([0, 1, 2]) == []
