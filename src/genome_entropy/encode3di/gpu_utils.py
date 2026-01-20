"""GPU discovery and management utilities for multi-GPU encoding."""

import os
from typing import List, Optional

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from ..logging_config import get_logger

logger = get_logger(__name__)


def discover_available_gpus() -> List[int]:
    """Discover available GPU devices from environment variables and CUDA.
    
    Checks multiple sources in order of priority:
    1. SLURM_JOB_GPUS - SLURM allocated GPU IDs
    2. SLURM_GPUS - Alternative SLURM GPU specification
    3. CUDA_VISIBLE_DEVICES - User-specified visible devices
    4. torch.cuda - Query CUDA directly if available
    
    Returns:
        List of GPU device IDs available for use. Empty list if no GPUs found.
        
    Examples:
        >>> # With SLURM_JOB_GPUS="0,1,2"
        >>> discover_available_gpus()
        [0, 1, 2]
        
        >>> # With CUDA_VISIBLE_DEVICES="2,3"
        >>> discover_available_gpus()
        [0, 1]  # Remapped to local indices
    """
    # Log environment for debugging
    logger.debug("SLURM_GPUS: %s", os.environ.get("SLURM_GPUS"))
    logger.debug("SLURM_JOB_GPUS: %s", os.environ.get("SLURM_JOB_GPUS"))
    logger.debug("CUDA_VISIBLE_DEVICES: %s", os.environ.get("CUDA_VISIBLE_DEVICES"))
    
    # Check SLURM_JOB_GPUS first (most specific)
    slurm_job_gpus = os.environ.get("SLURM_JOB_GPUS")
    if slurm_job_gpus:
        try:
            gpu_ids = _parse_gpu_list(slurm_job_gpus)
            logger.info("Discovered %d GPU(s) from SLURM_JOB_GPUS: %s", len(gpu_ids), gpu_ids)
            return gpu_ids
        except ValueError as e:
            logger.warning("Failed to parse SLURM_JOB_GPUS='%s': %s", slurm_job_gpus, e)
    
    # Check SLURM_GPUS (alternative SLURM format)
    slurm_gpus = os.environ.get("SLURM_GPUS")
    if slurm_gpus:
        try:
            gpu_ids = _parse_gpu_list(slurm_gpus)
            logger.info("Discovered %d GPU(s) from SLURM_GPUS: %s", len(gpu_ids), gpu_ids)
            return gpu_ids
        except ValueError as e:
            logger.warning("Failed to parse SLURM_GPUS='%s': %s", slurm_gpus, e)
    
    # Check CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        try:
            # CUDA_VISIBLE_DEVICES remaps GPUs to local indices 0, 1, 2, ...
            gpu_ids = _parse_gpu_list(cuda_visible)
            # Remap to local indices
            local_ids = list(range(len(gpu_ids)))
            logger.info(
                "Discovered %d GPU(s) from CUDA_VISIBLE_DEVICES (remapped to local indices): %s",
                len(local_ids), local_ids
            )
            return local_ids
        except ValueError as e:
            logger.warning("Failed to parse CUDA_VISIBLE_DEVICES='%s': %s", cuda_visible, e)
    
    # Fall back to querying CUDA directly
    if torch is not None and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_ids = list(range(gpu_count))
        logger.info("Discovered %d GPU(s) via torch.cuda: %s", len(gpu_ids), gpu_ids)
        return gpu_ids
    
    # No GPUs found
    logger.info("No GPUs discovered")
    return []


def _parse_gpu_list(gpu_string: str) -> List[int]:
    """Parse a comma-separated list of GPU IDs.
    
    Args:
        gpu_string: Comma-separated GPU IDs (e.g., "0,1,2" or "2,3")
        
    Returns:
        List of integer GPU IDs
        
    Raises:
        ValueError: If parsing fails
    """
    gpu_string = gpu_string.strip()
    if not gpu_string:
        return []
    
    try:
        gpu_ids = [int(x.strip()) for x in gpu_string.split(",")]
        return gpu_ids
    except ValueError as e:
        raise ValueError(f"Invalid GPU list format: {gpu_string}") from e


def select_device_for_gpu(gpu_id: int) -> str:
    """Get the device string for a specific GPU.
    
    Args:
        gpu_id: GPU device ID
        
    Returns:
        Device string (e.g., "cuda:0", "cuda:1")
    """
    return f"cuda:{gpu_id}"


def validate_gpu_availability(gpu_ids: List[int]) -> List[int]:
    """Validate that specified GPUs are actually available.
    
    Args:
        gpu_ids: List of GPU IDs to validate
        
    Returns:
        List of valid GPU IDs (subset of input)
    """
    if not gpu_ids:
        return []
    
    if torch is None or not torch.cuda.is_available():
        logger.warning("CUDA not available, cannot validate GPU IDs: %s", gpu_ids)
        return []
    
    device_count = torch.cuda.device_count()
    valid_ids = [gid for gid in gpu_ids if 0 <= gid < device_count]
    
    invalid_ids = [gid for gid in gpu_ids if gid not in valid_ids]
    if invalid_ids:
        logger.warning(
            "Invalid GPU IDs (device_count=%d): %s. Valid IDs: %s",
            device_count, invalid_ids, valid_ids
        )
    
    return valid_ids
