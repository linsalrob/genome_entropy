"""Multi-GPU asynchronous encoding for protein to 3Di conversion."""

import asyncio
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterator, List, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from ..errors import EncodingError
from ..logging_config import get_logger
from .gpu_utils import discover_available_gpus, select_device_for_gpu, validate_gpu_availability
from .types import IndexedSeq

logger = get_logger(__name__)


class MultiGPUEncoder:
    """Manages multi-GPU encoding of amino acid sequences to 3Di tokens.
    
    This class distributes encoding batches across multiple GPUs using asyncio
    for parallel processing. It handles GPU allocation, load balancing, and
    error recovery.
    """
    
    def __init__(
        self,
        model_name: str,
        encoder_class: type,
        gpu_ids: Optional[List[int]] = None,
    ):
        """Initialize multi-GPU encoder.
        
        Args:
            model_name: HuggingFace model identifier
            encoder_class: Encoder class to instantiate (e.g., ProstT5ThreeDiEncoder)
            gpu_ids: List of GPU IDs to use. If None, auto-discover available GPUs.
                    If empty list or None after discovery, falls back to single GPU.
        """
        self.model_name = model_name
        self.encoder_class = encoder_class
        
        # Discover and validate GPUs
        if gpu_ids is None:
            gpu_ids = discover_available_gpus()
        
        # Validate the GPU IDs
        self.gpu_ids = validate_gpu_availability(gpu_ids) if gpu_ids else []
        
        # Create encoders for each GPU
        self.encoders: List[Any] = []
        if self.gpu_ids:
            logger.info("Initializing multi-GPU encoding with %d GPU(s): %s", 
                       len(self.gpu_ids), self.gpu_ids)
            for gpu_id in self.gpu_ids:
                device = select_device_for_gpu(gpu_id)
                encoder = encoder_class(model_name=model_name, device=device)
                self.encoders.append(encoder)
                logger.info("Created encoder for %s", device)
        else:
            # Fallback to single GPU/CPU encoder
            logger.info("No GPUs available, falling back to single device encoder")
            encoder = encoder_class(model_name=model_name, device=None)
            self.encoders.append(encoder)
    
    @property
    def num_gpus(self) -> int:
        """Number of GPUs being used."""
        return len(self.encoders)
    
    def is_multi_gpu(self) -> bool:
        """Check if using multiple GPUs."""
        return len(self.encoders) > 1
    
    async def encode_batch_async(
        self,
        encoder_idx: int,
        batch: List[IndexedSeq],
    ) -> Tuple[List[int], List[str]]:
        """Encode a single batch on a specific GPU asynchronously.
        
        Args:
            encoder_idx: Index of encoder/GPU to use
            batch: List of IndexedSeq objects to encode
            
        Returns:
            Tuple of (original_indices, encoded_3di_sequences)
        """
        encoder = self.encoders[encoder_idx]
        gpu_id = self.gpu_ids[encoder_idx] if self.gpu_ids else None
        
        batch_seqs = [x.seq for x in batch]
        batch_idxs = [x.idx for x in batch]
        
        logger.info(
            "GPU %s: Encoding batch with %d sequences (total len: %d)",
            gpu_id if gpu_id is not None else "default",
            len(batch_seqs),
            sum(len(s) for s in batch_seqs)
        )
        
        # Run encoding in thread pool to avoid blocking asyncio
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Use encoder's _encode_batch method directly
            results = await loop.run_in_executor(
                executor,
                encoder._encode_batch,
                batch_seqs
            )
        
        return batch_idxs, results
    
    async def encode_all_batches_async(
        self,
        batches: List[List[IndexedSeq]],
        total_sequences: int,
    ) -> List[str]:
        """Encode all batches across multiple GPUs asynchronously.
        
        Args:
            batches: List of batches to encode
            total_sequences: Total number of sequences
            
        Returns:
            List of encoded 3Di sequences in original input order
            
        Raises:
            EncodingError: If encoding fails
        """
        three_di_sequences: List[str] = [None] * total_sequences  # type: ignore[list-item]
        
        t0 = time.perf_counter()
        total_batches = len(batches)
        
        logger.info(
            "Starting multi-GPU encoding of %d sequences in %d batches across %d GPU(s)",
            total_sequences, total_batches, len(self.encoders)
        )
        
        # Distribute batches across GPUs in round-robin fashion
        tasks = []
        for batch_idx, batch in enumerate(batches):
            encoder_idx = batch_idx % len(self.encoders)
            task = self.encode_batch_async(encoder_idx, batch)
            tasks.append((batch_idx, task))
        
        # Execute all tasks concurrently
        completed = 0
        try:
            for batch_idx, task in tasks:
                # Await each task and update progress
                batch_idxs, batch_results = await task
                
                # Reorder results to match original input order
                for bi, br in zip(batch_idxs, batch_results):
                    three_di_sequences[bi] = br
                
                completed += 1
                elapsed = time.perf_counter() - t0
                avg_batch_time = elapsed / completed
                eta_remaining = avg_batch_time * (total_batches - completed)
                
                logger.info(
                    "Completed batch %d/%d (%.1f%%) - Elapsed: %.1fs, ETA: %.1fs",
                    completed, total_batches,
                    100.0 * completed / total_batches,
                    elapsed, eta_remaining
                )
        
        except Exception as e:
            logger.error("Multi-GPU encoding failed: %s", e, exc_info=True)
            raise EncodingError(f"Multi-GPU encoding failed: {e}") from e
        
        # Check all sequences encoded
        missing = [i for i, v in enumerate(three_di_sequences) if v is None]
        if missing:
            raise RuntimeError(
                f"Missing encodings for {len(missing)} sequences "
                f"(e.g., indices {missing[:10]})"
            )
        
        elapsed_total = time.perf_counter() - t0
        logger.info(
            "Multi-GPU encoding complete! Encoded %d sequences in %.1fs (%.2f seqs/sec)",
            total_sequences, elapsed_total,
            total_sequences / elapsed_total if elapsed_total > 0 else 0
        )
        
        return three_di_sequences
    
    def encode_multi_gpu(
        self,
        aa_sequences: List[str],
        token_budget_batches_fn: Callable[[List[str], int], Iterator[Any]],
        encoding_size: int,
    ) -> List[str]:
        """Encode sequences using multiple GPUs.
        
        This is a synchronous wrapper around the async encoding method.
        
        Args:
            aa_sequences: List of preprocessed amino acid sequences
            token_budget_batches_fn: Function to create batches under token budget
            encoding_size: Maximum size (approx. amino acids) per batch
            
        Returns:
            List of 3Di token sequences (one per input sequence)
        """
        # Create batches
        batches_iter = token_budget_batches_fn(aa_sequences, encoding_size)
        batches = list(batches_iter)
        
        total_sequences = len(aa_sequences)
        
        # Load models for all encoders
        logger.info("Loading models on all GPUs...")
        for encoder in self.encoders:
            encoder._load_model()
        
        # Run async encoding
        if self.is_multi_gpu():
            logger.info("Using multi-GPU parallel encoding")
            result = asyncio.run(
                self.encode_all_batches_async(batches, total_sequences)
            )
        else:
            logger.info("Using single-GPU sequential encoding")
            # Fall back to sequential processing for single GPU
            result = self._encode_single_gpu_sequential(batches, total_sequences)
        
        return result
    
    def _encode_single_gpu_sequential(
        self,
        batches: List[List[IndexedSeq]],
        total_sequences: int,
    ) -> List[str]:
        """Fallback to sequential single-GPU encoding.
        
        Args:
            batches: List of batches to encode
            total_sequences: Total number of sequences
            
        Returns:
            List of encoded 3Di sequences in original input order
        """
        three_di_sequences: List[str] = [None] * total_sequences  # type: ignore[list-item]
        encoder = self.encoders[0]
        
        logger.info("Encoding %d sequences in %d batches (single GPU)", 
                   total_sequences, len(batches))
        
        t0 = time.perf_counter()
        for batch_idx, batch in enumerate(batches, 1):
            batch_seqs = [x.seq for x in batch]
            batch_idxs = [x.idx for x in batch]
            
            results = encoder._encode_batch(batch_seqs)
            
            for bi, br in zip(batch_idxs, results):
                three_di_sequences[bi] = br
            
            elapsed = time.perf_counter() - t0
            avg_time = elapsed / batch_idx
            eta = avg_time * (len(batches) - batch_idx)
            
            logger.info(
                "Batch %d/%d complete - Elapsed: %.1fs, ETA: %.1fs",
                batch_idx, len(batches), elapsed, eta
            )
        
        # Check all sequences encoded
        missing = [i for i, v in enumerate(three_di_sequences) if v is None]
        if missing:
            raise RuntimeError(
                f"Missing encodings for {len(missing)} sequences"
            )
        
        return three_di_sequences
